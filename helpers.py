import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from functools import lru_cache
from filterpy.kalman import KalmanFilter
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import t as student_t
from scipy.optimize import minimize_scalar


@lru_cache
def download_close_data(ticker, start_date, end_date, interval="1d"):
    """
    Downloads adjusted close price data for a given ticker.
    """
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    # Ensure we are working with a Series of Adjusted Close prices
    # Note: Recent yfinance versions may return a MultiIndex if multiple tickers are passed
    if 'Adj Close' in df.columns:
        close_data = df['Adj Close']
    else:
        close_data = df['Close']

    return close_data


def log_returns(prices: pd.Series[np.float64]) -> pd.Series[np.float64]:
    return np.log(prices / prices.shift(1))


class RobustKalmanFilter(KalmanFilter):
    def __init__(self, dim_x, dim_z, nu=5, iterations=5):
        super().__init__(dim_x, dim_z)
        self.nu = nu          # Degrees of freedom
        self.iterations = iterations
        self.lam = 1.0        # The outlier weighting factor (lambda)

    def update_robust(self, z, R=None, H=None):
        R_orig = R if R is not None else self.R
        H_local = H if H is not None else self.H

        z = np.atleast_2d(z)
        if z.shape[0] == 1 and self.dim_z > 1:
            z = z.T

        x_prior = self.x.copy()
        P_prior = self.P.copy()

        lam_k = 1.0

        # Variational Bayesian Iterative Loop
        for _ in range(self.iterations):
            R_scaled = R_orig / lam_k

            # Standard Kalman Gain calculation using scaled R
            S = H_local @ P_prior @ H_local.T + R_scaled
            K = P_prior @ H_local.T @ np.linalg.inv(S)

            # Temporary posterior estimate
            x_post = x_prior + K @ (z - H_local @ x_prior)
            P_post = (np.eye(self.dim_x) - K @ H_local) @ P_prior

            # Update Lambda (Weighting)
            # Calculate the expected value of the squared residual
            # This is the Mahalanobis distance plus a correction for P
            innovation = z - H_local @ x_post
            delta_sq = (innovation.T @ np.linalg.inv(R_orig) @ innovation +
                        np.trace(H_local.T @ np.linalg.inv(R_orig) @ H_local @ P_post))

            lam_k = (self.nu + self.dim_z) / (self.nu + delta_sq)

        # Finalize the state
        self.lam = float(lam_k[0, 0])
        super().update(z, R_orig / self.lam, H_local)


class DiscreteHMM:
    def __init__(self, n_states, n_observations):
        self.N = n_states
        self.M = n_observations

        # Initialize parameters randomly and normalize
        self.A = np.random.dirichlet([1.0] * self.N, size=self.N)
        self.B = np.random.dirichlet([1.0] * self.M, size=self.N)
        self.pi = np.random.dirichlet([1.0] * self.N)

    def _forward(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.N))
        scales = np.zeros(T)

        # Initialization
        alpha[0] = self.pi * self.B[:, obs[0]]
        scales[0] = 1.0 / np.sum(alpha[0])
        alpha[0] *= scales[0]

        # Induction
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.dot(alpha[t-1], self.A[:, j]) * self.B[j, obs[t]]
            scales[t] = 1.0 / np.sum(alpha[t])
            alpha[t] *= scales[t]

        return alpha, scales

    def _backward(self, obs, scales):
        T = len(obs)
        beta = np.zeros((T, self.N))

        # Initialization
        beta[T-1] = np.ones(self.N) * scales[T-1]

        # Induction
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs[t+1]] * beta[t+1])
            beta[t] *= scales[t]

        return beta

    def fit(self, obs, max_iter=100, tol=1e-6):
        T = len(obs)
        for i in tqdm(range(max_iter)):
            # E-Step
            alpha, scales = self._forward(obs)
            beta = self._backward(obs, scales)

            # Compute Gamma and Xi
            gamma = (alpha * beta) / np.sum(alpha * beta, axis=1, keepdims=True)

            xi = np.zeros((T-1, self.N, self.N))
            for t in range(T-1):
                denominator = np.sum(alpha[t].reshape(-1, 1) * self.A * self.B[:, obs[t+1]].reshape(1, -1) * beta[t+1].reshape(1, -1))
                for i_s in range(self.N):
                    numerator = alpha[t, i_s] * self.A[i_s, :] * \
                                self.B[:, obs[t+1]] * beta[t+1]
                    xi[t, i_s, :] = numerator / denominator

            # M-Step
            new_pi = gamma[0]
            new_A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0).reshape(-1, 1)

            new_B = np.zeros((self.N, self.M))
            for k in range(self.M):
                mask = (obs == k)
                new_B[:, k] = np.sum(gamma[mask], axis=0) / np.sum(gamma, axis=0)

            # Check convergence (simplified)
            change = np.max(np.abs(self.A - new_A)) + np.max(np.abs(self.B - new_B))
            self.A, self.B, self.pi = new_A, new_B, new_pi

            if change < tol:
                break

    def viterbi(self, obs):
        T = len(obs)
        # Work in log space to prevent underflow
        log_A = np.log(self.A + 1e-12)
        log_B = np.log(self.B + 1e-12)
        log_pi = np.log(self.pi + 1e-12)

        # viterbi_score[t, i] is the max log-probability of ending in state i at time t
        viterbi_score = np.zeros((T, self.N))
        # backpointer stores the index of the previous state that led to the max score
        backpointer = np.zeros((T, self.N), dtype=int)

        # Initialization
        viterbi_score[0] = log_pi + log_B[:, obs[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.N):
                # Calculate (prob of being in state i at t-1) + (transition i -> j)
                probabilities = viterbi_score[t-1] + log_A[:, j]
                backpointer[t, j] = np.argmax(probabilities)
                viterbi_score[t, j] = np.max(probabilities) + log_B[j, obs[t]]

        # Path backtracking
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(viterbi_score[T-1])

        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]

        return best_path


class GaussianHMM:
    def __init__(self, n_states, obs=None):
        self.N = n_states
        self.A = np.random.dirichlet([10.0] * self.N, size=self.N) # High diagonal bias
        self.pi = np.random.dirichlet([1.0] * self.N)

        if obs is not None:
            # Use data quantiles to separate initial means
            self.means = np.linspace(np.percentile(obs, 5),
                                    np.percentile(obs, 95),
                                    self.N)
            # Use the global variance as a starting point for both
            self.vars = np.full(self.N, np.var(obs))
        else:
            self.means = np.zeros(self.N)
            self.vars = np.ones(self.N)

    def _get_emission_probs(self, obs_t):
        """Calculates PDF values for an observation across all states."""
        # Using scipy.stats.norm.pdf for numerical precision
        return norm.pdf(obs_t, loc=self.means, scale=np.sqrt(self.vars))

    def _forward(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.N))
        scales = np.zeros(T)

        alpha[0] = self.pi * self._get_emission_probs(obs[0])
        scales[0] = 1.0 / (np.sum(alpha[0]) + 1e-12)
        alpha[0] *= scales[0]

        for t in range(1, T):
            emissions = self._get_emission_probs(obs[t])
            for j in range(self.N):
                alpha[t, j] = np.dot(alpha[t-1], self.A[:, j]) * emissions[j]
            scales[t] = 1.0 / (np.sum(alpha[t]) + 1e-12)
            alpha[t] *= scales[t]

        return alpha, scales

    def _backward(self, obs, scales):
        T = len(obs)
        beta = np.zeros((T, self.N))

        beta[T-1] = np.ones(self.N) * scales[T-1]

        for t in range(T-2, -1, -1):
            emissions = self._get_emission_probs(obs[t+1])
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * emissions * beta[t+1])
            beta[t] *= scales[t]

        return beta

    def fit(self, obs, max_iter=100, tol=1e-6):
        T = len(obs)
        for it in tqdm(range(max_iter)):
            # E-Step
            alpha, scales = self._forward(obs)
            beta = self._backward(obs, scales)

            # Compute Gamma (State Posterior)
            gamma = (alpha * beta) / (np.sum(alpha * beta, axis=1, keepdims=True) + 1e-12)

            # Compute Xi (Transition Posterior)
            xi = np.zeros((T-1, self.N, self.N))
            for t in range(T-1):
                emissions = self._get_emission_probs(obs[t+1])
                denom = np.sum(alpha[t].reshape(-1, 1) * self.A * emissions.reshape(1, -1) * beta[t+1].reshape(1, -1))
                for i in range(self.N):
                    xi[t, i, :] = (alpha[t, i] * self.A[i, :] * emissions * beta[t+1]) / (denom + 1e-12)

            # M-Step: Parameters update
            new_pi = gamma[0]
            new_A = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0).reshape(-1, 1) + 1e-12)

            # Update Means and Variances using weighted averages
            new_means = np.zeros(self.N)
            new_vars = np.zeros(self.N)
            for j in range(self.N):
                gamma_sum = np.sum(gamma[:, j])
                new_means[j] = np.sum(gamma[:, j] * obs) / (gamma_sum + 1e-12)
                new_vars[j] = np.sum(gamma[:, j] * (obs - new_means[j])**2) / (gamma_sum + 1e-12)

            # Prevent variance collapse (numerical stability)
            new_vars = np.maximum(new_vars, 1e-6)

            # Check Convergence
            change = np.max(np.abs(self.means - new_means)) + np.max(np.abs(self.A - new_A))
            self.A, self.pi, self.means, self.vars = new_A, new_pi, new_means, new_vars

            if change < tol:
                break

    def viterbi(self, obs):
        T = len(obs)
        log_A = np.log(self.A + 1e-12)
        log_pi = np.log(self.pi + 1e-12)

        viterbi_score = np.zeros((T, self.N))
        backpointer = np.zeros((T, self.N), dtype=int)

        viterbi_score[0] = log_pi + np.log(self._get_emission_probs(obs[0]) + 1e-12)

        for t in range(1, T):
            emissions = self._get_emission_probs(obs[t])
            log_emissions = np.log(emissions + 1e-12)
            for j in range(self.N):
                probs = viterbi_score[t-1] + log_A[:, j]
                backpointer[t, j] = np.argmax(probs)
                viterbi_score[t, j] = np.max(probs) + log_emissions[j]

        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(viterbi_score[T-1])
        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]

        return best_path


class MultivariateGaussianHMM:
    def __init__(self, n_states, n_features, obs=None):
        self.N = n_states
        self.D = n_features
        self.A = np.random.dirichlet([10.0] * self.N, size=self.N)
        self.pi = np.random.dirichlet([1.0] * self.N)

        if obs is not None:
            self.means = np.zeros((self.N, self.D))
            # Distribute initial means across the quantiles of each feature dimension
            for d in range(self.D):
                self.means[:, d] = np.linspace(np.percentile(obs[:, d], 5),
                                               np.percentile(obs[:, d], 95),
                                               self.N)
            # Use the global covariance matrix as a starting point
            global_cov = np.cov(obs.T) if self.D > 1 else np.array([[np.var(obs)]])
            self.covars = np.array([global_cov.copy() for _ in range(self.N)])
        else:
            self.means = np.zeros((self.N, self.D))
            self.covars = np.array([np.eye(self.D) for _ in range(self.N)])

    def _get_emission_probs(self, obs_t):
        """Calculates multivariate PDF values for an observation vector across all states."""
        emissions = np.zeros(self.N)
        for j in range(self.N):
            emissions[j] = multivariate_normal.pdf(obs_t, mean=self.means[j], cov=self.covars[j], allow_singular=True)
        return np.maximum(emissions, 1e-12)

    def _forward(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.N))
        scales = np.zeros(T)

        alpha[0] = self.pi * self._get_emission_probs(obs[0])
        scales[0] = 1.0 / (np.sum(alpha[0]) + 1e-12)
        alpha[0] *= scales[0]

        for t in range(1, T):
            emissions = self._get_emission_probs(obs[t])
            for j in range(self.N):
                alpha[t, j] = np.dot(alpha[t-1], self.A[:, j]) * emissions[j]
            scales[t] = 1.0 / (np.sum(alpha[t]) + 1e-12)
            alpha[t] *= scales[t]

        return alpha, scales

    def _backward(self, obs, scales):
        T = len(obs)
        beta = np.zeros((T, self.N))
        beta[T-1] = np.ones(self.N) * scales[T-1]

        for t in range(T-2, -1, -1):
            emissions = self._get_emission_probs(obs[t+1])
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * emissions * beta[t+1])
            beta[t] *= scales[t]

        return beta

    def fit(self, obs, max_iter=100, tol=1e-6):
        T = len(obs)
        for it in tqdm(range(max_iter)):
            # E-Step
            alpha, scales = self._forward(obs)
            beta = self._backward(obs, scales)

            # Compute Gamma (State Posterior)
            gamma = (alpha * beta) / (np.sum(alpha * beta, axis=1, keepdims=True) + 1e-12)

            # Compute Xi (Transition Posterior)
            xi = np.zeros((T-1, self.N, self.N))
            for t in range(T-1):
                emissions = self._get_emission_probs(obs[t+1])
                denom = np.sum(alpha[t].reshape(-1, 1) * self.A * emissions.reshape(1, -1) * beta[t+1].reshape(1, -1))
                for i in range(self.N):
                    xi[t, i, :] = (alpha[t, i] * self.A[i, :] * emissions * beta[t+1]) / (denom + 1e-12)

            # M-Step: Parameters update
            new_pi = gamma[0]
            new_A = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0).reshape(-1, 1) + 1e-12)

            # Update Means and Covariances
            new_means = np.zeros((self.N, self.D))
            new_covars = np.zeros((self.N, self.D, self.D))

            for j in range(self.N):
                gamma_sum = np.sum(gamma[:, j])

                # Vectorized mean update
                new_means[j] = np.sum(gamma[:, j, np.newaxis] * obs, axis=0) / (gamma_sum + 1e-12)

                # Covariance update via weighted outer product
                diff = obs - new_means[j]
                weighted_diff = gamma[:, j, np.newaxis] * diff
                new_covars[j] = np.dot(weighted_diff.T, diff) / (gamma_sum + 1e-12)

                # Ridge regularization for numerical stability (prevents singular matrices)
                new_covars[j] += np.eye(self.D) * 1e-6

            # Check Convergence
            change = np.max(np.abs(self.means - new_means)) + np.max(np.abs(self.A - new_A))
            self.A, self.pi, self.means, self.covars = new_A, new_pi, new_means, new_covars

            if change < tol:
                break

    def viterbi(self, obs):
        T = len(obs)
        log_A = np.log(self.A + 1e-12)
        log_pi = np.log(self.pi + 1e-12)

        viterbi_score = np.zeros((T, self.N))
        backpointer = np.zeros((T, self.N), dtype=int)

        viterbi_score[0] = log_pi + np.log(self._get_emission_probs(obs[0]))

        for t in range(1, T):
            emissions = self._get_emission_probs(obs[t])
            log_emissions = np.log(emissions)
            for j in range(self.N):
                probs = viterbi_score[t-1] + log_A[:, j]
                backpointer[t, j] = np.argmax(probs)
                viterbi_score[t, j] = np.max(probs) + log_emissions[j]

        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(viterbi_score[T-1])
        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]

        return best_path


class StudentTHMM:
    def __init__(self,
                 n_states,
                 obs=None,
                 A=None,
                 means=None,
                 vari=None,
                 df_lower_bounds=None,
                 pi=None):

        self.N = n_states

        if df_lower_bounds is not None:
            self.df_lower_bounds = df_lower_bounds
        else:
            self.df_lower_bounds = np.ones(shape=(self.N)) * 2.1

        # High diagonal bias helps persistence in financial regimes
        if A is not None:
            self.A = A
        else:
            self.A = np.random.dirichlet([10.0] * self.N, size=self.N)

        if means is not None:
            self.means = means
        else:
            self.means = np.zeros(self.N)

        if vari is not None:
            self.vars = vari
        else:
            self.vars = np.ones(self.N)

        if pi is not None:
            self.pi = pi
        else:
            self.pi = np.random.dirichlet([1.0] * self.N)

        if obs is not None:
            # Data-driven initialization
            self.means = np.linspace(np.percentile(obs, 5),
                                    np.percentile(obs, 95),
                                    self.N)
            self.vars = np.full(self.N, np.var(obs))
            # Initialize degrees of freedom (nu) between 3 and 6 for fat tails
            self.df = np.full(self.N, 4.0)
        else:
            self.df = np.full(self.N, 4.0)

    def _get_emission_probs(self, obs_t):
        # Update this to use the new alias
        return student_t.pdf(obs_t, df=self.df, loc=self.means, scale=np.sqrt(self.vars))

    def _forward(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.N))
        scales = np.zeros(T)

        alpha[0] = self.pi * self._get_emission_probs(obs[0])
        scales[0] = 1.0 / (np.sum(alpha[0]) + 1e-12)
        alpha[0] *= scales[0]

        for t in range(1, T):
            emissions = self._get_emission_probs(obs[t])
            for j in range(self.N):
                alpha[t, j] = np.dot(alpha[t-1], self.A[:, j]) * emissions[j]
            scales[t] = 1.0 / (np.sum(alpha[t]) + 1e-12)
            alpha[t] *= scales[t]

        return alpha, scales

    def _backward(self, obs, scales):
        T = len(obs)
        beta = np.zeros((T, self.N))
        beta[T-1] = np.ones(self.N) * scales[T-1]

        for t in range(T-2, -1, -1):
            emissions = self._get_emission_probs(obs[t+1])
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * emissions * beta[t+1])
            beta[t] *= scales[t]

        return beta

    def fit(self, obs, max_iter=100, tol=1e-6):
        T = len(obs)
        for it in range(max_iter):
            # E-Step
            alpha, scales = self._forward(obs)
            beta = self._backward(obs, scales)

            # State Posterior (Gamma) and Transition Posterior (Xi)
            gamma = (alpha * beta) / (np.sum(alpha * beta, axis=1, keepdims=True) + 1e-12)

            xi = np.zeros((T-1, self.N, self.N))
            for t in range(T-1):
                emissions = self._get_emission_probs(obs[t+1])
                denom = np.sum(alpha[t].reshape(-1, 1) * self.A * emissions.reshape(1, -1) * beta[t+1].reshape(1, -1))
                for i in range(self.N):
                    xi[t, i, :] = (alpha[t, i] * self.A[i, :] * emissions * beta[t+1]) / (denom + 1e-12)

            # M-Step: pi and A updates
            new_pi = gamma[0]
            new_A = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0).reshape(-1, 1) + 1e-12)

            # M-Step: Student's t parameters (ECM algorithm)
            new_means = np.zeros(self.N)
            new_vars = np.zeros(self.N)
            new_df = np.zeros(self.N)

            for j in range(self.N):
                # Calculate auxiliary weight u based on distance from current mean
                delta_sq = ((obs - self.means[j])**2) / (self.vars[j] + 1e-12)
                u = (self.df[j] + 1) / (self.df[j] + delta_sq)

                # Update Mean: weighted by both Gamma and u
                weights_u = gamma[:, j] * u
                new_means[j] = np.sum(weights_u * obs) / (np.sum(weights_u) + 1e-12)

                # Update Variance: weighted by Gamma and u
                new_vars[j] = np.sum(gamma[:, j] * u * (obs - new_means[j])**2) / (np.sum(gamma[:, j]) + 1e-12)

                def obj(nu):
                    return -np.sum(gamma[:, j] * student_t.logpdf(obs, df=nu, loc=new_means[j], scale=np.sqrt(new_vars[j])))

                res = minimize_scalar(
                    obj,
                    bounds=(self.df_lower_bounds[j], 50),
                    method="bounded"
                )
                new_df[j] = res.x

            new_vars = np.maximum(new_vars, 1e-6)

            # Check Convergence
            change = np.max(np.abs(self.means - new_means)) + np.max(np.abs(self.A - new_A))
            self.A, self.pi, self.means, self.vars, self.df = new_A, new_pi, new_means, new_vars, new_df

            if change < tol:
                break

    def fit_tie_mean(self, obs, max_iter=100, tol=1e-6):
        T = len(obs)
        for it in range(max_iter):
            # E-Step
            alpha, scales = self._forward(obs)
            beta = self._backward(obs, scales)

            # State Posterior (Gamma) and Transition Posterior (Xi)
            gamma = (alpha * beta) / (np.sum(alpha * beta, axis=1, keepdims=True) + 1e-12)

            xi = np.zeros((T-1, self.N, self.N))
            for t in range(T-1):
                emissions = self._get_emission_probs(obs[t+1])
                denom = np.sum(alpha[t].reshape(-1, 1) * self.A * emissions.reshape(1, -1) * beta[t+1].reshape(1, -1))
                for i in range(self.N):
                    xi[t, i, :] = (alpha[t, i] * self.A[i, :] * emissions * beta[t+1]) / (denom + 1e-12)

            # M-Step Part 1: pi and A updates
            new_pi = gamma[0]
            new_A = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0).reshape(-1, 1) + 1e-12)

            # M-Step Part 2: Tying the Means
            total_weighted_sum = 0
            total_weight = 0

            # We need an intermediate step to get the u weights for all states
            u_weights = np.zeros((T, self.N))
            for j in range(self.N):
                delta_sq = ((obs - self.means[j])**2) / (self.vars[j] + 1e-12)
                u_weights[:, j] = (self.df[j] + 1) / (self.df[j] + delta_sq)

                # Aggregate for the tied mean
                total_weighted_sum += np.sum(gamma[:, j] * u_weights[:, j] * obs)
                total_weight += np.sum(gamma[:, j] * u_weights[:, j])

            # This is the single drift value shared by all regimes
            tied_mean = total_weighted_sum / (total_weight + 1e-12)

            new_means = np.full(self.N, tied_mean)
            new_vars = np.zeros(self.N)
            new_df = np.zeros(self.N)

            # M-Step Part 3: Update Variance and DF using the Tied Mean
            for j in range(self.N):
                # Update Variance
                # Note: We use the new_means[j] (tied_mean) here
                new_vars[j] = np.sum(gamma[:, j] * u_weights[:, j] * (obs - new_means[j])**2) / (np.sum(gamma[:, j]) + 1e-12)

                # Update Degrees of Freedom
                def obj(nu):
                    return -np.sum(gamma[:, j] * student_t.logpdf(obs, df=nu, loc=new_means[j], scale=np.sqrt(new_vars[j])))

                # Keep your state-specific constraints here (e.g., df_bounds = (15, 50) for j=0)
                res = minimize_scalar(
                    obj,
                    bounds=(self.df_lower_bounds[j], 50),
                    method="bounded"
                )
                new_df[j] = res.x

            new_vars = np.maximum(new_vars, 1e-6)

            # Check Convergence
            change = np.max(np.abs(self.means - new_means)) + np.max(np.abs(self.A - new_A))
            self.A, self.pi, self.means, self.vars, self.df = new_A, new_pi, new_means, new_vars, new_df

            if change < tol:
                break

    def viterbi(self, obs):
        T = len(obs)
        log_A = np.log(self.A + 1e-12)
        log_pi = np.log(self.pi + 1e-12)

        viterbi_score = np.zeros((T, self.N))
        backpointer = np.zeros((T, self.N), dtype=int)

        # Log-space emission probs for Viterbi
        def log_emissions(obs_t):
        # Changed t.logpdf to student_t.logpdf
            return student_t.logpdf(obs_t, df=self.df, loc=self.means, scale=np.sqrt(self.vars))

        viterbi_score[0] = log_pi + log_emissions(obs[0])

        for t in range(1, T):
            le = log_emissions(obs[t])
            for j in range(self.N):
                probs = viterbi_score[t-1] + log_A[:, j]
                backpointer[t, j] = np.argmax(probs)
                viterbi_score[t, j] = np.max(probs) + le[j]

        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(viterbi_score[T-1])
        for t in range(T-2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]

        return best_path