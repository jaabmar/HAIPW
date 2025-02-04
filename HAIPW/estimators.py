import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_regression


class BaseEstimator:
    def estimate(self, X, Y, T, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class HAIPWEstimator(BaseEstimator):
    def __init__(self, alpha_ridge=0.1, n_folds=5, n_features=10):
        self.alpha_ridge = alpha_ridge
        self.n_folds = n_folds
        self.n_features = n_features

    def compute_aipw(self, X, Y, T):
        kf = KFold(n_splits=self.n_folds)
        psi_aipw = np.zeros(X.shape[0])
        pi = T.mean()

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, T_train = Y[train_idx], T[train_idx]

            mu1_rct = Ridge(alpha=self.alpha_ridge)
            mu0_rct = Ridge(alpha=self.alpha_ridge)

            mu1_rct.fit(X_train[T_train == 1], Y_train[T_train == 1])
            mu0_rct.fit(X_train[T_train == 0], Y_train[T_train == 0])

            psi_aipw[test_idx] = (
                mu1_rct.predict(X_test) - mu0_rct.predict(X_test)
                + (T[test_idx] * (Y[test_idx] - mu1_rct.predict(X_test))) / pi
                - ((1 - T[test_idx]) * (Y[test_idx] - mu0_rct.predict(X_test))) / (1 - pi)
            )
        return psi_aipw

    def estimate(self, X, Y, T, y1_preds, y0_preds, psi_aipw=None):
        # Feature selection
        feature_selector = SelectKBest(score_func=f_regression, k=self.n_features)
        X_selected = feature_selector.fit_transform(X, Y)

        # Compute AIPW if not provided
        if psi_aipw is None:
            psi_aipw = self.compute_aipw(X_selected, Y, T)

        pi = T.mean()

        psi_model_estimates = np.array([
            (T * (Y - y1_hat)) / pi - ((1 - T) * (Y - y0_hat)) / (1 - pi) + (y1_hat - y0_hat)
            for y1_hat, y0_hat in zip(y1_preds, y0_preds)
        ])

        Sigma = np.cov(np.vstack((psi_aipw, psi_model_estimates)))
        lambda_star = self.compute_lambda(Sigma)

        haipw_if = (np.vstack((psi_aipw, psi_model_estimates)) * lambda_star[:, np.newaxis]).sum(axis=0)

        return np.mean(haipw_if), lambda_star.T @ Sigma @ lambda_star

    def compute_lambda(self, Sigma):
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma += 1e-6 * np.eye(Sigma.shape[0])
        Sigma_inv = np.linalg.inv(Sigma)
        ones = np.ones(Sigma.shape[0])
        lambda_star = Sigma_inv @ ones / (ones.T @ Sigma_inv @ ones)
        return lambda_star


class AIPWEstimator(BaseEstimator):
    def __init__(self, alpha_ridge=0.1, n_folds=5, n_features=10):
        self.alpha_ridge = alpha_ridge
        self.n_folds = n_folds
        self.n_features = n_features

    def estimate(self, X, Y, T, *args, **kwargs):
        feature_selector = SelectKBest(score_func=f_regression, k=self.n_features)
        X_selected = feature_selector.fit_transform(X, Y)

        kf = KFold(n_splits=self.n_folds)
        psi_rct = np.zeros(X_selected.shape[0])
        pi = T.mean()

        for train_idx, test_idx in kf.split(X_selected):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            Y_train, T_train = Y[train_idx], T[train_idx]

            mu1_rct = Ridge(alpha=self.alpha_ridge)
            mu0_rct = Ridge(alpha=self.alpha_ridge)

            mu1_rct.fit(X_train[T_train == 1], Y_train[T_train == 1])
            mu0_rct.fit(X_train[T_train == 0], Y_train[T_train == 0])

            psi_rct[test_idx] = (
                mu1_rct.predict(X_test) - mu0_rct.predict(X_test)
                + (T[test_idx] * (Y[test_idx] - mu1_rct.predict(X_test))) / pi
                - ((1 - T[test_idx]) * (Y[test_idx] - mu0_rct.predict(X_test))) / (1 - pi)
            )
        return np.mean(psi_rct), np.var(psi_rct, ddof=1)


class PPIEstimator(BaseEstimator):
    def estimate(self, X, Y, T, f_model):
        sigma_t = np.std(Y[T == 1], ddof=1)
        sigma_c = np.std(Y[T == 0], ddof=1)
        sigma_f = np.std(f_model, ddof=1)

        pi_t = T.mean()
        pi_c = 1 - pi_t

        rho_c = pearsonr(f_model[T == 0], Y[T == 0])[0] if np.sum(T == 0) > 0 else 0.0
        rho_t = pearsonr(f_model[T == 1], Y[T == 1])[0] if np.sum(T == 1) > 0 else 0.0

        optimal_lambda = (pi_c * sigma_t * rho_t + pi_t * sigma_c * rho_c) / (sigma_f + 1e-3)
        ppi_est = np.mean(Y[T == 1] - optimal_lambda * f_model[T == 1]) - np.mean(Y[T == 0] - optimal_lambda * f_model[T == 0])

        sigma_t_sq = np.var(Y[T == 1], ddof=1)
        sigma_c_sq = np.var(Y[T == 0], ddof=1)
        sigma_f_sq = np.var(f_model, ddof=1)
        ppi_var = sigma_t_sq / pi_t + sigma_c_sq / pi_c - optimal_lambda**2 * sigma_f_sq * (1 / pi_c + 1 / pi_t)

        return ppi_est, ppi_var


class DifferenceInMeansEstimator(BaseEstimator):
    def estimate(self, X, Y, T, *args, **kwargs):
        pi = T.mean()
        dm_i = Y * T / pi - Y * (1 - T) / (1 - pi)
        dm_est = dm_i.mean()
        dm_var = np.var(Y[T == 1], ddof=1) / pi + np.var(Y[T == 0], ddof=1) / (1 - pi)

        return dm_est, dm_var
