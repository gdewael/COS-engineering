import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import shap
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression


def load_data(
    path,
    keep_mu=False,
    keep_od=False,
    log_transform=True,
    OD_range=[0.7, 1.3],
    chito_variables=False,
):
    data = pd.read_csv(path, header=[0, 1], index_col=0)
    target = "chitoheptose" if "original" in path else "chitopentose"
    data[data == 1e-6] = np.nan  ## treating them properly as nans
    columns_keep = (
        pd.isnull(data).sum() / data.shape[0] < 0.80
    )  ## flat out removing features for which more than 80% of the samples have a NaN.
    data = data.loc[:, columns_keep]

    ODs = data.iloc[:, data.columns.get_level_values(0) == "OD"]
    data = data.iloc[((OD_range[0] <= ODs) & (ODs <= OD_range[1])).values]

    if not keep_mu:
        data = data.iloc[:, data.columns.get_level_values(0) != "mu"]
    if not keep_od:
        data = data.iloc[:, data.columns.get_level_values(0) != "OD"]

    target_ix = np.array(
        [
            True if target in metabolite else False
            for metabolite in data.columns.get_level_values(0)
        ]
    )
    y = data.iloc[:, target_ix]
    X = data.iloc[:, ~target_ix]

    X.columns = X.columns.get_level_values(0)

    if log_transform:
        y = np.log(y)
        X_temp = X.drop(
            ["OD", "mu", "split"], axis=1, errors="ignore"
        )  # don't log transform od and mu
        X_temp = np.log(X_temp)
        X = pd.concat([X_temp, X.loc[:, X.columns.isin(["OD", "mu", "split"])]], axis=1)

    ## filter out those samples for which chitopentose was not measured
    X = X.iloc[~np.isnan(y.iloc[:, 0]).values]
    y = y.iloc[~np.isnan(y.iloc[:, 0]).values]

    split = X["split"]
    X = X.drop(["split"], axis=1)

    if chito_variables == False:
        X = X[X.columns[~X.columns.str.lower().str.contains("chito")]]

    return X, y, split


class Imputer(object):
    def __init__(self, mode="0.50*min"):
        super().__init__()
        self.mode = mode  # mode should be either "keep", "1e-6", "min", or "0.50*min"

    def transform(self, X, y):
        if self.mode == "keep":
            return X, y
        elif self.mode == "1e-6":
            X[np.isnan(X)] = 1e-6
            return X, y
        elif (
            "min" in self.mode
        ):  # keep note: we fill it with the mean value of the training data
            X = X.fillna(self._mins)
            return X, y

    def fit(self, X, y=None):
        if self.mode == "min":
            self._mins = X.min()
        elif self.mode == "0.50*min":
            self._mins = X.min() * 0.5
        return self


class XGBshap(BaseEstimator):
    def __init__(
        self, n_estimators=500, lr=0.1, sample_samples=0.5, sample_features=0.25
    ):
        super().__init__()
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=2,
            learning_rate=lr,
            subsample=sample_samples,
            colsample_bytree=sample_features,
        )

    def transform(self, X, y):
        raise NotImplementedError

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        self.model.fit(X, y.reshape(-1))
        return self

    def _get_feature_importances(self, X, y, print_=False, plot=False, n_top=None):
        explainer = shap.Explainer(self.model)
        shap_values = explainer(X.values)

        if n_top == None:
            n_top = X.shape[1]

        scores = shap_values.abs.mean(0).values
        if print_ == True:
            print("%-141s %-9s" % ("metabolite", "mean abs shap-value"))
            print("-" * 175)
            for i in np.argsort(scores)[-n_top:][::-1]:
                print("%-140s %9f" % (X.columns[i], scores[i]))

        res = []
        for ix, (v, b, name) in enumerate(
            zip(shap_values.values.T, shap_values.data.T, X.columns)
        ):
            for vv, bb in zip(v, b):
                res.append(
                    [
                        ix,
                        name,
                        vv,
                        bb,
                    ]
                )

        res = pd.DataFrame(res)

        if plot:
            self.res2plot(res, X.columns, n_top)

        return scores, res

    def score(self, X, y, plot=False):
        y_pred = self.model.predict(X.values)
        score = r2_score(y.values.reshape(-1), y_pred)
        if plot:
            range_ = np.arange(y.min().item() - 0.1, y.max().item() + 0.1, 0.05)
            plt.figure()
            plt.plot(range_, range_)

            print("R^2:", np.round(score, 4))
            plt.title("R$^2$: " + str(np.round(score, 4)))
            sns.scatterplot(x=y.values.reshape(-1), y=y_pred.reshape(-1), alpha=0.25)
            plt.xlabel("true (log-)intensity of COS")
            plt.ylabel("predicted (log-)intensity of COS")
        return score

    @staticmethod
    def res2plot(res, column_names, n_top):

        res[4] = res[2].apply(abs)
        grouped = res.groupby([0], axis=0).mean()
        shaps = grouped[4]

        order = np.array(column_names)[np.argsort(shaps)[-n_top:][::-1]]
        res = res[res[1].isin(order)]

        res[0] = res[0].map(
            {t: -ix for ix, t in enumerate(np.argsort(shaps)[-n_top:][::-1])}
        )
        res[0] = res[0] + np.random.randn(res.shape[0]) / 5

        binned = []
        for i in order:
            subset = res[res[1] == i]
            subset[3] = pd.cut(subset[3], 20, labels=False)
            binned.append(subset)

        res = pd.concat(binned, axis=0)
        res = res.sample(frac=0.1)
        plt.figure(figsize=(8, n_top * 0.15))
        plt.axvline(color="black", linestyle=":")
        sns.scatterplot(
            data=res,
            x=2,
            y=0,
            hue=3,
            palette="blend:#1f77b4,#d62728",
            edgecolor="none",
            s=15,
            alpha=0.5,
        )

        plt.xlabel("Shap values")
        plt.ylabel("Metabolite")
        plt.yticks(
            ticks=-np.arange(order.shape[0]),
            labels=[
                (name[:50] + ".." + name[-50:]) if len(name) > 100 else name
                for name in order
            ],
        )
        plt.legend().remove()
        plt.tight_layout()


class PLS(BaseEstimator):
    def __init__(
        self,
        n_components=20,
        bootstrap=False,
        sample_samples=0.5,
        sample_features=0.5,
        n_estimators=250,
    ):
        super().__init__()

        self.model = PLSRegression(n_components=n_components)
        if bootstrap:
            self.model = BaggingRegressor(
                self.model,
                max_samples=sample_samples,
                max_features=sample_features,
                n_estimators=n_estimators,
                verbose=0,
            )
        self.bootstrapped = bootstrap

    def transform(self, X, y):
        raise NotImplementedError

    def fit(self, X, y, print_=True, plot=True):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        self.model.fit(X, y.reshape(-1))
        return self

    def _get_feature_importances(self, X, y, print_=True, plot=True, n_top=None):

        if n_top == None:
            n_top = X.shape[1]

        if self.bootstrapped:
            estimator_coefficients = {k: [] for k in np.arange(X.shape[1])}
            for feat, model in zip(
                self.model.estimators_features_, self.model.estimators_
            ):
                for f, c in zip(feat, model.coef_.reshape(-1)):
                    estimator_coefficients[f].append(c)

            p_vals = []
            scores = []
            for k in np.arange(X.shape[1]):
                p_vals.append(
                    2
                    * np.minimum(
                        (np.array(estimator_coefficients[k]) >= 0).sum()
                        / len(estimator_coefficients[k]),
                        (np.array(estimator_coefficients[k]) <= 0).sum()
                        / len(estimator_coefficients[k]),
                    )
                )
                scores.append(np.mean(estimator_coefficients[k]))

            if print_ == True:
                print(
                    "%-106s %-9s %-9s"
                    % (
                        "metabolite",
                        "p-val (uncorrected)",
                        "mean coeff in PLS-bootstrap",
                    )
                )
                print("-" * 175)
                for i in np.argsort(np.abs(scores))[-n_top:][::-1]:
                    print("%-105s %9f %9f" % (X.columns[i], p_vals[i], scores[i]))

            res = []
            for ix, name in enumerate(X.columns):
                for t in estimator_coefficients[ix]:
                    res.append(
                        [(name[:50] + ".." + name[-50:]) if len(name) > 100 else name, t]
                    )
            res = pd.DataFrame(res)

        else:
            scores = self.model.coef_.reshape(-1)

            if print_ == True:
                print("%-106s %-9s" % ("metabolite", "coeff in PLS"))
                print("-" * 175)
                for i in np.argsort(np.abs(scores))[-n_top:][::-1]:
                    print("%-105s %9f" % (X.columns[i], scores[i]))

            res = []
            for ix, name in enumerate(X.columns):
                res.append(
                    [
                        (name[:50] + ".." + name[-50:]) if len(name) > 100 else name,
                        scores[ix],
                    ]
                )
            res = pd.DataFrame(res)

        order = np.array(X.columns)[np.argsort(np.abs(scores))[-n_top:][::-1]]
        order = np.array(
            [
                (name[:50] + ".." + name[-50:]) if len(name) > 100 else name
                for name in order
            ]
        )
        res = res[res[0].isin(order)]

        if plot:
            self.res2plot(res, order, n_top)

        if self.bootstrapped:
            return scores, order, p_vals, res
        else:
            return scores, order, res

    def score(self, X, y, plot=False):
        y_pred = self.model.predict(X)
        score = r2_score(y.values.reshape(-1), y_pred)
        if plot:
            range_ = np.arange(y.min().item() - 0.1, y.max().item() + 0.1, 0.05)
            plt.figure()
            plt.plot(range_, range_)

            print("R^2:", np.round(score, 4))
            plt.title("R$^2$: " + str(np.round(score, 4)))
            sns.scatterplot(x=y.values.reshape(-1), y=y_pred.reshape(-1), alpha=0.25)
            plt.xlabel("true (log-)intensity of COS")
            plt.ylabel("predicted (log-)intensity of COS")
        return score

    @staticmethod
    def res2plot(res, order, n_top):
        plt.figure(figsize=(8, n_top * 0.15))
        plt.axvline(color="black", linestyle=":")
        if res[0].nunique() == res.shape[0]:
            sns.barplot(data=res, x=1, y=0, order=order)
        else:
            sns.violinplot(data=res, x=1, y=0, order=order)

        plt.xlabel("Coefficients in PLS")
        plt.ylabel("Metabolite")
        plt.tight_layout()