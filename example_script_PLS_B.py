from utils import *
from sklearn.preprocessing import StandardScaler

X, y, split = load_data("data.csv")

X_test, y_test = X[split == "test"], y[split == "test"]
X_train, y_train = (X[split != "test"], y[split != "test"])

imp = Imputer("min").fit(X_train, y_train)
X_train, y_train = imp.transform(X_train, y_train)
X_test, y_test = imp.transform(X_test, y_test)

ss = StandardScaler().fit(X_train, y_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

model = PLS(
    bootstrap = True,
    n_components = 30,
    ).fit(X_train, y_train)

r2_test = model.score(X_test, y_test, plot = True)
plt.savefig("r2_PLS_B.svg")

model._get_feature_importances(X, y_test, print_=True, plot=True)
plt.savefig("feat_importance_PLS_B.svg")
