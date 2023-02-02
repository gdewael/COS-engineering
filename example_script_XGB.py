from utils import *

X, y, split = load_data("data.csv")

X_test, y_test = X[split == "test"], y[split == "test"]
X_train, y_train = (X[split != "test"], y[split != "test"])

imp = Imputer().fit(X_train, y_train)
X_train, y_train = imp.transform(X_train, y_train)
X_test, y_test = imp.transform(X_test, y_test)

model = XGBshap().fit(X_train, y_train)

r2_test = model.score(X_test, y_test, plot = True)
plt.savefig("r2_XGB.svg")

model._get_feature_importances(X_test, y_test, print_=True, plot=True)
plt.savefig("feat_importance_XGB.svg")
