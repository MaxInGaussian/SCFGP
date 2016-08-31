from SCFGP import GeneralizedPredictor, Gaussian

def load_boston_data(proportion=106./506):
    from sklearn import datasets
    from sklearn import cross_validation
    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    y = y[:, None]
    X = X.astype(np.float64)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=proportion)
    return X_train, y_train, X_test, y_test

trials_per_model = 50
X_train, y_train, X_test, y_test = load_boston_data()
likelihood = Gaussian()
rank = "full"
M = 30
model = Regressor(likelihood, rank, M, fftype=fftype, msg=False)
model.fit(X_train, y_train, X_test, y_test)