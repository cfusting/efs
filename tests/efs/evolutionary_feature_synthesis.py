import numpy as np

from sklearn.model_selection import train_test_split

from efs.evolutionary_feature_synthesis import EFSRegressor


class TestEFS:

    def test_(self):
        def target(x):
            return x

        X = np.linspace(-10, 10, 1000, endpoint=True)
        y = target(X)
        X = np.expand_dims(X, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        efs = EFSRegressor(verbose=3, max_gens=20, normalize=True, num_additions=20)
        efs.fit(X_train, y_train)
        score = efs.score(X_test, y_test)
        assert score == 0.99
