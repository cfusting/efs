import numpy as np

from sklearn.model_selection import train_test_split

from efs.evolutionary_feature_synthesis import EFSRegressor


def target(x):
    return x**3 + x**2 + x


X = np.linspace(-10, 10, 100, endpoint=True)

y = target(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
sr = EFSRegressor()
sr.fit(X_train, y_train)
score = sr.score(X_test, y_test)
print('Score: ' + score)
