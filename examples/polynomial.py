import numpy as np

from sklearn.model_selection import train_test_split

from efs.evolutionary_feature_synthesis import EFSRegressor


def target(x):
    return x**2


X = np.linspace(-100, 100, 100, endpoint=True)
y = target(X)
X = np.expand_dims(X, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
efs = EFSRegressor(verbose=3, max_gens=30, normalize=True, num_additions=2, splits=6)
efs.fit(X_train, y_train)
score = efs.score(X_test, y_test)
print('Score: ' + str(score))
