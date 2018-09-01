import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold

from efs.evolutionary_feature_synthesis import EFSRegressor

seed = 432
splits = 6


def target(x):
    return x**2


X = np.linspace(-100, 100, 100, endpoint=True)
y = target(X)
X = np.expand_dims(X, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
efs = EFSRegressor(verbose=4, max_gens=1, normalize=True, num_additions=2, splits=splits, seed=seed)
efs.fit(X_train, y_train)
score = efs.score(X_test, y_test)
print('Score: ' + str(score))


cv = KFold(n_splits=splits, random_state=seed)
en = ElasticNetCV(l1_ratio=1, selection='random', cv=cv, random_state=seed, normalize=False)
en.fit(X_train, y_train)
score = en.score(X_test, y_test)
print('Score: ' + str(score))
