# Evolutionary Feature Synthesis 

Evolutionary feature Synthesis is a machine learning technique that evolves the features of a linear model such that the 
score of a model is maximized. For regression this is the r^2 values and for classification this is accuracy. 
Evolutionary Feature Synthesis has been shown to be competitive with the state of the art in machine learning, while retaining
all the nice properties of linear models such as a convex error function and interpretability. 
For details refer to the paper.

Installation
------------
efs is compatible with Python 2.7+
```bash
pip install efs 
```

Example Usage
-------------
```python
import matplotlib.pyplot as plt
import numpy as np
from efs.evolutionary_feature_synthesis import EFSRegressor
from sklearn.model_selection import train_test_split
```
```python
def target(x):
    return x**3 + x**2 + x 
```
Now we'll generate some data on the domain \[-10, 10\].
```python
X = np.linspace(-10, 10, 100, endpoint=True)
y = target(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
```
Finally we'll create and fit the EFSRegressor estimator and check the score.
```python
sr = EFSRegressor()
sr.fit(X_train, y_train)
score = sr.score(X_test, y_test)
print('Score: ' + score)
```
