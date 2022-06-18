# python-logistic-regression

Logistic Regression Model applied in python

### Getting Started

To use this model, you only need to:

1. Instantiate the model
2. Train the model
3. Predict values with the model

```python
from logistic_regression_model import *

model = LogisticRegressionModel()
model.train(X_train, Y_train, X_test, Y_text) # refer to function for more details
model.predict(X)
```

Please notice that you can also add more functionalities, such as `save()`, `load()` and etc.

### Demo

To see how you can use this model, refer to `example.py`.
