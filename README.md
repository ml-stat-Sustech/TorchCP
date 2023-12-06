DeepCP is a Python toolbox for conformal prediction research on deep learning models. The primary functionalities are implemented in PyTorch. Specifically, DeepCP contains modules of post-hoc methods and training methods for classification problems and regression problems.

## Installation

### Installing DeepCP itself

We developed DeepCP under Python 3.9 and PyTorch 2.0.1. To install DeepCP, simply run

```
pip install deepcp
```

or clone the repo and run
```
python setup.py install
```

To install the package in "editable" mode:
```
pip install -e .
```


## Examples
```python
logits_cal = ...
Y_cal = ...

# init a conformal prediction predictor
predictor = APS()
 
# run a calibration process
predictor.fit(logits_cal,Y_cal,alpha)

# test examples (logits_test,Y_test)
Y_Sets = predictor.predict(logits_test)

# evaluate the prediction sets
metrics = utils.coverage_rate(Y_sets,Y_test)
```


## Coming Soon

DeepCP is still under active development. We will add the following features/items down the road:

* more CP algorithms 
* loss functions for CP
* ...

## License

This project is licensed under the LGPL. The terms and conditions can be found in the LICENSE and LICENSE.GPL files.



## Contributors

* [Hongxin Wei](https://hongxin001.github.io/)
* [Jianguo Huang](https://github.com/Jianguo99)
* [Huajun Xi]
* [Xuanning Zhou]

