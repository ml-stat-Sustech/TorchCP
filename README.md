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
cal_labels = ...
cal_probailities =  ...

test_labels = ...
test_probailities =  ...

# define a score function
thr_score_function = THR()

# set significance level
alpha = 0.1

predictor = StandardPredictor(thr_score_function)
predictor.fit(cal_probailities, cal_labels, alpha)

# test examples
print("testing examples...")
prediction_sets = []
for index,ele in enumerate(test_probailities):
    prediction_set  = predictor.predict(ele)
    prediction_sets.append(prediction_set)

print("computing metrics...")
metrics = Metrics(["coverage_rate"])
print(metrics.compute(prediction_sets,test_labels))

```


## Coming Soon later

DeepCP is still under active development. We will add the following features/items down the road:

* ClusterCP, Weigthed split CP.
* loss functions for CP
* ...

## License

This project is licensed under the LGPL. The terms and conditions can be found in the LICENSE and LICENSE.GPL files.



## Contributors

* [Hongxin Wei](https://hongxin001.github.io/)
* [Jianguo Huang](https://jianguo99.github.io/)
* [Huajun Xi]
* [Xuanning Zhou]

