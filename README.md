TorchCP is a Python toolbox for conformal prediction research on deep learning models, using PyTorch. Specifically, this toolbox has implemented some representative methods (including posthoc and training methods) for
classification and regression tasks. This codebase is still under construction. Comments, issues, contributions, and collaborations are all welcomed!



# Overview
TorchCP has implemented the following methods:
## Classification
 Year | Title                                                                                                                                           | Venue   | Code Link |
|------|-------------------------------------------------------------------------------------------------------------------------------------------------|---------|-------------|
| 2023 | [**Class-Conditional Conformal Prediction with Many Classes**](https://arxiv.org/abs/2306.09335)                                                | NeurIPS | [Link](https://github.com/tiffanyding/class-conditional-conformal) |
| 2023 | [**Conformal Prediction for Deep Classifier via Label Ranking**](https://arxiv.org/abs/2310.06430)                                              | Arxiv   | [Link](https://github.com/ml-stat-Sustech/conformal_prediction_via_label_ranking) |
| 2021 | [**Learning Optimal Conformal Classifiers**](https://arxiv.org/abs/2110.09192)                                                               | ICLR    | [Link](https://github.com/google-deepmind/conformal_training/tree/main) |
| 2020 | [**Uncertainty Sets for Image Classifiers using Conformal Prediction**](https://arxiv.org/abs/2009.14193       )                                | ICLR    | [Link](https://github.com/aangelopoulos/conformal_classification) |
| 2020 | [**Classification with Valid and Adaptive Coverage**](https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf) | NeurIPS | [Link](https://github.com/msesia/arc) |
| 2019 | [**Conformal Prediction Under Covariate Shift**](https://arxiv.org/abs/1904.06019)                                                              | NeurIPS | [Link](https://github.com/ryantibs/conformal/) |
| 2016 | [**Least Ambiguous Set-Valued Classifiers with Bounded Error Levels**](https://arxiv.org/abs/1609.00451)                                        | JASA    | |
| 2013 | [**Applications of Class-Conditional Conformal Predictor in Multi-Class Classification**](https://ieeexplore.ieee.org/document/6784618)         | ICMLA   | |

## Regression
 Year | Title                                                                                                                       | Venue   | Code Link                                            |
|------|-----------------------------------------------------------------------------------------------------------------------------|---------|------------------------------------------------------|
| 2021 | [**Adaptive Conformal Inference Under Distribution Shift**](https://arxiv.org/abs/2106.00170)                               | NeurIPS | [Link](https://github.com/isgibbs/AdaptiveConformal) |
| 2019 | [**Conformalized Quantile Regression**](https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf) | NeurIPS | [Link](https://github.com/yromano/cqr)               |
| 2016 | [**Distribution-Free Predictive Inference For Regression**](https://arxiv.org/abs/1604.04173)                               | JASA    | [Link](https://github.com/ryantibs/conformal)        |



## TODO
TorchCP is still under active development. We will add the following features/items down the road:

 Year | Title                                                                                                      | Venue   | Code Link |
|------|------------------------------------------------------------------------------------------------------------|---------|---------------|
| 2022 | [**Training Uncertainty-Aware Classifiers with Conformalized Deep Learning**](https://arxiv.org/abs/2205.05878) | NeurIPS | [Link](https://github.com/bat-sheva/conformal-learning) |
| 2022 | [**Adaptive Conformal Predictions for Time Series**](https://arxiv.org/abs/2202.07282)                     | ICML    | [Link](https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries) |
| 2022 | [**Predictive Inference with Feature Conformal Prediction**](https://arxiv.org/abs/2210.00173)             | ICLR    | [Link](https://github.com/AlvinWen428/FeatureCP) |
| 2022 | [**Conformal Prediction Sets with Limited False Positives**](https://arxiv.org/abs/2202.07650)             | ICML    | [Link](https://github.com/ajfisch/conformal-fp) |
| 2021 | [**Optimized conformal classification using gradient descent approximation**](https://arxiv.org/abs/2105.11255)                           | Arxiv   | |





## Installation

We developed TorchCP under Python 3.9 and PyTorch 2.0.1. To install TorchCP, simply run

```
pip install torchcp
```

## Examples

```python
from torchcp.classification.scores import THR
from torchcp.classification.predictors import SplitPredictor

cal_dataloader = ...
test_dataloader = ...
model = ...
model.eval()

# define a score function
thr_score_function = THR()

# significance level
alpha = 0.1

# define a conformal prediction algorithm
predictor = SplitPredictor(thr_score_function, model)

# calibration process
predictor.calibrate(cal_dataloader, alpha)

# test examples and return basic metrics
print(predictor.evaluate(test_dataloader))
```

## License

This project is licensed under the LGPL. The terms and conditions can be found in the LICENSE and LICENSE.GPL files.

## Contributors

* [Hongxin Wei](https://hongxin001.github.io/)
* [Jianguo Huang](https://jianguo99.github.io/)


