TorchCP is a Python toolbox for conformal prediction research on deep learning models, using PyTorch. Specifically, this
toolbox has implemented some representative methods (including posthoc and training methods) for
classification and regression tasks. We build the framework of TorchCP based
on [`AdverTorch`](https://github.com/BorealisAI/advertorch/tree/master). This codebase is still under construction and
maintained by [`Hongxin Wei`](https://hongxin001.github.io/)'s research group at SUSTech.
Comments, issues, contributions, and collaborations are all welcomed!

# Overview

TorchCP has implemented the following methods:

## Classification

| Year | Title                                                                                                                                                                        | Venue                | Code Link                                                                         | Implementation                      |
|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|-----------------------------------------------------------------------------------|-------------------------------------|
| 2023 | [**Class-Conditional Conformal Prediction with Many Classes**](https://arxiv.org/abs/2306.09335)                                                                             | NeurIPS'23              | [Link](https://github.com/tiffanyding/class-conditional-conformal)                | classification.predictors.cluster   |
| 2023 | [**Conformal Prediction Sets for Ordinal Classification**](https://proceedings.neurips.cc/paper_files/paper/2023/file/029f699912bf3db747fe110948cc6169-Paper-Conference.pdf) | NeurIPS'23             |                                                                                   | classification.utils.ordinal        |
| 2023 | [**Conformal Prediction for Deep Classifier via Label Ranking**](https://arxiv.org/abs/2310.06430)                                                                           | ICML'24                | [Link](https://github.com/ml-stat-Sustech/conformal_prediction_via_label_ranking) | classification.scores.saps          |
| 2021 | [**Learning Optimal Conformal Classifiers**](https://arxiv.org/abs/2110.09192)                                                                                               | ICLR'22                 | [Link](https://github.com/google-deepmind/conformal_training/tree/main)           | classification.loss.conftr          |       
| 2020 | [**Uncertainty Sets for Image Classifiers using Conformal Prediction**](https://arxiv.org/abs/2009.14193       )                                                             | ICLR'21                 | [Link](https://github.com/aangelopoulos/conformal_classification)                 | classification.scores.raps          |
| 2020 | [**Classification with Valid and Adaptive Coverage**](https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf)                             | NeurIPS'20              | [Link](https://github.com/msesia/arc)                                             | classification.scores.aps           |
| 2019 | [**Conformal Prediction Under Covariate Shift**](https://arxiv.org/abs/1904.06019)                                                                                           | NeurIPS'19            | [Link](https://github.com/ryantibs/conformal/)                                    | classification.predictors.weight    |
| 2016 | [**Least Ambiguous Set-Valued Classifiers with Bounded Error Levels**](https://arxiv.org/abs/1609.00451)                                                                     | JASA                 |                                                                                   | classification.scores.thr           |
| 2015 | [**Bias reduction through conditional conformal prediction**](https://dl.acm.org/doi/abs/10.3233/IDA-150786)                                                                 | Intell. Data Anal.   |                                                                                   | classification.scores.margin        |
| 2013 | [**Applications of Class-Conditional Conformal Predictor in Multi-Class Classification**](https://ieeexplore.ieee.org/document/6784618)                                      | ICMLA                |                                                                                   | classification.predictors.classwise |
| 2007 | [**Hedging Predictions in Machine Learning**](https://ieeexplore.ieee.org/document/8129828)                                                                                  | The Computer Journal |                                                                                   | classification.score.knn            |

## Regression

| Year | Title                                                                                                                                           | Venue                | Code Link                                              | Implementation              | Time Series |
|------|-------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|--------------------------------------------------------|-----------------------------|-------------|
| 2024 | [**Conformal Prediction via Regression-as-Classification**](http://etash.me/papers/Bayesian_Conformal_Prediction_through_Memory_Adaptation.pdf) | RegML @ NeurIPS 2023 | [link](https://github.com/EtashGuha/R2CCP/tree/master) | regression.predictors.r2ccp |             |
| 2022 | [**Ensemble Conformalized Quantile Regression for Probabilistic Time Series Forecasting**](https://ieeexplore.ieee.org/abstract/document/9940232/)                                                         | TNNLS      | [Link](https://github.com/FilippoMB/Ensemble-Conformalized-Quantile-Regression) | regression.predictors.ensemble | |
| 2021 | [**Adaptive Conformal Inference Under Distribution Shift**](https://arxiv.org/abs/2106.00170)                                                   | NeurIPS              | [Link](https://github.com/isgibbs/AdaptiveConformal)   | regression.predictors.aci   | ✅           |
| 2020 | [**A comparison of some conformal quantile regression methods**](https://onlinelibrary.wiley.com/doi/epdf/10.1002/sta4.261)                                                                                | Stat, 2020 | [Link](https://github.com/soroushzargar/DAPS)                                   |regression.predictors.cqrfm/cqrr | |
| 2019 | [**Conformalized Quantile Regression**](https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf)  | NeurIPS              | [Link](https://github.com/yromano/cqr)                 | regression.predictors.cqr   |             |
| 2016 | [**Distribution-Free Predictive Inference For Regression**](https://arxiv.org/abs/1604.04173)                                                   | JASA                 | [Link](https://github.com/ryantibs/conformal)          | regression.predictors.split |             |




## Graph
| Year | Title                                                                                                                                           | Venue                | Code Link                                              | Implementation              |
|------|-------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|--------------------------------------------------------|-----------------------------|
| 2023 | [**Similarity-Navigated Conformal Prediction for Graph Neural Networks**](https://arxiv.org/abs/2405.14303)                                                                                        | NeuIPS'24    |                                  |graph.scores.snaps|
| 2023 | [**Distribution Free Prediction Sets for Node Classification**](https://proceedings.mlr.press/v202/clarkson23a)                                                                                            | ICML'23    | [Link](https://github.com/jase-clarkson/graph_cp)                            | graph.scores.daps|
| 2023 | [**Conformal Prediction Sets for Graph Neural Networks**](https://proceedings.mlr.press/v202/h-zargarbashi23a.html)                                                                                        | ICML'23    | [Link](https://github.com/soroushzargar/DAPS)                                   |graph.predictors.naps_split |
<!-- | 2023 | [**Uncertainty Quantification over Graph with Conformalized Graph Neural Networks**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html) | NeurIPS'23 | [Link](https://github.com/snap-stanford/conformalized-gnn)                      | graph.| -->


# Language Models
| Year | Title                                                                                                                                           | Venue                | Code Link                                              | Implementation              |
|------|-------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|--------------------------------------------------------|-----------------------------|
| 2023 | [**Conformal Language Modeling**](https://openreview.net/forum?id=pzUhfQ74c5)                                                                                                                              | ICLR'24      | [Link](https://github.com/Varal7/conformal-language-modeling)                   | llm.predictors.conformal_llm|



## TODO

TorchCP is still under active development. We will add the following features/items down the road:

| Year | Title                                                                                                                                                                                                      | Venue      | Code                                                                   |
|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------|
| 2022 | [**Training Uncertainty-Aware Classifiers with Conformalized Deep Learning**](https://arxiv.org/abs/2205.05878)                                                                                            | NeurIPS    | [Link](https://github.com/bat-sheva/conformal-learning)                         |
| 2022 | [**Adaptive Conformal Predictions for Time Series**](https://arxiv.org/abs/2202.07282)                                                                                                                     | ICML       | [Link](https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries)      |
| 2022 | [**Conformal Prediction Sets with Limited False Positives**](https://arxiv.org/abs/2202.07650)                                                                                                             | ICML       | [Link](https://github.com/ajfisch/conformal-fp)                                 |
| 2021 | [**Optimized conformal classification using gradient descent approximation**](https://arxiv.org/abs/2105.11255)                                                                                            | Arxiv      |                                                                                 |

## Installation

TorchCP is developed with Python 3.9 and PyTorch 2.0.1. To install TorchCP, simply run

```
pip install torchcp
```

To install from TestPyPI server, run

```
pip install --index-url https://test.pypi.org/simple/ --no-deps torchcp
```

## Examples

Here, we provide a simple example for a classification task, with THR score and SplitPredictor.

```python
from torchcp.classification.scores import THR
from torchcp.classification.predictors import SplitPredictor

# Preparing a calibration data and a test data.
cal_dataloader = ...
test_dataloader = ...
# Preparing a pytorch model
model = ...

model.eval()

# Options of score function: THR, APS, SAPS, RAPS
# Define a conformal prediction algorithm. Optional: SplitPredictor, ClusteredPredictor, ClassWisePredictor
predictor = SplitPredictor(score_function=THR(), model=model)

# Calibrating the predictor with significance level as 0.1
predictor.calibrate(cal_dataloader, alpha=0.1)

#########################################
# Predicting for test instances
########################################
test_instances = ...
predict_sets = predictor.predict(test_instances)
print(predict_sets)

#########################################
# Evaluating the coverage rate and average set size on a given dataset.
########################################
result_dict = predictor.evaluate(test_dataloader)
print(result_dict["Coverage_rate"], result_dict["Average_size"])

```

You may find more tutorials in [`examples`](https://github.com/ml-stat-Sustech/TorchCP/tree/master/examples) folder.

## Documentation

The documentation webpage is on readthedocs https://torchcp.readthedocs.io/en/latest/index.html.

## License

This project is licensed under the LGPL. The terms and conditions can be found in the LICENSE and LICENSE.GPL files.

## Citation

If you find our repository useful for your research, please consider citing the
following [technical report](https://arxiv.org/abs/2402.12683):

```
@misc{wei2024torchcp,
      title={TorchCP: A Library for Conformal Prediction based on PyTorch}, 
      author={Hongxin Wei and Jianguo Huang},
      year={2024},
      eprint={2402.12683},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

We welcome you to cite the following works:

```
@inproceedings{huangconformal,
  title={Conformal Prediction for Deep Classifier via Label Ranking},
  author={Huang, Jianguo and Xi, HuaJun and Zhang, Linjun and Yao, Huaxiu and Qiu, Yue and Wei, Hongxin},
  booktitle={Forty-first International Conference on Machine Learning}
}

@article{xi2024does,
  title={Does Confidence Calibration Help Conformal Prediction?},
  author={Xi, Huajun and Huang, Jianguo and Feng, Lei and Wei, Hongxin},
  journal={arXiv preprint arXiv:2402.04344},
  year={2024}
}
```

## Contributors

* [Hongxin Wei](https://hongxin001.github.io/)
* [Jianguo Huang](https://jianguo99.github.io/)
* [Xuanning Zhou](https://github.com/Shinning-Zhou)




