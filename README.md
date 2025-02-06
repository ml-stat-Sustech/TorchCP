<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->

<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Latest Tag][tag-shield]][tag-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
<!-- 
***[![MIT License][license-shield]][license-url]
-->

<!-- PROJECT LOGO -->

<br />
<div align="center">
  <!-- <a href="https://github.com/microsoft/promptbench">
    <img src="imgs/promptbench_logo.png" alt="Logo" width="300">
  </a> -->
<img src="https://raw.githubusercontent.com/ml-stat-Sustech/TorchCP/master/logo.png" width="300px" />
<!-- <h3 align="center">USB</h3> -->
<!-- ![TorchCP_logo](./logo.png) -->
<p align="center">
    <strong>TorchCP</strong>: A Python toolbox for Conformal Prediction in Deep Learning.
    <br />
    <a href="https://arxiv.org/abs/2402.12683">Technical Report</a>
    ·
    <a href="https://torchcp.readthedocs.io/en/latest/">Documentation</a>
  </p>
</div>


TorchCP is a Python toolbox for conformal prediction research on deep learning models, built on the PyTorch Library with
strong GPU acceleration. In the toolbox, we implement representative methods (including posthoc and training methods)
for many tasks of conformal prediction, including: Classification, Regression, Graph Node Classification, and LLM. We
build the basic framework of TorchCP based on [`AdverTorch`](https://github.com/BorealisAI/advertorch/tree/master). This
codebase is still under construction and maintained by [`Hongxin Wei`](https://hongxin001.github.io/)'s research group
at SUSTech. Comments, issues, contributions, and collaborations are all welcomed!

# Updates of New Version (1.0.0)

- Added new score functions and training methods for classification, including KNN, TOPK, C-Adapter, and ConfTS.
- Introduced CP algorithms for graph node classification, such as DAPS, SNAPS, and NAPS.
- Added new conformal algorithms for regression, including CQRFM, CQRR, CQRM, and Ensemble CP.
- Introduced CP algorithms for LLMs.
- Added unit-test and examples.
- Optimized the form of prediction sets to improve the computational efficiency.
- Refactored the module design of Regression to improve the scalability.

# Overview

TorchCP has implemented the following methods:

## Classification

| Year | Title                                                                                                                                                                        | Venue                | Code Link                                                                         | Implementation                                      |
|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------|
| 2024 | [**C-Adapter: Adapting Deep Classifiers for Efficient Conformal Prediction Sets**](https://arxiv.org/abs/2410.09408)                                                         | Arxiv                |                                                                                   | classification.loss.cd_loss                         |
| 2024 | [**Delving into temperature scaling for adaptive conformal prediction**](https://arxiv.org/abs/2402.04344)                                                                   | Arxiv                |                                                                                   | classification.loss.confts                          |
| 2024 | [**Conformal Prediction for Deep Classifier via Label Ranking**](https://arxiv.org/abs/2310.06430)                                                                           | ICML'24              | [Link](https://github.com/ml-stat-Sustech/conformal_prediction_via_label_ranking) | classification.score.saps                           |
| 2023 | [**Class-Conditional Conformal Prediction with Many Classes**](https://arxiv.org/abs/2306.09335)                                                                             | NeurIPS'23           | [Link](https://github.com/tiffanyding/class-conditional-conformal)                | classification.predictor.cluster                    |
| 2023 | [**Conformal Prediction Sets for Ordinal Classification**](https://proceedings.neurips.cc/paper_files/paper/2023/file/029f699912bf3db747fe110948cc6169-Paper-Conference.pdf) | NeurIPS'23           |                                                                                   | classification.trainer.ordinal                      |
| 2022 | [**Training Uncertainty-Aware Classifiers with Conformalized Deep Learning**](https://arxiv.org/abs/2205.05878)                                                              | NeurIPS'22           | [Link](https://github.com/bat-sheva/conformal-learning)                           | classification.loss.uncertainty_aware               |
| 2022 | [**Learning Optimal Conformal Classifiers**](https://arxiv.org/abs/2110.09192)                                                                                               | ICLR'22              | [Link](https://github.com/google-deepmind/conformal_training/tree/main)           | classification.loss.conftr                          |       
| 2021 | [**Uncertainty Sets for Image Classifiers using Conformal Prediction**](https://arxiv.org/abs/2009.14193       )                                                             | ICLR'21              | [Link](https://github.com/aangelopoulos/conformal_classification)                 | classification.score.raps classification.score.topk |
| 2020 | [**Classification with Valid and Adaptive Coverage**](https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf)                             | NeurIPS'20           | [Link](https://github.com/msesia/arc)                                             | classification.score.aps                            |
| 2019 | [**Conformal Prediction Under Covariate Shift**](https://arxiv.org/abs/1904.06019)                                                                                           | NeurIPS'19           | [Link](https://github.com/ryantibs/conformal/)                                    | classification.predictor.weight                     |
| 2016 | [**Least Ambiguous Set-Valued Classifiers with Bounded Error Levels**](https://arxiv.org/abs/1609.00451)                                                                     | JASA                 |                                                                                   | classification.score.thr                            |
| 2016 | [**Hedging Predictions in Machine Learning**](https://ieeexplore.ieee.org/document/8129828)                                                                                  | The Computer Journal |                                                                                   | classification.score.knn                            |
| 2015 | [**Bias reduction through conditional conformal prediction**](https://dl.acm.org/doi/abs/10.3233/IDA-150786)                                                                 | Intell. Data Anal.   |                                                                                   | classification.score.margin                         |
| 2012 | [**Conditional Validity of Inductive Conformal Predictors**](https://proceedings.mlr.press/v25/vovk12.html)                                                                  | ACML'12              |                                                                                   | classification.predictor.classwise                  |
| 2007 | [**Hedging Predictions in Machine Learning**](https://ieeexplore.ieee.org/document/8129828)                                                                                  | The Computer Journal |                                                                                   | classification.score.knn                            |

## Regression

| Year | Title                                                                                                                                           | Venue                | Code Link                                              | Implementation                             | Remark              |
|------|-------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|--------------------------------------------------------|--------------------------------------------|---------------------|
| 2023 | [**Conformal Prediction via Regression-as-Classification**](http://etash.me/papers/Bayesian_Conformal_Prediction_through_Memory_Adaptation.pdf) | RegML @ NeurIPS 2023 | [link](https://github.com/EtashGuha/R2CCP/tree/master) | regression.score.r2ccp                     |                     |
| 2021 | [**Adaptive Conformal Inference Under Distribution Shift**](https://arxiv.org/abs/2106.00170)                                                   | NeurIPS'21           | [Link](https://github.com/isgibbs/AdaptiveConformal)   | regression.predictor.aci                   | support time series |
| 2020 | [**A comparison of some conformal quantile regression methods**](https://onlinelibrary.wiley.com/doi/epdf/10.1002/sta4.261)                     | Stat                 | [Link](https://github.com/soroushzargar/DAPS)          | regression.score.cqm regression.score.cqrr |                     |
| 2020 | [**Conformal Prediction Interval for Dynamic Time-Series**](https://proceedings.mlr.press/v139/xu21h.html)                                      | ICML'21              | [Link](https://github.com/hamrel-cxu/EnbPI)            | regression.predictor.ensemble              | support time series |
| 2019 | [**Adaptive, Distribution-Free Prediction Intervals for Deep Networks**](https://proceedings.mlr.press/v108/kivaranovic20a.html)                | AISTATS'19           | [Link](https://github.com/yromano/cqr)                 | regression.score.cqrfm                     |                     |
| 2019 | [**Conformalized Quantile Regression**](https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf)  | NeurIPS'19           | [Link](https://github.com/yromano/cqr)                 | regression.score.cqr                       |                     |
| 2017 | [**Distribution-Free Predictive Inference For Regression**](https://arxiv.org/abs/1604.04173)                                                   | JASA                 | [Link](https://github.com/ryantibs/conformal)          | regression.predictor.split                 |                     |

## Graph

| Year | Title                                                                                                                                                                                                      | Venue      | Code Link                                                  | Implementation                   |
|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|------------------------------------------------------------|----------------------------------|
| 2024 | [**Similarity-Navigated Conformal Prediction for Graph Neural Networks**](https://arxiv.org/abs/2405.14303)                                                                                                | NeuIPS'24  | [Link](https://github.com/janqsong/SNAPS)                  | graph.score.snaps                |
| 2023 | [**Distribution Free Prediction Sets for Node Classification**](https://proceedings.mlr.press/v202/clarkson23a)                                                                                            | ICML'23    | [Link](https://github.com/jase-clarkson/graph_cp)          | graph.predictor.naps |
| 2023 | [**Conformal Prediction Sets for Graph Neural Networks**](https://proceedings.mlr.press/v202/h-zargarbashi23a.html)                                                                                        | ICML'23    | [Link](https://github.com/soroushzargar/DAPS)              | graph.score.daps                 |
| 2023 | [**Uncertainty Quantification over Graph with Conformalized Graph Neural Networks**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html) | NeurIPS'23 | [Link](https://github.com/snap-stanford/conformalized-gnn) | graph.trainer.cfgnn              |

# Language Models

| Year | Title                                                                         | Venue   | Code Link                                                     | Implementation              |
|------|-------------------------------------------------------------------------------|---------|---------------------------------------------------------------|-----------------------------|
| 2023 | [**Conformal Language Modeling**](https://openreview.net/forum?id=pzUhfQ74c5) | ICLR'24 | [Link](https://github.com/Varal7/conformal-language-modeling) | llm.predictor.conformal_llm |

## TODO

TorchCP is still under active development. We will add the following features/items down the road:

| Year | Title                                                                                                           | Venue   | Code                                                                       |
|------|-----------------------------------------------------------------------------------------------------------------|---------|----------------------------------------------------------------------------|
| 2022 | [**Adaptive Conformal Predictions for Time Series**](https://arxiv.org/abs/2202.07282)                          | ICML'22 | [Link](https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries) |
| 2022 | [**Conformal Prediction Sets with Limited False Positives**](https://arxiv.org/abs/2202.07650)                  | ICML'22 | [Link](https://github.com/ajfisch/conformal-fp)                            |
| 2021 | [**Optimized conformal classification using gradient descent approximation**](https://arxiv.org/abs/2105.11255) | Arxiv   |                                                                            |

## Installation

TorchCP is developed with Python 3.9 and PyTorch 2.0.1. To install TorchCP, simply run

```
pip install torchcp
```

To install from TestPyPI server, run

```
pip install --index-url https://test.pypi.org/simple/ --no-deps torchcp
```

## Unit Test

TorchCP achieves 100% unit test coverage. You can use the following command to test the code implementation:

```
pytest --cov=torchcp tests
```

## Examples

Here, we provide a simple example for a classification task, with THR score and SplitPredictor.

```python
from torchcp.classification.score import THR
from torchcp.classification.predictor import SplitPredictor

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
print(f"Coverage Rate: {result_dict['coverage_rate']:.4f}")
print(f"Average Set Size: {result_dict['average_size']:.4f}")

```

You may find more tutorials in [`examples`](https://github.com/ml-stat-Sustech/TorchCP/tree/master/examples) folder.

## License

This project is licensed under the LGPL. The terms and conditions can be found in the LICENSE and LICENSE.GPL files.

## Citation

If you find our repository useful for your research, please consider citing the
following [technical report](https://arxiv.org/abs/2402.12683):

```
@misc{huang2024torchcppythonlibraryconformal,
      title={TorchCP: A Python Library for Conformal Prediction}, 
      author={Jianguo Huang and Jianqing Song and Xuanning Zhou and Bingyi Jing and Hongxin Wei},
      year={2024},
      eprint={2402.12683},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
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

@article{liu2024c,
  title={C-Adapter: Adapting Deep Classifiers for Efficient Conformal Prediction Sets},
  author={Liu, Kangdao and Zeng, Hao and Huang, Jianguo and Zhuang, Huiping and Vong, Chi-Man and Wei, Hongxin},
  journal={arXiv preprint arXiv:2410.09408},
  year={2024}
}
```

## Contributors

* [Hongxin Wei](https://hongxin001.github.io/)
* [Jianguo Huang](https://jianguo99.github.io/)
* [Xuanning Zhou](https://github.com/Shinning-Zhou)
* [Jianqing Song](https://janqsong.github.io/)

[contributors-shield]: https://img.shields.io/github/contributors/ml-stat-Sustech/TorchCP.svg?style=for-the-badge

[contributors-url]: https://github.com/ml-stat-Sustech/TorchCP/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/ml-stat-Sustech/TorchCP.svg?style=for-the-badge

[forks-url]: https://github.com/ml-stat-Sustech/TorchCP/network/members

[stars-shield]: https://img.shields.io/github/stars/ml-stat-Sustech/TorchCP.svg?style=for-the-badge

[stars-url]: https://github.com/ml-stat-Sustech/TorchCP/stargazers

[issues-shield]: https://img.shields.io/github/issues/ml-stat-Sustech/TorchCP.svg?style=for-the-badge

[issues-url]: https://github.com/ml-stat-Sustech/TorchCP/issues

[license-shield]: https://img.shields.io/github/license/ml-stat-Sustech/TorchCP.svg?style=for-the-badge

[license-url]: https://github.com/ml-stat-Sustech/TorchCP/blob/main/LICENSE.txt

[tag-shield]: https://img.shields.io/github/v/tag/ml-stat-Sustech/TorchCP?style=for-the-badge&label=version

[tag-url]: https://github.com/ml-stat-Sustech/TorchCP/tags