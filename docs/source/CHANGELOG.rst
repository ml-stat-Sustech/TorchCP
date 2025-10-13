Changelog
=========
1.2.1 (2025-10-13)  
------------------  
* Updated the automatic dependency installation of TorchCP.


1.2.0 (2025-08-26)  
------------------  
* Added Conformal Predictive Distribution in regression task.  
* Added p-value computation in classification predictors and regression predictors.  
* Added more tyeps of difficulty estimation in the NORABS score function.

1.1.0 (2025-07-15)  
------------------  
* Refactored `__init__` of predictors to include `alpha` and `device` parameters for greater flexibility.  
* Added EntmaxScore, SCPO and RC3P methods for classification tasks.  
* Added the normalized residual score in regression task.
* Added p-value computation and efficiency metrics from the paper "Criteria of Efficiency for Conformal Prediction".  



1.0.2 (2025-02-17)
------------------
* Refactored examples codebase for better organization and clarity
* Enhanced classification and Graph trainers with improved architecture
* Added new loss functions and trainer for Uncertainty-aware classifiers
* Changed default quantile value to infinity for better handling of edge cases
* Fixed handling of large calibration sets (>2^24 elements) in quantile computation (`#45 <https://github.com/ml-stat-Sustech/TorchCP/issues/45>`_)



1.0.1 (2024-12-16)
------------------

* Fixed the bugs of the RAPS score function and covgap in classification task
* Refactored the classification.loss, graph.score.snaps and regression.predictor.aci
* Fixed the bug where logo was not displayed in PyPi
* Updated the requirements.txt and examples for classification
* Added the trainer for Temperature Scaling and ConfTS in classification.trainer
* Added the Changelog page in the ReadtheDocs documentation

1.0.0 (2024-12-06)
------------------

* Added new score functions and training methods for classification, including KNN, TOPK, C-Adapter, and ConfTS.
* Introduced CP algorithms for graph node classification, such as DAPS, SNAPS, and NAPS.
* Added new conformal algorithms for regression, including CQRFM, CQRR, CQRM, and Ensemble CP.
* Introduced CP algorithms for LLMs.
* Added unit-test and examples.
* Optimized the form of prediction sets to improve the computational efficiency.
* Refactored the module design of Regression to improve the scalability.


0.1.3 (2024-02-22)
------------------
* Introduced R2CCP in regression task.

0.1.2 (2023-12-24)
------------------
* Introduced the ReadtheDocs documentation for TorchCP.

0.1.1 (2023-12-24)
------------------
* Introduced Margin score in classification task.


0.1.0 (2023-12-23)
------------------
* Introduced CP algorithms for classification, including ConfTr, LAC, APS, RAPS, SAPS, ClassConditional CP, Clustered CP and Weighted CP.
* Introduced CP algorithms for regression, including ACI, ABS and CQR.
