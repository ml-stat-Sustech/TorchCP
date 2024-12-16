
Changelog
=========

1.0.1 (2024-12-16)
------------------

* Fixed the bugs of the RAPS score function and covgap in classification task
* Refactored the classification.loss, graph.score.SNAPS and regression.predictor.aci
* Fixed the bug where logo was not displayed in PyPi
* Updated the requirements.txt and examples for classification
* Added a trainer for Temperature Scaling

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


0.1.0 (2023-12-23)
------------------
* Introduced CP algorithms for classification, including ConfTr, THR, APS, RAPS, SAPS, Classwise CP, Clustered CP and Weighted CP.
* Introduced CP algorithms for regression, including ACI, ABS and CQR.
