:mod:`torchcp.classification`
================================

.. automodule:: torchcp.classification.score
   :members:
   :undoc-members:
   :show-inheritance:

score function
--------------------------------

..  autosummary::
    :nosignatures:

    LAC
    APS
    RAPS
    SAPS
    Margin
    TOPK
    KNN


.. autoclass:: LAC
   :members:

.. autoclass:: APS
   :members:

.. autoclass:: RAPS
   :members:

.. autoclass:: SAPS
   :members:

.. autoclass:: Margin
   :members:

.. autoclass:: TOPK
   :members:

.. autoclass:: KNN
   :members:

.. automodule:: torchcp.classification.predictor

predictor
--------------------------------

..  autosummary::
    :nosignatures:

    SplitPredictor
    ClassWisePredictor
    ClusteredPredictor
    WeightedPredictor

.. autoclass:: SplitPredictor
   :members:

.. autoclass:: ClassWisePredictor
   :members:

.. autoclass:: ClusteredPredictor
   :members:

.. autoclass:: WeightedPredictor
   :members:


.. automodule:: torchcp.classification.loss

loss function
--------------------------------

..  autosummary::
    :nosignatures:

    ConfTrLoss
    ConfTSLoss 
    CDLoss
    UncertaintyAwareLoss

.. autoclass:: ConfTrLoss
   :members:

.. autoclass:: ConfTSLoss
   :members:

.. autoclass:: CDLoss
   :members:

.. autoclass:: UncertaintyAwareLoss
   :members:

.. automodule:: torchcp.classification.trainer

trainer
--------------------------------

..  autosummary::
    :nosignatures:

    BaseTrainer
    ConfTSTrainer 
    TSTrainer
    OrdinalTrainer
    UncertaintyAwareTrainer

.. autoclass:: BaseTrainer
   :members:

.. autoclass:: ConfTSTrainer
   :members:

.. autoclass:: TSTrainer
   :members:

.. autoclass:: OrdinalTrainer
   :members:

.. autoclass:: UncertaintyAwareTrainer
   :members:

.. automodule:: torchcp.classification.utils.metrics
   
metrics
--------------------------------

..  autosummary::
    :nosignatures:
   
   coverage_rate
   average_size
   CovGap
   VioClasses
   DiffViolation
   SSCV
   WSC

.. autofunction:: coverage_rate
.. autofunction:: average_size
.. autofunction:: CovGap
.. autofunction:: VioClasses
.. autofunction:: DiffViolation
.. autofunction:: SSCV
.. autofunction:: WSC
.. autofunction:: singleton_hit_ratio

.. automodule:: torchcp.classification.utils

utils
--------------------------------

..  autosummary::
    :nosignatures:
   
   TS

.. autoclass:: TS
   :members:



