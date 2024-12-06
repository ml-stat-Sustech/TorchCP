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

    THR
    APS
    RAPS
    SAPS
    Margin
    TOPK
    KNN


.. autoclass:: THR
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

    ConfTr
    ConfTS 
    CDLoss

.. autoclass:: ConfTr
   :members:

.. autoclass:: ConfTS
   :members:

.. autoclass:: CDLoss
   :members:

.. automodule:: torchcp.classification.trainer

trainer
--------------------------------

..  autosummary::
    :nosignatures:

    Trainer
    OrdinalTrainer 

.. autoclass:: Trainer
   :members:

.. autoclass:: OrdinalTrainer
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

.. automodule:: torchcp.classification.utils

utils
--------------------------------

..  autosummary::
    :nosignatures:
   
   TS

.. autoclass:: TS
   :members:



