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
    EntmaxScore


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

.. autoclass:: EntmaxScore
   :members:


.. automodule:: torchcp.classification.predictor

predictor
--------------------------------

..  autosummary::
    :nosignatures:

    SplitPredictor
    ClassConditionalPredictor
    ClusteredPredictor
    RC3PPredictor
    WeightedPredictor

.. autoclass:: SplitPredictor
   :members:

.. autoclass:: ClassConditionalPredictor
   :members:

.. autoclass:: ClusteredPredictor
   :members:

.. autoclass:: RC3PPredictor
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
    SCPOTrainer


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

.. autoclass:: SCPOTrainer
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
.. autofunction:: compute_p_values
.. autofunction:: pvalue_criterion_S
.. autofunction:: pvalue_criterion_N
.. autofunction:: pvalue_criterion_U
.. autofunction:: pvalue_criterion_F
.. autofunction:: pvalue_criterion_M
.. autofunction:: pvalue_criterion_E
.. autofunction:: pvalue_criterion_OU
.. autofunction:: pvalue_criterion_OF
.. autofunction:: pvalue_criterion_OM
.. autofunction:: pvalue_criterion_OE




.. automodule:: torchcp.classification.utils

utils
--------------------------------

..  autosummary::
    :nosignatures:
   
   TS

.. autoclass:: TS
   :members:



