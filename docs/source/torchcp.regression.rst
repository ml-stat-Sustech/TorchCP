:mod:`torchcp.regression`
==========================

.. automodule:: torchcp.regression.score

score function
----------------------------------------

..  autosummary::
    :nosignatures:

    ABS
    CQR
    CQRFM
    CQRM
    CQRR
    R2CCP

.. autoclass:: ABS
   :members:

.. autoclass:: CQR
   :members:

.. autoclass:: CQRFM
   :members:

.. autoclass:: CQRM
   :members:

.. autoclass:: CQRR
   :members:

.. autoclass:: R2CCP
   :members:


.. automodule:: torchcp.regression.predictor

predictor
----------------------------------------

..  autosummary::
    :nosignatures:

    SplitPredictor
    EnsemblePredictor
    ACIPredictor

.. autoclass:: SplitPredictor
   :members:

.. autoclass:: EnsemblePredictor
   :members:

.. autoclass:: ACIPredictor
   :members:

.. automodule:: torchcp.regression.loss

loss
----------------------------------------

..  autosummary::
    :nosignatures:

    QuantileLoss
    R2ccpLoss

.. autoclass:: QuantileLoss
   :members:

.. autoclass:: R2ccpLoss
   :members:


.. automodule:: torchcp.regression.utils.metrics
   
metrics
----------------------------------------

..  autosummary::
    :nosignatures:

    coverage_rate
    average_size

..  autofunction:: coverage_rate
..  autofunction:: average_size


.. automodule:: torchcp.regression.utils.utils
   
utils
----------------------------------------


..  autosummary::
    :nosignatures:

    calculate_midpoints

..  autofunction:: calculate_midpoints
   