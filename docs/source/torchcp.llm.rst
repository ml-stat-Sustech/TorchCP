:mod:`torchcp.llm`
==========================

.. automodule:: torchcp.llm.predictor

predictor
-----------------------------------

..  autosummary::
    :nosignatures:

    ConformalLM

.. autoclass:: ConformalLM
   :members:


.. automodule:: torchcp.llm.utils.scoring

scoring
-----------------------------------

..  autosummary::
    :nosignatures:

    geometric
    marginal
    first_k
    first_k_no_mask
    max
    sum

.. autofunction:: geometric
.. autofunction:: marginal
.. autofunction:: first_k
.. autofunction:: first_k_no_mask
.. autofunction:: max
.. autofunction:: sum

.. automodule:: torchcp.llm.utils.scaling

scaling
-----------------------------------

..  autosummary::
    :nosignatures:

    LogisticRegression
    PlattScaler
    BinningScaler
    PlattBinningScaler
    RecurrentScaler

.. autofunction:: LogisticRegression
.. autofunction:: PlattScaler
.. autofunction:: BinningScaler
.. autofunction:: PlattBinningScaler
.. autofunction:: RecurrentScaler

.. automodule:: torchcp.llm.utils.metrics

metrics
-----------------------------------

..  autosummary::
    :nosignatures:

    average_size
    average_sample_size
    average_set_loss
    SSCL

.. autofunction:: average_size
.. autofunction:: average_sample_size
.. autofunction:: average_set_loss
.. autofunction:: SSCL
