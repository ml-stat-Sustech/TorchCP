:mod:`torchcp.graph`
==========================

.. automodule:: torchcp.graph.score

score function
-----------------------------

..  autosummary::
    :nosignatures:

    DAPS
    SNAPS

.. autoclass:: DAPS
   :members:

.. autoclass:: SNAPS
   :members:

.. automodule:: torchcp.graph.predictor

predictor
-----------------------------

..  autosummary::
    :nosignatures:

    GraphSplitPredictor
    NAPSPredictor

.. autoclass:: GraphSplitPredictor
   :members:

.. autoclass:: NAPSPredictor
   :members:

.. automodule:: torchcp.graph.trainer

trainer
-----------------------------

..  autosummary::
    :nosignatures:

    CFGNNTrainer

.. autoclass:: CFGNNTrainer
   :members:

.. automodule:: torchcp.graph.utils.metrics

metrics
-----------------------------

..  autosummary::
    :nosignatures:
   
   coverage_rate
   average_size
   singleton_hit_ratio


.. autofunction:: coverage_rate
.. autofunction:: average_size
.. autofunction:: singleton_hit_ratio