# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from torchcp.llm.predictor.conformal_llm import (
    StoppingCriteriaSub,
    NAME_TO_SCORE,
    NAME_TO_SCALER,
    ConformalLM,
    DEFAULT_EPSILONS
)


class TestStoppingCriteriaSub:
    @pytest.fixture
    def setup_stopping_criteria(self):
        return StoppingCriteriaSub(input_length=2, stop_ids=[50256])

    def test_stopping_criteria_no_stop_ids(self):
        criteria = StoppingCriteriaSub(input_length=0)
        input_ids = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([[0.1, 0.2, 0.3]])
        assert not criteria(input_ids, scores)

    def test_stopping_criteria_with_stop_ids(self, setup_stopping_criteria):
        input_ids = torch.tensor([[1, 2, 50256, 4]])
        scores = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        assert setup_stopping_criteria(input_ids, scores)

    def test_stopping_criteria_no_stop_found(self, setup_stopping_criteria):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        assert not setup_stopping_criteria(input_ids, scores)


class TestConformalLM:

    @pytest.fixture
    def sample_data(self):
        return {
            'training_scores': torch.randn(10, 5),
            'training_labels': torch.randint(0, 2, (10, 5)),
            'tuning_scores': torch.randn(10, 5),
            'tuning_similarities': torch.rand(10, 5, 5),
            'tuning_labels': torch.randint(0, 2, (10, 5)),
            'cal_scores': torch.randn(10, 5),
            'cal_similarities': torch.rand(10, 5, 5),
            'cal_labels': torch.randint(0, 2, (10, 5)),
            'test_scores': torch.randn(10, 5),
            'test_similarities': torch.rand(10, 5, 5),
            'test_labels': torch.randint(0, 2, (10, 5))
        }

    @pytest.fixture
    def setup_basic_model(self, sample_data):
        conformal_llm = ConformalLM(
            epsilons=torch.linspace(0, 1, 5),
            scaling_type="none",
            set_score_function_name="none",
            rejection=False
        )
        return conformal_llm

    def test_initialization_invalid_params(self):
        # Test invalid scaling_type
        with pytest.raises(ValueError):
            ConformalLM(scaling_type="invalid")

        # Test invalid set_score_function_name
        with pytest.raises(ValueError):
            ConformalLM(set_score_function_name="invalid")

    def test_scaling(self, setup_basic_model, sample_data):
        setup_basic_model.scaling(
            sample_data['training_scores'],
            sample_data['training_labels']
        )
        assert hasattr(setup_basic_model, 'scaler')

    def test_tuning(self, setup_basic_model, sample_data):
        # First do scaling
        setup_basic_model.scaling(
            sample_data['training_scores'],
            sample_data['training_labels']
        )

        # Then do tuning
        setup_basic_model.tuning(
            sample_data['tuning_scores'],
            sample_data['tuning_similarities'],
            sample_data['tuning_labels']
        )
        assert hasattr(setup_basic_model, 'candidate_configs')

    def test_calibrate_configs(self, setup_basic_model, sample_data):
        # Setup required prerequisites
        setup_basic_model.scaling(
            sample_data['training_scores'],
            sample_data['training_labels']
        )
        setup_basic_model.tuning(
            sample_data['tuning_scores'],
            sample_data['tuning_similarities'],
            sample_data['tuning_labels']
        )

        # Test calibration
        setup_basic_model.calibrate_configs(
            sample_data['cal_scores'],
            sample_data['cal_similarities'],
            sample_data['cal_labels'],
            alpha=0.1
        )
        assert hasattr(setup_basic_model, 'best_valid_configs')

    def test_get_pareto_frontier(self, setup_basic_model, sample_data):
        setup_basic_model.scaling(
            sample_data['training_scores'],
            sample_data['training_labels']
        )
        scores = sample_data['training_scores']
        similarities = sample_data['tuning_similarities'][:, 0, :]
        labels = sample_data['training_labels']

        frontier = setup_basic_model.get_pareto_frontier(scores, similarities, labels)
        assert isinstance(frontier, torch.Tensor)
        assert frontier.dim() == 2
        assert frontier.shape[1] == 3  # 3 parameters in config

    def test_predict_with_config(self, setup_basic_model, sample_data):
        config = torch.tensor([-float('inf'), 0.0, float('inf')])
        setup_basic_model.scaling(
            sample_data['training_scores'],
            sample_data['training_labels']
        )

        predictions = setup_basic_model.predict_with_config(
            config,
            sample_data['test_scores'],
            sample_data['test_similarities']
        )
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape == sample_data['test_scores'].shape

    def test_evaluate(self, setup_basic_model, sample_data):
        # Setup required prerequisites
        setup_basic_model.scaling(
            sample_data['training_scores'],
            sample_data['training_labels']
        )
        setup_basic_model.tuning(
            sample_data['tuning_scores'],
            sample_data['tuning_similarities'],
            sample_data['tuning_labels']
        )
        setup_basic_model.calibrate_configs(
            sample_data['cal_scores'],
            sample_data['cal_similarities'],
            sample_data['cal_labels'],
            alpha=0.1
        )

        # Test evaluation
        results = setup_basic_model.evaluate(
            sample_data['test_scores'],
            sample_data['test_similarities'],
            sample_data['test_labels']
        )
        assert isinstance(results, dict)
        assert 'configs' in results
        assert 'avg_losses' in results
        assert 'avg_size' in results
        assert 'avg_SSCL' in results

    def test_rejection_mode(self, sample_data):
        model = ConformalLM(rejection=True)
        model.scaling(
            sample_data['training_scores'],
            sample_data['training_labels']
        )

        config = torch.tensor([0.5, 0.5, 0.5])
        predictions = model.predict_with_config(
            config,
            sample_data['test_scores'],
            sample_data['test_similarities']
        )
        assert isinstance(predictions, torch.Tensor)

    def test_select_lambdas(self, setup_basic_model):
        values = torch.randn(100)
        lambdas = setup_basic_model._ConformalLM__select_lambdas(values, max_lambdas=10)
        assert isinstance(lambdas, torch.Tensor)
        assert lambdas[0] == -float('inf')
        assert lambdas[-1] == float('inf')

    def test_get_C_cutoff(self, setup_basic_model):
        set_scores = torch.randn(5, 10)
        set_lambda = torch.tensor(0.5)
        cutoff = setup_basic_model._ConformalLM__get_C_cutoff(set_scores, set_lambda)
        assert isinstance(cutoff, torch.Tensor)
        assert cutoff.shape == torch.Size([5])
        assert (cutoff >= 0).all()
        assert (cutoff < set_scores.shape[1]).all()

    # @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_calibrate_with_model(self, setup_basic_model):
        # This test only runs if CUDA is available
        dataset = [
            {'question': 'What is 2+2?', 'answer': '4'},
            {'question': 'What is the capital of France?', 'answer': 'Paris'}
        ]
        prompt_template = "Q: {}\nA:"

        # Mock tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")

        setup_basic_model.tokenizer = tokenizer
        setup_basic_model.model = model

        setup_basic_model.calibrate(dataset, prompt_template)
