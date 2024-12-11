# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from torchcp.classification.trainer.ordinal_trainer import OrdinalClassifier, \
    OrdinalTrainer  # Replace with the actual import path


class TestOrdinalClassifier:
    @pytest.fixture
    def sample_classifier(self):
        """
        Create a simple linear classifier for testing
        
        Returns:
            nn.Linear: A linear classifier with 10 input features and 5 output classes
        """
        return nn.Linear(10, 5)

    def test_forward_pass_unimodal_distribution(self, sample_classifier):
        """
        Test the unimodal distribution property of the ordinal classifier
        - Verify that the output first increases and then decreases
        """
        # Create model
        model = OrdinalClassifier(sample_classifier)

        # Create random input
        x = torch.randn(32, 10)

        # Perform forward propagation
        output = model(x)

        # Check output shape
        assert output.shape == (32, 5), f"Incorrect output shape. Got {output.shape}"

        # Verify each batch's output
        for batch_idx in range(x.size(0)):
            # Find the index of the maximum value
            max_idx = torch.argmax(output[batch_idx])

            # Check increasing before peak
            if max_idx > 0:
                # Correct way to check increasing before peak
                increasing_segment = output[batch_idx, :max_idx + 1]
                assert torch.all(increasing_segment[:-1] <= increasing_segment[1:]), \
                    f"Batch {batch_idx}: Output is not increasing before peak"

            # Check decreasing after peak
            if max_idx < len(output[batch_idx]) - 1:
                # Correct way to check decreasing after peak
                decreasing_segment = output[batch_idx, max_idx:]
                assert torch.all(decreasing_segment[:-1] >= decreasing_segment[1:]), \
                    f"Batch {batch_idx}: Output is not decreasing after peak"

    @pytest.mark.parametrize("phi", ["abs", "square"])
    @pytest.mark.parametrize("varphi", ["abs", "square"])
    def test_transformation_configurations(self, sample_classifier, phi, varphi):
        """
        Test different transformation function configurations
        - Verify model works with various configurations
        
        Args:
            sample_classifier: Base classifier model
            phi: Transformation function for classifier output
            varphi: Transformation function for cumulative sum
        """
        # Create model with specific transformation functions
        model = OrdinalClassifier(sample_classifier, phi=phi, varphi=varphi)

        # Create random input
        x = torch.randn(16, 10)

        # Perform forward propagation
        output = model(x)

        # Basic property checks
        assert output.shape == (16, 5), f"Incorrect output shape. Got {output.shape}"

        # Check output validity
        assert not torch.isnan(output).any(), f"Output contains NaN values (phi={phi}, varphi={varphi})"
        assert not torch.isinf(output).any(), f"Output contains infinite values (phi={phi}, varphi={varphi})"

    def test_gradient_flow(self, sample_classifier):
        """
        Verify that gradients can flow through the model
        - Ensure backpropagation works correctly
        """
        model = OrdinalClassifier(sample_classifier)
        x = torch.randn(5, 10, requires_grad=True)

        output = model(x)
        loss = output.sum()

        # Attempt to compute gradients
        loss.backward()

        # Check that input gradient exists and is not zero
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_input_dimension_validation(self, sample_classifier):
        """
        Test input dimension validation
        - Ensure model raises an error for inputs with fewer than 3 features
        """
        model = OrdinalClassifier(sample_classifier)

        # Test input with less than 3 features
        with pytest.raises(ValueError, match="The input dimension must be greater than 2."):
            x = torch.randn(3, 2)
            model(x)

    def test_reproducibility(self, sample_classifier):
        """
        Test model reproducibility
        - Verify consistent output for the same input and seed
        """
        # Set a fixed seed
        torch.manual_seed(42)

        # Create model
        model = OrdinalClassifier(sample_classifier)

        # Create input
        x = torch.randn(10, 10)

        # First run
        output1 = model(x)

        # Reset seed and rerun
        torch.manual_seed(42)
        model = OrdinalClassifier(sample_classifier)
        output2 = model(x)

        # Check if outputs are the same
        assert torch.allclose(output1, output2), "Outputs are not reproducible"

    def test_invalid_phi_function(self, sample_classifier):
        """
        Test that an error is raised when an invalid phi function is provided
        """
        optimizer = optim.Adam(sample_classifier.parameters())
        loss_fn = nn.CrossEntropyLoss()

        with pytest.raises(NotImplementedError,
                           match="phi function 'invalid' is not implemented. Options are 'abs' and 'square'."):
            OrdinalTrainer(
                model=sample_classifier,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=torch.device('cpu'),
                ordinal_config={"phi": "invalid"}
            )

    def test_invalid_varphi_function(self, sample_classifier):
        """
        Test that an error is raised when an invalid varphi function is provided
        """
        optimizer = optim.Adam(sample_classifier.parameters())
        loss_fn = nn.CrossEntropyLoss()

        with pytest.raises(NotImplementedError,
                           match="varphi function 'invalid' is not implemented. Options are 'abs' and 'square'."):
            OrdinalTrainer(
                model=sample_classifier,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=torch.device('cpu'),
                ordinal_config={"varphi": "invalid"}
            )

    def test_valid_configuration_combinations(self, sample_classifier):
        """
        Test various valid combinations of phi and varphi functions
        """
        optimizer = optim.Adam(sample_classifier.parameters())
        loss_fn = nn.CrossEntropyLoss()

        valid_configs = [
            {"phi": "abs", "varphi": "abs"},
            {"phi": "abs", "varphi": "square"},
            {"phi": "square", "varphi": "abs"},
            {"phi": "square", "varphi": "square"}
        ]

        for config in valid_configs:
            try:
                trainer = OrdinalTrainer(
                    model=sample_classifier,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=torch.device('cpu'),
                    ordinal_config=config
                )
                assert isinstance(trainer.model, OrdinalClassifier), \
                    f"Model not wrapped correctly for config {config}"
            except Exception as e:
                pytest.fail(f"Unexpected error with config {config}: {str(e)}")

    @pytest.fixture
    def sample_classifier(self):
        """
        Create a simple linear classifier for testing
        
        Returns:
            nn.Linear: A linear classifier with 10 input features and 5 output classes
        """
        return nn.Linear(10, 5)

    def test_ordinal_classifier_initialization(self, sample_classifier):
        """
        Test the OrdinalClassifier initialization with different configurations
        """
        configurations = [
            {"phi": "abs", "varphi": "abs"},
            {"phi": "square", "varphi": "square"},
            {}
        ]

        for config in configurations:
            try:
                ordinal_model = OrdinalClassifier(sample_classifier, **config)

                assert ordinal_model.classifier == sample_classifier

                assert hasattr(ordinal_model, 'phi_function')
                assert hasattr(ordinal_model, 'varphi_function')
            except Exception as e:
                pytest.fail(f"Unexpected error with config {config}: {str(e)}")


if __name__ == '__main__':
    pytest.main([__file__])
