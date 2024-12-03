import pytest
import torch


# Import all metrics functions
from torchcp.classification.utils.metrics import (
    coverage_rate, average_size, CovGap, VioClasses, 
    DiffViolation, SSCV, WSC, Metrics
)

class TestClassificationMetrics:
    @pytest.fixture
    def basic_data(self):
        """Basic test data setup"""
        prediction_sets = [
            [0, 1],    # Contains correct label 0
            [1],       # Contains correct label 1
            [0, 1, 2], # Contains correct label 1
            [2]        # Does not contain correct label 0
        ]
        labels = torch.tensor([0, 1, 1, 0])
        return prediction_sets, labels

    @pytest.fixture
    def multi_class_data(self):
        """Multi-class test data setup"""
        prediction_sets = [
            {0, 1},    # Class 0
            {1, 2},    # Class 1
            {0, 1, 2}, # Class 2
            {1, 2},    # Class 0
            {0, 2},    # Class 1
            {0, 1}     # Class 2
        ]
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        return prediction_sets, labels

    @pytest.fixture
    def logits_data(self):
        """Test logits data setup"""
        logits = torch.tensor([
            [0.8, 0.1, 0.1],  # High confidence for class 0
            [0.2, 0.7, 0.1],  # High confidence for class 1
            [0.1, 0.2, 0.7],  # High confidence for class 2
            [0.4, 0.3, 0.3]   # Low confidence prediction
        ])
        return logits

    def test_coverage_rate_default(self, basic_data):
        prediction_sets, labels = basic_data
        cvg = coverage_rate(prediction_sets, labels)
        # 3 out of 4 predictions contain correct label
        assert cvg == 0.75

    def test_coverage_rate_macro(self, multi_class_data):
        prediction_sets, labels = multi_class_data
        cvg = coverage_rate(prediction_sets, labels, coverage_type="macro", num_classes=3)
        # Should calculate per-class coverage and average
        assert isinstance(cvg, float)
        assert 0 <= cvg <= 1
        
    def test_coverage_rate_macro_error(self, multi_class_data):
        prediction_sets, labels = multi_class_data
        with pytest.raises(ValueError):
            cvg = coverage_rate(prediction_sets, labels, coverage_type="macro")

    def test_coverage_rate_invalid_input(self):
        with pytest.raises(ValueError):
            coverage_rate([{0}], torch.tensor([0, 1]))  # Length mismatch
        with pytest.raises(ValueError):
            coverage_rate([{0}], torch.tensor([0]), coverage_type="invalid")

    def test_average_size(self, basic_data):
        prediction_sets, labels = basic_data
        avg_sz = average_size(prediction_sets)
        # Set sizes are: 2, 1, 3, 1
        assert avg_sz == 1.75
        

    def test_covgap(self, multi_class_data):
        prediction_sets, labels = multi_class_data
        alpha = 0.1
        gap = CovGap(prediction_sets, labels, alpha, num_classes=3).item()
        assert isinstance(gap, float)
        assert gap >= 0

    def test_vioclasses(self, multi_class_data):
        prediction_sets, labels = multi_class_data
        alpha = 0.1
        violations = VioClasses(prediction_sets, labels, alpha, num_classes=3)
        assert isinstance(violations, int)
        assert violations >= 0
        assert violations <= 3  # Can't have more violations than classes

    def test_diffviolation(self, basic_data, logits_data):
        prediction_sets, labels = basic_data
        alpha = 0.1
        diff_viol, stats = DiffViolation(
            logits_data, prediction_sets, labels, alpha,
            strata_diff=[[1, 1], [2, 3]]
        )
        assert isinstance(diff_viol, float)
        assert diff_viol >= -1
        assert isinstance(stats, dict)

    def test_sscv(self, basic_data):
        prediction_sets, labels = basic_data
        alpha = 0.1
        sscv_value = SSCV(
            prediction_sets, labels, alpha,
            stratified_size=[[0, 1], [2, 3]]
        )
        assert isinstance(sscv_value, float)
        assert sscv_value >= -1

    def test_wsc(self):
        # Create small feature set for WSC testing
        features = torch.randn(10, 5)  # 10 samples, 5 features
        labels = torch.randint(0, 2, (10,))  # Binary labels
        prediction_sets = [{0, 1} for _ in range(10)]  # Simple prediction sets
        
        coverage = WSC(
            features, prediction_sets, labels,
            delta=0.1, M=10,  # Small M for testing
            test_size=0.5, random_state=42,
            verbose=False
        )
        assert isinstance(coverage, float)
        assert 0 <= coverage <= 1

    def test_metrics_class(self):
        metrics = Metrics()
        # Test valid metric name
        assert callable(metrics("coverage_rate"))
        # Test invalid metric name
        with pytest.raises(NameError):
            metrics("invalid_metric")