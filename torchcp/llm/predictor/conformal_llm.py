# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Evaluate set uncertainty metrics."""
import math
import collections
import itertools

import torch
from scipy.stats import binom
from transformers import set_seed, StoppingCriteria, StoppingCriteriaList

from torchcp.utils.common import get_device
from torchcp.llm.utils import Metrics, scoring, scaling, loss


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, input_length=0, stop_ids=None):
        super().__init__()
        self.stop_ids = stop_ids
        self.input_length = input_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> bool:
        if self.stop_ids is None:
            return False
        output = input_ids[:, self.input_length:]
        has_stop_ids = []
        for stop_id in self.stop_ids:
            has_stop_id = torch.any(output == stop_id, dim=1)
            has_stop_ids.append(has_stop_id)
        has_stop_ids = torch.stack(has_stop_ids, dim=1)
        return (has_stop_ids.any(dim=1).all())


NAME_TO_SCORE = {
    'geo': scoring.geometric,
    'marginal': scoring.marginal,
    'first_k': scoring.first_k,
    'first_k_no_mask': scoring.first_k_no_mask,
    'max': scoring.max,
    'sum': scoring.sum,
    'none': lambda x, m=None: x,
}

NAME_TO_SCALER = {
    'platt': scaling.PlattScaler,
    'bin': scaling.BinningScaler,
    'platt_bin': scaling.PlattBinningScaler,
    'rnn': scaling.RecurrentScaler,
    'none': scaling.BaseScaler,
}

DEFAULT_EPSILONS = torch.linspace(0, 1, 101)


class ConformalLM:
    """
    Method: Conformal Language Modeling
    Paper:  Conformal Language Modeling (Victor Quach et al., ICLR'24)
    Link: https://openreview.net/forum?id=pzUhfQ74c5
    Github: https://github.com/Varal7/conformal-language-modeling
    
    
    Args:
        tokenizer (Any, optional): A tokenizer for the language model. Default is None.
        model (Any, optional): A language model based on PyTorch. Default is None.
        epsilons (torch.Tensor, optional): The risk levels that need to be controlled. Default is None.
        scaling_type (str, optional): The scaling type for scores. Default is "none". Score scaling method, one of {platt, bin, platt_bin, rnn, none}.
        scale_kwargs (dict, optional): The parameters for the scaling function. Default is None.
        set_score_function_name (str, optional): The name of the score function to use. Default is "none". Score function name, one of {geo, marginal, first_k, first_k_no_mask, max, sum, none}.
        rejection (bool, optional): Indicates whether to use rejection sampling. Default is False.
        seed (int, optional): The random seed. Default is 2024.
        alpha (float, optional): The significance level. Default is 0.1.
        device (torch.device, optional): The device on which the model is located. Default is None.
    """

    def __init__(self, tokenizer=None, model=None, epsilons=None, scaling_type="none", scale_kwargs=None,
                 set_score_function_name="none", rejection=False, seed=2024, alpha=0.1, device=None) -> None:
        if scaling_type not in NAME_TO_SCALER:
            raise ValueError(f"Invalid scaling_type: {scaling_type}. Must be one of: {list(NAME_TO_SCALER.keys())}")
        if set_score_function_name not in NAME_TO_SCORE:
            raise ValueError(
                f"Invalid set_score_function_name: {set_score_function_name}. Must be one of: {list(NAME_TO_SCORE.keys())}")

        self.tokenizer = tokenizer
        self.model = model

        if not (0 < alpha < 1):
            raise ValueError("alpha should be a value in (0, 1).")
        self.alpha = alpha

        if device is not None:
            self._device = torch.device(device)
        elif model is not None:
            self._device = get_device(model)
        else:
            self._device = torch.device("cpu")

        if epsilons is None:
            epsilons = DEFAULT_EPSILONS
        self.epsilons = epsilons
        self.scaling_type = scaling_type
        self.scale_kwargs = scale_kwargs
        self.rejection = rejection
        self.set_score_function = NAME_TO_SCORE[set_score_function_name]
        self.seed = seed
        self.metrics = Metrics()

    def scaling(self, training_scores, training_labels):
        if self.scale_kwargs is None:
            self.scale_kwargs = {}
        self.scaler = NAME_TO_SCALER[self.scaling_type](**self.scale_kwargs)
        self.scaler.fit(training_scores, training_labels)

    def tuning(self, tuning_scores, tuning_similarities, tuning_labels):
        tuning_scores = self.scaler.predict(tuning_scores)

        self.candidate_configs = self.get_pareto_frontier(
            item_scores=tuning_scores,
            similarity_scores=tuning_similarities,
            item_labels=tuning_labels)

    def calibrate(self, dataset, prompt_template):

        repeat_per_prompt = 4
        num_return_sequences = 5
        stop_word_ids = [13, 1919, 2982, 869, 29889]

        for sample in dataset:
            generations = []
            question = sample['question']
            answer = sample['answer']
            input_text = prompt_template.format(question)
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self._device)
            stopping_criteria = StoppingCriteriaList(
                [StoppingCriteriaSub(stop_ids=stop_word_ids, input_length=input_ids.shape[1])])
            kwargs = {
                "max_new_tokens": 100,
                "return_dict_in_generate": True,
                "output_scores": True,
                "stopping_criteria": stopping_criteria,
                "num_return_sequences": num_return_sequences,
                "do_sample": True,
            }
            for i in range(repeat_per_prompt):
                set_seed(self.seed + i)
                with torch.no_grad():
                    outputs = self.model.generate(input_ids, **kwargs)

                transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores,
                                                                         normalize_logits=True)
                input_length = 1 if self.model.config.is_encoder_decoder else input_ids.shape[1]
                generated_tokens = outputs.sequences[:, input_length:]

                for i in range(len(generated_tokens)):
                    tokens = []
                    scores = []
                    for tok, score in zip(generated_tokens[i], transition_scores[i]):
                        if tok in stop_word_ids and len(tokens) > 0:
                            break
                        tokens.append(tok)
                        scores.append(score)

                    tokens = torch.stack(tokens, dim=0)
                    scores = torch.stack(scores, dim=0)

                    generations.append({
                        'tokens': tokens.cpu().tolist(),
                        'scores': scores.cpu().tolist(),
                        'decoded': self.tokenizer.decode(tokens)
                    })

    def calibrate_configs(self, cal_scores, cal_similarities, cal_labels, alpha=None):
        if alpha is None:
            alpha = self.alpha

        cal_scores = self.scaler.predict(cal_scores)

        self.best_valid_configs = torch.full((len(self.epsilons), 3), float('nan'))
        is_stopped = [False] * len(self.epsilons)
        for i in range(self.candidate_configs.shape[0]):
            config = self.candidate_configs[i]
            prediction_sets = self.predict_with_config(config=config, item_scores=cal_scores,
                                                       similarity_scores=cal_similarities)

            avg_losses = self.metrics("average_set_loss")(prediction_sets, loss.set_losses_from_labels(cal_labels))
            for j, epsilon in enumerate(self.epsilons):
                n = len(cal_labels)
                p_value = binom.cdf(n * avg_losses, n, epsilon)
                if p_value <= alpha and not is_stopped[j]:
                    self.best_valid_configs[j] = config
                else:
                    is_stopped[j] = True

    def get_pareto_frontier(self, item_scores, similarity_scores, item_labels):
        """Compute a pareto frontier."""
        lambda_1 = self.__select_lambdas(similarity_scores, max_lambdas=25)
        lambda_2 = self.__select_lambdas(item_scores, max_lambdas=25)
        lambda_3 = self.__select_lambdas(self.set_score_function(item_scores), max_lambdas=25)

        configs = []
        costs = []
        for config in itertools.product(lambda_1, lambda_2, lambda_3):
            config = torch.tensor(list(config), dtype=torch.float32, device=item_scores.device)
            configs.append(config)
            prediction_sets = self.predict_with_config(config=config,
                                                       item_scores=item_scores,
                                                       similarity_scores=similarity_scores)

            losses = self.metrics("average_set_loss")(prediction_sets, loss.set_losses_from_labels(item_labels))

            avg_preidction_size = self.metrics("average_size")(prediction_sets)
            avg_sample_size = self.metrics("average_sample_size")(prediction_sets)
            costs.append((losses, avg_preidction_size + avg_sample_size))

        configs = torch.stack(configs)
        costs = torch.tensor(costs)

        is_efficient = torch.ones(costs.shape[0], dtype=torch.bool)

        for i, c in enumerate(costs):
            if is_efficient[i]:
                this_is_efficient = is_efficient.clone()
                is_efficient[this_is_efficient] = torch.any(costs[this_is_efficient] < c, dim=1)
                is_efficient[i] = True

        pareto_configs = configs[is_efficient]
        pareto_costs = costs[is_efficient]

        sort_idx = torch.argsort(pareto_costs[:, 0])
        pareto_costs = pareto_costs[sort_idx]
        ordered_configs = pareto_configs[sort_idx]

        return ordered_configs

    def __select_lambdas(self, values, max_lambdas=1000):
        """Select unique quantiles of the empirical distribution."""

        quantiles = torch.linspace(0, 1, max_lambdas, dtype=values.dtype, device=values.device)
        lambdas = torch.quantile(values, quantiles)
        lambdas = torch.unique(lambdas, sorted=True)
        lambdas = torch.cat([torch.tensor([-float('inf')], dtype=values.dtype),
                             lambdas,
                             torch.tensor([float('inf')], dtype=values.dtype)])
        return lambdas

    def evaluate(self, test_scores, test_similarities, test_labels):

        trial_results = collections.defaultdict(list)
        trial_results['configs'] = self.best_valid_configs

        for j, config in enumerate(self.best_valid_configs):
            prediction_sets = self.predict_with_config(config=config,
                                                       item_scores=test_scores,
                                                       similarity_scores=test_similarities)

            prediction_set_losses = loss.set_losses_from_labels(test_labels)
            avg_losses = self.metrics("average_set_loss")(prediction_sets, prediction_set_losses)
            avg_size = self.metrics("average_size")(prediction_sets)

            output = dict(
                # Average loss.
                avg_losses=avg_losses,
                # Average set size.
                avg_size=avg_size,
                # Size-stratified conditional loss
                avg_SSCL=self.metrics("SSCL")(prediction_sets, prediction_set_losses)
            )
            for k, v in output.items():
                trial_results[k].append(v)
        return trial_results

    def predict_with_config(self, config, item_scores, similarity_scores):
        """
        Construct the prediction set for a given config.

        Args:
            set_scores: [num_examples, max_generations]
                set_scores[i, j] = score of set after sample j for example i.
            set_sizes: [num_examples, max_generations]
                set_sizes[i, j] = effective set size after sample j for example i.
            set_losses: [num_examples, max_generations]
                set_loss[i, j] = loss of set after sample j for example i.
            lambdas: [num_thresholds]
                Array of thresholds to test.

        Returns:
            Dictionary of metrics (per lambda).
        """

        item_scores = self.scaler.predict(item_scores)

        if torch.isnan(config).any():
            return torch.ones_like(item_scores)

        lambda_1, lambda_2, lambda_3 = config[0], config[1], config[2]

        # If doing rejections, remove low quality and low diversity items.
        if self.rejection:
            # Reject low-quality examples.
            quality_mask = (item_scores >= lambda_2)

            # Reject examples that are too similar to previous items.
            similarity_mask = torch.ones_like(quality_mask)

            # Make similarity score symmetric.
            similarity_scores = torch.maximum(similarity_scores, similarity_scores.transpose(1, 2))

            # Low quality scores are rejected, so they don't count --- set similarity to 0.
            similarity_scores = similarity_scores * quality_mask.unsqueeze(1)

            for k in range(1, similarity_scores.shape[1]):
                # Anything that has been rejected up until this point also doesn't count.
                # Set those scores to 0 so we don't pick them up as well.
                similarity_scores = similarity_scores * similarity_mask.unsqueeze(1)

                # We only look at scores up until this point.
                max_similarity, _ = torch.max(similarity_scores[:, k, :k], dim=-1)
                similarity_mask[:, k] = (max_similarity <= lambda_1)

            # Combine rejection rules.
            kept_mask = quality_mask * similarity_mask
        else:
            kept_mask = torch.ones_like(item_scores)

        # Compute set selections for all values of lambda.
        set_scores = self.set_score_function(item_scores, kept_mask)
        Set_indices = self.__get_C_cutoff(set_scores, lambda_3)

        temp_matrix = torch.ones_like(kept_mask) * torch.arange(kept_mask.shape[1])

        temp_mask = temp_matrix <= Set_indices.unsqueeze(-1)
        prediction_mask = temp_mask * kept_mask

        return prediction_mask.to(torch.float)

    def __get_C_cutoff(self, set_scores, set_lambda):
        """Compute prediction sets C for given thresholds tau.

        Args:
            set_scores: [num_examples, max_generations]
                set_scores[i, j] = score of set j for example i.
            set_lambda: Threshold to use.

        Returns:
            C_indices: [num_examples]
                Indices of the selected sets C for each example.
        """
        cummax_scores, _ = torch.cummax(set_scores, dim=-1)

        # Create mask where cummax_scores < set_lambda
        mask = cummax_scores < set_lambda.unsqueeze(-1)

        # Sum along the last dimension to get C_indices
        C_indices = torch.sum(mask, dim=-1)

        # Clip C_indices to be within valid range
        C_indices = torch.clamp(C_indices, min=0, max=set_scores.shape[1] - 1)
        return C_indices
