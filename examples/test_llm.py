import argparse
import os
import sys
import json
import torch
import math
import re
import string
import numpy as np
import numpy as np
import collections
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, set_seed, StoppingCriteriaList
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


from examples.llm.prompts import  few_shot_qa, StoppingCriteriaSub
from examples.utils import get_dataset_dir
from examples.llm.llm_utils import get_dataset
from torchcp.llm.predictor import ConformalLM



def preprocess_data(dataset_name, model_path, output_path):
    dataset = get_dataset(dataset_name)

    few_shot_size = 32
    score_type = "transition" 
    strategy = "sample"  
    seed = 2024
    repeat_per_prompt = 4
    num_return_sequences = 5


    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    pre_prompt = "Answer these questions\n\n"
    prompt_template = pre_prompt + "\n".join(few_shot_qa[:few_shot_size * 2]) + "\nQ: {}\nA:"

    # Define stopping criteria
    stop_word_ids = [13, 1919, 2982, 869, 29889]

    def run_triviaqa(sample):
        question = sample['question']
        answer = sample['answer']
        input_text = prompt_template.format(question)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_ids=stop_word_ids, input_length=input_ids.shape[1])])
        set_seed(seed)


        kwargs = {
            "max_new_tokens": 100,
            "return_dict_in_generate": True,
            "output_scores": True,
            "stopping_criteria": stopping_criteria,
            "num_return_sequences": num_return_sequences,
        }

        if strategy == "greedy":
            kwargs["do_sample"] = False
            kwargs["num_beams"] = 1
        elif strategy == "sample":
            kwargs["do_sample"] = True

        generations = []

        for batch in range(repeat_per_prompt):
            set_seed(seed + batch)
            with torch.no_grad():
                outputs = model.generate(input_ids, **kwargs)

            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
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
                    'decoded': tokenizer.decode(tokens)
                })

        return {
            'question': question,
            'answer': answer,
            'generations': generations
        }

    # Fix EOS
    def fix_eos(example):
        for p in example['generations']:
            if tokenizer.eos_token_id in p['tokens']:
                truncate = p['tokens'].index(tokenizer.eos_token_id)
                p['tokens'] = p['tokens'][:truncate]
                p['scores'] = p['scores'][:truncate]
                p['decoded'] = tokenizer.decode(p['tokens'])
        return example

    # Self-evaluation
    CORRECT = 12521
    INCORRECT = 797


    def exact_match(prediction, answers):
        prediction = re.split(r'[.,\n]', prediction, maxsplit=1)[0]
        prediction = normalize_text(prediction)
        answers = [normalize_text(a) for a in answers]
        return float(any([prediction == a for a in answers]))

    def normalized_likelihood(log_probs, alpha=0.6):
        total_log_probs = np.sum(np.clip(log_probs, -1e5, 0))
        penalty = (5 + len(log_probs)) ** alpha / (5 + 1) ** alpha
        return np.exp(total_log_probs / penalty)

    def compute_qa_scores(examples):
        all_labels = []
        all_scores = []
        for example in examples:
            answers = example['answer']['normalized_aliases']
            predictions = [p['decoded'] for p in example['generations']]
            # self_eval_scores = [p['self_eval'] for p in example['generations']]
            scores = [normalized_likelihood(p['scores']) for p in example['generations']]
            labels = [exact_match(p, answers) for p in predictions]
            all_scores.append(scores)
            all_labels.append(labels)
        return np.array(all_labels), np.array(all_scores)


    def normalize_text(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        
        def white_space_fix(text):
            return " ".join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_diversity(results):
        all_predictions = []
        for example in results:
            predictions = [normalize_text(p['decoded']) for p in example['generations']]
            all_predictions.append(predictions)
        
        N = len(all_predictions)
        g = len(all_predictions[0])
        
        def compute_match(i):
            arr = np.zeros((g, g))
            for j in range(g):
                for k in range(j, g):
                    arr[j, k] = (all_predictions[i][j]) == (all_predictions[i][k])
            return arr
        
        diversity = np.array([compute_match(i) for i in tqdm(range(N), desc="Computing diversity")])
        
        return diversity

    results = []
    for sample in tqdm(dataset):
        result = run_triviaqa(sample)
        result = fix_eos(result)
        results.append(result)

    all_labels, all_scores = compute_qa_scores(results)
    diversity = compute_diversity(results)

    np.savez(output_path, labels=all_labels, scores=all_scores, diversity = diversity)




def test_conformal_llm():
    dataset_name = "trivia_qa"
    model_path = "meta-llama/Llama-2-7b-hf"
    
    output_path = os.path.join(get_dataset_dir(), f"{dataset_name}_results.npz")
    if not os.path.exists(output_path):
        preprocess_data(dataset_name, model_path, output_path)


    methods = [
        dict(scaling=('none', {}), scoring='first_k', rejection=False),
        dict(scaling=('none', {}), scoring='first_k', rejection=True),
        dict(scaling=('none', {}), scoring='max', rejection=False),
        dict(scaling=('none', {}), scoring='max', rejection=True),
        dict(scaling=('none', {}), scoring='sum', rejection=False),
        dict(scaling=('none', {}), scoring='sum', rejection=True),
        dict(scaling=('none', {}), scoring='first_k_no_mask', rejection=False),
        dict(scaling=('none', {}), scoring='first_k_no_mask', rejection=True),
        dict(scaling=('platt', {}), scoring='geo', rejection=False),
        dict(scaling=('bin', {}), scoring='geo', rejection=False),
        dict(scaling=('platt_bin', {}), scoring='geo', rejection=False),
        dict(scaling=('rnn', {}), scoring='geo', rejection=False),
        dict(scaling=('none', {}), scoring='geo', rejection=False),
        dict(scaling=('platt', {}), scoring='geo', rejection=True),
    ]

    p_cal = 0.3 
    p_tuning= 0.3

    data = np.load(output_path)
    all_labels = torch.from_numpy(data['labels'])
    all_scores = torch.from_numpy(data['scores'])
    diversity = torch.from_numpy(data['diversity'])
    
    breakpoint()
    

    epsilons = np.linspace(0, 1, 101)
    num_trials = 1
    alpha = 0.05

    all_results = [collections.defaultdict(list) for _ in range(len(methods))]

    all_trial_results = []


    for i in range(len(methods)):
        print(methods[i])
        for seed in tqdm(range(num_trials)):
            set_seed(seed)

            shuffle = np.random.permutation(len(all_labels))
            scaling_type = methods[i].get('scaling', (None, {}))[0]

            if scaling_type is not None:
                N_train = 2000
            else:
                N_train = 0
                
            shuffle = np.random.permutation(len(all_labels))

            remaining_N = len(all_labels) - N_train
            training_idx =  shuffle[:N_train]

            N_cal = int(p_cal * remaining_N)
            val_idx =  shuffle[N_train + N_cal:]
            N_val= remaining_N - N_cal

            N_tuning = int(N_cal*p_tuning)
            N_cal -= N_tuning

            tuning_idx = shuffle[N_train : N_train + N_tuning]
            cal_idx = shuffle[N_train + N_tuning: N_train + N_tuning + N_cal]
            
            
            training_scores = all_scores[training_idx]
            tuning_scores = all_scores[tuning_idx]
            cal_scores = all_scores[cal_idx]
            val_scores = all_scores[val_idx]
            
            

            training_labels = all_labels[training_idx]
            tuning_labels = all_labels[tuning_idx]
            cal_labels = all_labels[cal_idx]
            val_labels = all_labels[val_idx]

            training_simlarties = diversity[training_idx]
            tuning_simlarties = diversity[tuning_idx]
            cal_simlarties = diversity[cal_idx]
            val_simlarties = diversity[val_idx]
            

            
            conformal_llm = ConformalLM(epsilons=epsilons, 
                                        scaling_type = scaling_type, 
                                        scale_kwargs=  methods[i].get('scaling', (None, {}))[1],
                                        set_score_function_name = methods[i].get('scoring', 'none'),
                                        rejection = methods[i].get('rejection', False))
            
            
            if scaling_type is not None:
                conformal_llm.scaling(training_scores, training_labels)
            conformal_llm.tuning(tuning_scores, tuning_simlarties, tuning_labels )
            conformal_llm.calibrate_configs(cal_scores, cal_simlarties, cal_labels, alpha)
            trial_results = conformal_llm.evaluate(val_scores, val_simlarties, val_labels)
            all_trial_results.append(trial_results)
            
        for j, method_results in enumerate(all_trial_results):
            for k, v in method_results.items():
                all_results[i][k].append(np.array(v))
                
    combined_results = []
    for results in all_results:
        combined = {}
        for k, v in results.items():
            combined[k] = np.stack(v, axis=0)
        combined_results.append(combined)

    

test_conformal_llm()