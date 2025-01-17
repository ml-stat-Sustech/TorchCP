# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import re
import string

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, set_seed, StoppingCriteriaList

few_shot_qa = """Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
A: Sinclair Lewis
Q: Where in England was Dame Judi Dench born?
A: York
Q: In which decade did Billboard magazine first publish and American hit chart?
A: 30s
Q: From which country did Angola achieve independence in 1975?
A: Portugal
Q: Which city does David Soul come from?
A: Chicago
Q: Who won Super Bowl XX?
A: Chicago Bears
Q: Which was the first European country to abolish capital punishment?
A: Norway
Q: In which country did he widespread use of ISDN begin in 1988?
A: Japan
Q: What is Bruce Willis' real first name?
A: Walter
Q: Which William wrote the novel Lord Of The Flies?
A: Golding
Q: Which innovation for the car was developed by Prince Henry of Prussia in 1911?
A: Windshield wipers
Q: How is musician William Lee Conley better known?
A: Big Bill Broonzy
Q: How is Joan Molinsky better known?
A: Joan Rivers
Q: In which branch of the arts is Patricia Neary famous?
A: Ballet
Q: Which country is Europe's largest silk producer?
A: Italy
Q: The VS-300 was a type of what?
A: Helicopter
Q: At which university did Joseph Goebbels become a doctor of philosophy?
A: Heidelberg
Q: Which prince is Queen Elizabeth II's youngest son?
A: Edward
Q: When did the founder of Jehovah's Witnesses say the world would end?
A: 1914
Q: Who found the remains of the Titanic?
A: Robert Ballard
Q: Who was the only Spice Girl not to have a middle name?
A: Posh Spice
Q: What are the international registration letters of a vehicle from Algeria?
A: DZ
Q: How did Jock die in Dallas?
A: Helicopter accident
Q: What star sign is Michael Caine?
A: Pisces
Q: Who wrote the novel Evening Class?
A: Maeve Binchy
Q: Which country does the airline Air Pacific come from?
A: Fiji
Q: In which branch of the arts does Allegra Kent work?
A: Ballet
Q: Who had a 70s No 1 hit with Billy, Don't Be A Hero?
A: Bo Donaldson & The Heywoods
Q: Banting and Best pioneered the use of what?
A: Insulin
Q: Who directed the movie La Dolce Vita?
A: Federico Fellini
Q: Which country does the airline LACSA come from?
A: Costa Rica
Q: Who directed 2001: A Space Odyssey?
A: Stanley Kubrick
Q: Which is the largest of the Japanese Volcano Islands?
A: Iwo Jima
Q: Ezzard Charles was a world champion in which sport?
A: Boxing
Q: Who was the first woman to make a solo flight across the Atlantic?
A: Amelia Earhart
Q: Which port lies between Puget Sound and Lake Washington?
A: Seattle
Q: In which city were Rotary Clubs set up in 1905?
A: Chicago
Q: Who became US Vice President when Spiro Agnew resigned?
A: Gerald Ford
Q: In which decade of the 20th century was Billy Crystal born?
A: 1940s
Q: Which George invented the Kodak roll-film camera?
A: Eastman
Q: Which series had the characters Felix Unger and Oscar Madison?
A: The Odd Couple
Q: Who along with Philips developed the CD in the late 70s?
A: Sony
Q: Where is the multinational Nestle based?
A: Switzerland
Q: Do You Know Where You're Going To? was the theme from which film?
A: Mahogany
Q: 19969 was the Chinese year of which creature?
A: Rat
Q: In the 90s how many points have been awarded for finishing second in a Grand Prix?
A: 6
Q: Stapleton international airport is in which US state?
A: Colorado
Q: What was Kevin Kline's first movie?
A: Sophie's Choice
Q: Which actor had a Doberman Pinscher called Kirk?
A: William Shatner
Q: What day of the week was the Wall Street Crash?
A: Thursday
Q: The US signed a treaty with which country to allow the construction of the Panama Canal?
A: Columbia
Q: What was Prince's last No 1 of the 80s?
A: Batdance
Q: Man In The Mirror first featured on which Michel Jackson album?
A: Bad
Q: Where was the first battle with US involvement in the Korean War?
A: Suwon
Q: On which Caribbean island did Princess Diana spend he first Christmas after her divorce was announced?
A: Barbuda
Q: In which decade was Arnold Schwarzenegger born?
A: 1950s
Q: Which musical featured the song Thank Heaven for Little Girls?
A: Gigi
Q: The Queen Elizabeth liner was destroyed by fire in the 70s in which harbour?
A: Hong Kong
Q: What breed of dog did Columbo own?
A: Basset hound
Q: What was the first movie western called?
A: Kit Carson
Q: Which Oscar-winning actress was born on exactly the same day as actress Lindsay Wagner?
A: Meryl Streep
Q: Which Amendment to the Constitution brought in prohibition in 1920?
A: 18th
Q: Which oil scandal hit the US in 1924?
A: Teapot Dome Scandal
Q: Phil Collins appeared in which Spielberg film with Robin Williams?
A: Hook""".split("\n")

import torch
from transformers import StoppingCriteria


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


def get_dataset(dataset_name="trivia_qa", mode="validation", max_predict_samples=None, starting_x=0):
    dataset = load_dataset(dataset_name, "rc")
    dataset = dataset[mode]
    if max_predict_samples is not None:
        dataset = dataset.select(range(starting_x, starting_x + max_predict_samples))
    return dataset


def preprocess_data(dataset_name, model_path, output_path):
    dataset = get_dataset(dataset_name, max_predict_samples=30)

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

        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stop_ids=stop_word_ids, input_length=input_ids.shape[1])]).to(dtype=torch.long)
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

            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores,
                                                                normalize_logits=True)
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
            print(example['question'])
            print(predictions)
            print(scores)
            print(labels)
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
    for sample in tqdm(dataset, desc="Generating samples"):
        result = run_triviaqa(sample)
        result = fix_eos(result)
        results.append(result)

    all_labels, all_scores = compute_qa_scores(results)
    diversity = compute_diversity(results)

    np.savez(output_path, labels=all_labels, scores=all_scores, diversity=diversity)


def split_indices(total_samples, N_train, p_cal, p_tuning):
    # Create random permutation of indices
    shuffle = np.random.permutation(total_samples)

    # Calculate sizes for each split
    remaining_samples = total_samples - N_train
    N_cal = int(p_cal * remaining_samples)
    N_tuning = int(p_tuning * remaining_samples)

    return shuffle[:N_train], shuffle[N_train:N_train + N_tuning], shuffle[
                                                                   N_train + N_tuning:N_train + N_tuning + N_cal], shuffle[
                                                                                                                   N_train + N_tuning + N_cal:]
