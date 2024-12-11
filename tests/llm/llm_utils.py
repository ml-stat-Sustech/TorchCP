from datasets import load_dataset


def get_dataset(dataset_name="trivia_qa", mode="validation", max_predict_samples=None, starting_x=0):
    dataset = load_dataset(dataset_name, "rc")
    dataset = dataset[mode]
    if max_predict_samples is not None:
        dataset = dataset.select(range(starting_x, starting_x + max_predict_samples))
    return dataset