import argparse

import torch
import torch.nn.functional as F
from transformers import set_seed

from torchcp.classification.score import APS
from torchcp.graph.trainer import CFGNNTrainer
from torchcp.classification.predictor import SplitPredictor
from examples.utils import build_gnn_model, build_transductive_gnn_data


def train_transductive(model, optimizer, graph_data, train_idx):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    training_loss = F.cross_entropy(out[train_idx], graph_data.y[train_idx])
    training_loss.backward()
    optimizer.step()


def evaluate_model(best_logits, calib_eval_idx, predictor, graph_data, args):
    coverage_list = []
    size_list = []
    calib_num = min(1000, int(calib_eval_idx.shape[0] / 2))
    calib_fraction = 0.5
    
    for _ in range(100):
        eval_perms = torch.randperm(calib_eval_idx.size(0))
        eval_calib_idx = calib_eval_idx[eval_perms[:int(calib_num * calib_fraction)]]
        eval_test_idx = calib_eval_idx[eval_perms[int(calib_num * calib_fraction):]]

        predictor.calculate_threshold(
            best_logits[eval_calib_idx], graph_data.y[eval_calib_idx], args.alpha)
        pred_sets = predictor.predict_with_logits(best_logits[eval_test_idx])

        coverage = predictor._metric('coverage_rate')(pred_sets, graph_data.y[eval_test_idx])
        size = predictor._metric('average_size')(pred_sets, graph_data.y[eval_test_idx])

        coverage_list.append(coverage)
        size_list.append(size)

    return (torch.mean(torch.tensor(coverage_list)), 
            torch.mean(torch.tensor(size_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    args = parser.parse_args()

    set_seed(seed=args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loading dataset and a model for transductive
    data_name = 'cora_ml'
    graph_data, label_mask, train_idx, val_idx, test_idx = build_transductive_gnn_data(data_name)
    graph_data = graph_data.to(device)

    model = build_gnn_model('GCN')(
        graph_data.x.shape[1], 64, graph_data.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    # Initial training
    n_epochs = 200
    for _ in range(n_epochs):
        train_transductive(model, optimizer, graph_data, train_idx)

    model.eval()
    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index)

    # Split calib/test sets
    calib_test_idx = test_idx
    calib_num = min(1000, int(test_idx.shape[0] / 2))
    rand_perms = torch.randperm(calib_test_idx.size(0))
    calib_train_idx = calib_test_idx[rand_perms[:int(calib_num * 0.5)]]
    calib_eval_idx = calib_test_idx[rand_perms[int(calib_num * 0.5):]]

    predictor = SplitPredictor(APS(score_type="softmax"))
    # Basic results
    ce_coverage, ce_size = evaluate_model(logits, calib_eval_idx, predictor, graph_data, args)

    # Run ConfTr experiment
    # breakpoint()
    graph_data['train_idx'] = train_idx
    graph_data['val_idx'] = test_idx
    graph_data['calib_train_idx'] = calib_train_idx
    confmodel_conftr = CFGNNTrainer(model,
                                    graph_data)
    best_logits = confmodel_conftr.train()
    conftr_coverage, conftr_size = evaluate_model(best_logits, calib_eval_idx, predictor, graph_data, args)

    print("\nResults Comparison:")
    print(f"CrossEntropy - Coverage: {ce_coverage:.4f}, Size: {ce_size:.4f}")
    print(f"ConfTr       - Coverage: {conftr_coverage:.4f}, Size: {conftr_size:.4f}")