import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import set_seed

from torchcp.classification.score import APS
from torchcp.graph.score import DAPS
from torchcp.graph.trainer import ConfGNN
from torchcp.classification.loss import ConfTr
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import THR
from torchcp.graph.utils.metrics import Metrics
from examples.utils import build_gnn_model, build_transductive_gnn_data, build_inductive_gnn_data


def train_transductive(model, optimizer, graph_data, train_idx):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    training_loss = F.cross_entropy(out[train_idx], graph_data.y[train_idx])
    training_loss.backward()
    optimizer.step()


def run_experiment(confmodel, criterion, logits, graph_data, train_idx, val_idx, calib_train_idx, 
                  calib_eval_idx, device, args, use_conftr=True):
    epochs = 5000
    best_valid_size = 10000
    best_logits = logits
    
    optimizer = torch.optim.Adam(confmodel.parameters(), weight_decay=5e-4, lr=0.001)
    predictor = SplitPredictor(APS(score_type="softmax"))
    predictor._device = device

    print(f'Starting {"ConfTr" if use_conftr else "CrossEntropy"} training...')
    for epoch in tqdm(range(1, epochs + 1)):
        confmodel.train()
        optimizer.zero_grad()
        
        adjust_logits = confmodel(logits, graph_data.edge_index)
        loss = F.cross_entropy(adjust_logits[train_idx], graph_data.y[train_idx])
        
        if use_conftr and epoch > 1000:
            loss += criterion(adjust_logits[calib_train_idx], graph_data.y[calib_train_idx])

        loss.backward()
        optimizer.step()

        # Validation Stage
        confmodel.eval()
        with torch.no_grad():
            adjust_logits = confmodel(logits, graph_data.edge_index)

        size_list = []
        for _ in range(10):
            val_perms = torch.randperm(val_idx.size(0))
            valid_calib_idx = val_idx[val_perms[:int(len(val_idx) / 2)]]
            valid_test_idx = val_idx[val_perms[int(len(val_idx) / 2):]]

            predictor.calculate_threshold(
                adjust_logits[valid_calib_idx], graph_data.y[valid_calib_idx], args.alpha)
            pred_sets = predictor.predict_with_logits(adjust_logits[valid_test_idx])
            size = predictor._metric('average_size')(pred_sets, graph_data.y[valid_test_idx])
            size_list.append(size)

        eff_valid = torch.mean(torch.tensor(size_list))

        if eff_valid < best_valid_size:
            best_valid_size = eff_valid
            best_logits = adjust_logits

    return evaluate_model(best_logits, calib_eval_idx, predictor, graph_data, args)


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

    train_calib_idx = calib_train_idx[int(len(calib_train_idx) / 2):]
    train_test_idx = calib_train_idx[:int(len(calib_train_idx) / 2)]

    # Run CrossEntropy experiment
    confmodel_ce = ConfGNN(base_model='GCN',
                       output_dim=graph_data.y.max().item() + 1,
                       confnn_hidden_dim=64).to(device)
    
    criterion = ConfTr(weight=1.0,
                    predictor=SplitPredictor(score_function=APS(score_type="softmax", randomized=False)),
                    alpha=args.alpha,
                    fraction=0.5,
                    loss_type="cfgnn",
                    target_size=0)

    ce_coverage, ce_size = run_experiment(
        confmodel_ce, criterion, logits, graph_data, train_idx, val_idx,
        calib_train_idx, calib_eval_idx, device, args, use_conftr=False
    )

    # Run ConfTr experiment
    confmodel_conftr = ConfGNN(base_model='GCN',
                           output_dim=graph_data.y.max().item() + 1,
                           confnn_hidden_dim=64).to(device)

    conftr_coverage, conftr_size = run_experiment(
        confmodel_conftr, criterion, logits, graph_data, train_idx, val_idx,
        calib_train_idx, calib_eval_idx, device, args, use_conftr=True
    )

    print("\nResults Comparison:")
    print(f"CrossEntropy - Coverage: {ce_coverage:.4f}, Size: {ce_size:.4f}")
    print(f"ConfTr       - Coverage: {conftr_coverage:.4f}, Size: {conftr_size:.4f}")