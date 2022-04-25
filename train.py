import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader
import os
from os.path import join, exists
from tqdm import tqdm
from genGraphData import TLDataSet, collate
from model import LTLfNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from argparse import ArgumentParser

parser = ArgumentParser(description='Train TLNet')
parser.add_argument('--device', type=int, default=0, help="GPU number")
parser.add_argument('--model', type=str, default="LTLfNet", help="Used model", choices=["LTLfNet"])
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
parser.add_argument('--ts', action='store_true', help="Whether test")
parser.add_argument('--ne', action='store_true', help="Whether add next edge")
parser.add_argument('--ud', action='store_true', help="Whether edge is undirected")
parser.add_argument('--rd', type=int, default=1, help="data type")
args = parser.parse_args()
device_num = args.device

# para
epochs = 256
lr = args.lr
clip_grads = True
early = 30
bs = 128
method = args.model
next_edge = args.ne
undirected = args.ud
if method == "LTLfNet":
    mod = LTLfNet
else:
    raise ValueError(f"Unexpected method: {method}.")
# endif

device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
bestPath = ""
time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
model_dir_name = f"{time_str}_lr_{lr}_method_{method}_bs_{bs}_ne_{int(next_edge)}_ud_{int(undirected)}_rd_{int(rand_data)}"
model_dir = join('model', model_dir_name)
os.makedirs(model_dir)
if not args.ts and not exists(model_dir):
    os.makedirs(model_dir)
if not args.ts:
    print(f"Save model at {model_dir}.")

def plotCurve(valLosses):
    plt.figure()
    plt.xlabel('Training step')
    plt.ylabel('Validation Loss')
    plt.title("Learning Curve")
    plt.grid()
    plt.plot(range(1, len(valLosses) + 1), valLosses, 'o-', color="r")
    plt.savefig(join(model_dir, 'train_curve.jpg'))
    plt.show()

def get_time_str(ts) -> str:
    day = ts // 86400
    hour = ts % 86400 // 3600
    minute = ts % 3600 // 60
    sec = ts % 60
    if day > 0:
        return f"{day:.0f} d {hour:.0f} h {minute:.0f} m {sec:.0f} s"
    elif hour > 0:
        return f"{hour:.0f} h {minute:.0f} m {sec:.0f} s"
    elif minute > 0:
        return f"{minute:.0f} m {sec:.0f} s"
    else:
        return f"{sec:.0f} s"

def train(trPath, devPath):
    global bestPath
    train_dataset = TLDataSet(trPath, next_edge, undirected)
    train_loader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate, shuffle=True)
    dev_dataset = TLDataSet(devPath, next_edge, undirected)
    dev_loader = DataLoader(dev_dataset, batch_size=bs, collate_fn=collate, shuffle=False)
    model = mod(device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = F.nll_loss  

    best_model, best_loss = None, None
    val_loss = []
    # early stop
    epsilon = 0
    n = len(train_loader)
    check_time = 4  
    val_time = 2 
    total_step = 0
    epoch_time = []   
    for epoch in range(epochs):
        start = time.time()
        print('Epoch %d / %d' % (epoch + 1, epochs))
        step = 0
        for node_list, edge_index, y, batch_index in train_loader:
            model.train()
            edge_index, y, batch_index = edge_index.to(device), y.to(device), batch_index.to(device)
            optimizer.zero_grad()
            out = model(node_list, batch_index)
            loss = criterion(out, y)
            # endif
            loss.backward()
            if clip_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            step += 1
            if step % (n // check_time) == 0:
                total_step += 1
                print('Step %d / %d. Train Loss: %4f' % (step, n, loss.cpu().item()))
                if total_step % val_time == 0:
                    model.eval()
                    dev_loss = 0
                    acc_cnt = 0
                    for node_list, edge_index, y, batch_index in tqdm(dev_loader, desc='Validating', ncols=80):
                        edge_index, y, batch_index = edge_index.to(device), y.to(device), batch_index.to(device)
                        out = model(node_list, batch_index)
                        prediction = out
                        # endif
                        loss = criterion(prediction, y)
                        dev_loss += loss.cpu().item()
                        for i, p in enumerate(prediction):
                            p = p[0] < p[1]
                            acc_cnt += 1 if p == y[i] else 0
                        # endfor
                    # endfor
                    print(f'Validation Done. Dev Loss = {dev_loss:.2f}, '
                          f'Best Loss = {np.inf if best_loss is None else best_loss:.2f}, '
                          f'Acc is {acc_cnt / len(dev_dataset) * 100: .2f}%')
                    val_loss.append(dev_loss)
                    if best_loss is None or dev_loss <= best_loss:
                        best_model = model
                        best_loss = dev_loss
                        bestPath = join(model_dir, 'model_{%s}-step{%d}-lr{%.4f}-early{%d}-loss{%.2f}.pth' % (
                            method, total_step, lr, early, dev_loss))
                        torch.save(best_model.state_dict(), bestPath)
                        print(f"Best model save at {bestPath}.")
                        epsilon = 0
                    else:
                        epsilon += 1
                        if epsilon >= early:
                            break
                        # endif
                    # endif
                # endif
            # endif
        # endfor
        end = time.time()
        elapsed = end - start
        print(f"Time elapsed: {get_time_str(elapsed)}")
        epoch_time.append(elapsed)
        if epsilon >= early:
            print(f"Done due to early stopping.")
            break
        # endif
    # endfor
    plotCurve(val_loss)
    print(f"Done training. Best model save at {bestPath}.")
    print(f"Avg training time: {get_time_str(np.mean(epoch_time))}.")
    print(f"Total time trained: {get_time_str(np.sum(epoch_time))}")


def test(model_path, test_path):
    test_begin = time.time()
    model = mod(device).to(device)
    model.load_state_dict(torch.load(model_path))
    ts_dataset = TLDataSet(test_path, next_edge, undirected)
    TP, FP, TN, FN = 0, 0, 0, 0

    for data in tqdm(ts_dataset, ncols=80, desc='Testing'):
        node_list, edge_index, y = data
        if method == "GCN":
            out = model(node_list, torch.LongTensor(edge_index).to(device))
        else:
            out = model(node_list)
        # endif
        predict = bool(out[0][0] < out[0][1])
        expect = y == 1
        if expect and expect == predict:
            TP += 1
        elif expect and expect != predict:
            FN += 1
        elif not expect and expect == predict:
            TN += 1
        else:
            FP += 1
    # endfor
    Acc, Pre, Rec, F1 = (TP + TN) / (TP + TN + FP + FN), \
                        TP / (TP + FP), \
                        TP / (TP + FN), \
                        (2 * TP) / (2 * TP + FP + FN)
    test_end = time.time()
    print('Average test time: ', (test_end - test_begin) / (TP + TN + FP + FN))
    print('Total   : (TP, TN, FP, FN) = (%d, %d, %d, %d)' % (TP, TN, FP, FN))
    print('Total   : (Acc, P, R, F1) = (%.4f, %.4f, %.4f, %.4f)' % (Acc, Pre, Rec, F1))


if __name__ == '__main__':
    if rand_data == 1:
        trP = 'data/LTLfSATUNSAT-{and-or-not-F-G-X-until}-100-contrasive-[20,100]/train.json'
        devP = 'data/LTLfSATUNSAT-{and-or-not-F-G-X-until}-100-contrasive-[20,100]/dev.json'
        tsP = 'data/LTLfSATUNSAT-{and-or-not-F-G-X-until}-100-random-[20,100]/test.json'
    elif rand_data == 2:
        trP = 'data/LTLfSATUNSAT-{and-or-not-F-G-X-until}-100-random-[20,100]/train.json'
        devP = 'data/LTLfSATUNSAT-{and-or-not-F-G-X-until}-100-random-[20,100]/dev.json'
        tsP = 'data/LTLfSATUNSAT-{and-or-not-F-G-X-until}-100-random-[20,100]/test.json'
    train(trP, devP)
    test(bestPath, tsP)
