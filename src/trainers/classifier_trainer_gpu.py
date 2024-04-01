# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main script for training KNF."""
import os
import random
import time
from tqdm import tqdm
from collections import OrderedDict
import re

from src.model.cnn.model import FMRI_CNN
from src.data.datasets import generate_data_sets

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils import data
from torch.utils.data import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from src.trainers.utils import train_one_epoch, eval_one_epoch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def ddp_setup(rank: int, world_size: int):
    """
    Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12375"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank, world_size, seed,
         input_length, num_feats,
         num_layers, hidden_channel,
         batch_size, learning_rate,
         mode, num_epochs, dropout):
    ddp_setup(rank, world_size)
    random.seed(seed)  # python random generator
    np.random.seed(seed)  # numpy random generator

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_dir = {
        "M4": "data/M4",
        "Cryptos": "data/Cryptos",
        "Task": "data/task",
        "megatrawl": "data/megatrawl",
        "hcptask": "/home/mturja/HCP_7tasks"
    }
    dataset_name = "hcptask"
    data_dir = data_dir[dataset_name]
    print(f"Dataset directory: {data_dir}")
    train_set = generate_data_sets(
        input_length=input_length,
        mode="train",
        jumps=1,
        task_names=None,
        datapath=data_dir,
        random_state=seed
    )
    valid_set = generate_data_sets(
        input_length=input_length,
        mode="valid",
        jumps=1,
        task_names=None,
        datapath=data_dir,
        random_state=seed
    )
    test_set = generate_data_sets(
        input_length=input_length,
        mode="test",
        jumps=1,
        task_names=None,
        datapath=data_dir,
        random_state=seed
    )

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False,
        sampler=DistributedSampler(train_set)
    )
    valid_loader = data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        sampler=DistributedSampler(valid_set)
    )
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    model_name = (
            "HCPTaskClassifier_"
            + str(dataset_name)
            + f"_seed{seed}_"
              f"lr{learning_rate}_"
              f"inp{input_length}"
    )
    print(model_name)
    model = FMRI_CNN(
        num_layers=num_layers,
        in_channels=num_feats,
        hidden_channel=hidden_channel,
        num_classes=21,
        num_dense_layers=3,
        dropout=dropout
    ).to(rank)
    results_dir = dataset_name + "_results/"
    if os.path.exists(results_dir + model_name + ".pth"):
        ckpt = torch.load(results_dir + model_name + ".pth",
                          map_location={"cuda:0": f"cuda:{rank}"})
        state_dict = ckpt["model"]
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k, v in state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict[k] = v
        model.load_state_dict(model_dict)
        model = DDP(model, device_ids=[rank])
        last_epoch = ckpt["epoch"]
        learning_rate = ckpt["lr"]
        print("Resume Training")
        print("last_epoch:", last_epoch, "learning_rate:", learning_rate)
    else:
        last_epoch = 0
        model = DDP(model, device_ids=[rank])
        if rank == 0 and not os.path.exists(results_dir):
            os.mkdir(results_dir)
        model = DDP(model, device_ids=[rank])
        print("New model")

    best_model = model
    print("number of params:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.95
    )  # stepwise learning rate decay

    all_train_rmses, all_eval_rmses = [], []
    best_eval_loss = 1e6

    if mode == "train":
        for epoch in tqdm(range(last_epoch, num_epochs)):
            start_time = time.time()
            train_loss = train_one_epoch(
                train_loader,
                model,
                optimizer,
                rank=rank)
            eval_loss, y_pred, y_true = eval_one_epoch(
                valid_loader,
                model,
                rank=rank)

            if eval_loss < best_eval_loss and rank == 0:
                best_eval_loss = eval_loss
                best_model = model
                torch.save({"model": best_model.state_dict(),
                            "epoch": epoch, "lr": get_lr(optimizer)},
                           results_dir + model_name + ".pth")

            all_train_rmses.append(train_loss)
            all_eval_rmses.append(eval_loss)

            if torch.isnan(train_loss) or torch.isnan(eval_loss):
                raise ValueError("The model generate NaN values")

            epoch_time = time.time() - start_time
            scheduler.step()
            print(f'Epoch {epoch + 1} | T: {epoch_time / 60:0.2f} | '
                  f'Train Loss: {train_loss:0.3f}, '
                  f'Eval Loss: {eval_loss:0.3f}', flush=True)
    elif mode == "test":
        print(f"Test set length: {len(test_loader)}")
        test_loss, test_preds, test_labels = eval_one_epoch(
            test_loader, best_model, rank=0)

        test_preds = np.argmax(test_preds, axis=1)
        print(classification_report(test_labels, test_preds))
        if rank == 0:
            torch.save(
                {
                    "test_preds": test_preds,
                    "test_labels": test_labels,
                    "test_loss": test_loss
                }, results_dir + model_name + ".pt")
    destroy_process_group()


def run_training():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--max_epochs", nargs="?", type=int, default=100)
    parser.add_argument("--lr", default=0.0003, type=float, help="learning rate")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file for prediction")
    parser.add_argument("--from_ckpt", dest='from_ckpt', default=False, action='store_true')
    parser.add_argument("--exp_name", type=str, default="emo",
                        help="Name you experiment")
    parser.add_argument("--write_dir", type=str, default="log/varKNF")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_feats", type=int, default=360)
    parser.add_argument("--input_length", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--hidden_channel", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--dataset", type=str, default="emotion")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.set_defaults(
        use_revin=False,
        use_instancenorm=False)

    args = parser.parse_args()
    print("Hyper-parameters:")
    for k, v in vars(args).items():
        print("{} -> {}".format(k, v))
    world_size = torch.cuda.device_count()
    mp.spawn(main,
             args=(
                 world_size, args.seed,
                 args.input_length, args.num_feats,
                 args.num_layers, args.hidden_channel,
                 args.batch_size, args.lr, args.mode,
                 args.max_epochs, args.dropout),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    start_time = time.time()
    run_training()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed {elapsed_time // 3600}h "
          f"{(elapsed_time % 3600) // 60}m "
          f"{((elapsed_time % 3600) % 60)}s")
