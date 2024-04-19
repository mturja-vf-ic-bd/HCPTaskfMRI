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

from sklearn.metrics import classification_report
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from collections import OrderedDict, Counter
import re


import numpy as np
import torch
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from src.data.datasets import generate_data_sets, generate_preloaded_dataset
from src.model.cnn.model import FMRI_CNN
from src.model.knf.knf_cnn import KNF_CNN
from src.model.knf.knf_cnn_global import KNF_CNN_GLOBAL
from src.trainers.utils import train_one_knf_epoch, eval_one_knf_epoch


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


def print_epoch_statement(train_loss, eval_loss):
    output = "|"
    for k, v in train_loss.items():
        output += f"{k}: ({v:0.3f}, {eval_loss[k]:0.3f}), "
    return output


def main(rank,
         world_size,
         seed,
         backbone_layers,
         backbone_hidden_channel,
         cls_layers,
         cls_hidden_channel,
         transformer_layers,
         transformer_hidden,
         add_global_operator,
         input_length,
         output_length,
         num_feats,
         num_classes, task_names,
         batch_size, learning_rate,
         mode, num_epochs, dropout, model_str):

    ddp_setup(rank, world_size)
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
        # "hcptask": "/Users/mturja/PycharmProjects/DeepGraphKoopmanOperator/data/HCP_7tasks"
        "hcptask": "/home/mturja/HCP_7task_preloaded"
    }
    dataset_name = "hcptask"
    data_dir = data_dir[dataset_name]
    print(f"Dataset directory: {data_dir}")

    train_set = generate_preloaded_dataset(
        data_dir, input_length, output_length,
        task_names, "train", seed)
    valid_set = generate_preloaded_dataset(
        data_dir, input_length, output_length,
        task_names, "valid", seed)
    test_set = generate_preloaded_dataset(
        data_dir, input_length, output_length,
        task_names, "test", seed)

    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False,
        sampler=DistributedSampler(train_set))
    valid_loader = data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        sampler=DistributedSampler(valid_set))
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        sampler=DistributedSampler(test_set))

    model_name = (
            f"HCPTask_{model_str}"
            + str(dataset_name)
            + f"_seed{seed}_"
              f"lr{learning_rate}_"
              f"tfchannel{transformer_hidden}_"
              f"tflayer{transformer_layers}_"
              f"clslayer{cls_layers}_"
              f"clschannel{cls_hidden_channel}_"
              f"global{add_global_operator}_"
              f"inp{input_length}_"
              f"pred{output_length}"
    )
    print(model_name)
    model = KNF_CNN(
        backbone_layers=backbone_layers,
        backbone_hidden_channel=backbone_hidden_channel,
        cls_layers=cls_layers,
        cls_hidden_channel=cls_hidden_channel,
        transformer_layers=transformer_layers,
        transformer_hidden=transformer_hidden,
        dropout=dropout,
        time_points=input_length,
        num_nodes=num_feats,
        num_classes=num_classes,
        add_global_attention=add_global_operator
    ).to(rank)

    print(model)
    results_dir = dataset_name + "_results/"

    if os.path.exists(results_dir + model_name + ".pth"):
        ckpt = torch.load(results_dir + model_name + ".pth")
        state_dict = ckpt["model"]
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k, v in state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict[k] = v
        model.load_state_dict(model_dict)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        last_epoch = ckpt["epoch"]
        learning_rate = ckpt["lr"]
        print("Resume Training")
        print("last_epoch:", last_epoch, "learning_rate:", learning_rate)
    else:
        last_epoch = 0
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        print("New model")

    print("number of params:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.95
    )  # stepwise learning rate decay

    best_eval_loss = 1e6
    best_model = model

    if mode == "train":
        for epoch in tqdm(range(last_epoch, num_epochs)):
            start_time = time.time()
            train_loss = train_one_knf_epoch(
                train_loader,
                model,
                optimizer,
                rank=rank)
            eval_loss, _, _, _, _ = eval_one_knf_epoch(
                valid_loader,
                model,
                rank=rank)

            if eval_loss["loss"] <= best_eval_loss:
                best_eval_loss = eval_loss["loss"]
                best_model = model
                torch.save({"model": best_model.state_dict(),
                            "epoch": epoch, "lr": get_lr(optimizer)},
                           results_dir + model_name + ".pth")

            epoch_time = time.time() - start_time
            scheduler.step()
            statement = print_epoch_statement(train_loss, eval_loss)
            print(f'Epoch {epoch + 1} | T: {epoch_time / 60:0.2f} | {statement}', flush=True)
    elif mode == "test":
        print(f"Test set length: {len(test_loader)}")
        test_losses, _, test_preds, test_labels, test_smape = eval_one_knf_epoch(
            test_loader, best_model, rank=rank)

        test_preds = np.argmax(test_preds, axis=1)
        print(f"Test label shape: {test_labels.shape}")
        print(f"Test label dist: {Counter(test_preds)}")
        print(f"Test true labels: {Counter(test_labels.squeeze())}")
        print(test_preds)
        test_loss = test_losses["loss"]
        print(f"Test Loss: {test_loss:0.3f}, Test SMAPE: {test_smape:0.4f}")
        print(classification_report(test_labels, test_preds))
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
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file for prediction")
    parser.add_argument("--from_ckpt", dest='from_ckpt', default=False, action='store_true')
    parser.add_argument("--exp_name", type=str, default="emo",
                        help="Name you experiment")
    parser.add_argument("--write_dir", type=str, default="log/knf_cnn")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_feats", type=int, default=360)
    parser.add_argument("--input_length", type=int, default=16)
    parser.add_argument("--output_length", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--backbone_hidden_channel", type=int, default=64)
    parser.add_argument("--cls_hidden_channel", type=int, default=64)
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_hidden_channel", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--add_global_operator",
                        dest='add_global_operator', action='store_true')
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="fmri_cnn")
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
                args.num_layers,
                args.backbone_hidden_channel,
                8,
                args.cls_hidden_channel,
                args.transformer_layers,
                args.transformer_hidden_channel,
                args.add_global_operator,
                args.input_length,
                args.output_length,
                args.num_feats,
                21,
                None,
                args.batch_size,
                args.dropout,
                args.mode,
                args.max_epochs,
                args.lr,
                args.model),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    start_time = time.time()
    run_training()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed {elapsed_time // 3600}h "
          f"{(elapsed_time % 3600) // 60}m "
          f"{((elapsed_time % 3600) % 60)}s")
