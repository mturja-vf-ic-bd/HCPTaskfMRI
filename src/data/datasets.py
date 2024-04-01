import unittest
from collections import defaultdict

import torch
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import List
import pandas as pd


def retrieve_hcptask_filepaths(datadir, task_names, subject_ids):
    subject_ids = set(subject_ids)
    path_dict = []
    for task in task_names:
        task_dir = os.path.join(datadir, task + "_csv_formats")
        path_dict.extend([os.path.join(task_dir, f) for f in os.listdir(task_dir)
                          if f.endswith("csv") and "LR" in f and f.split("_")[0] in subject_ids])
    return path_dict


def get_segments(datadir, task):
    task_label_file = os.path.join(
        datadir,
        task + "_csv_formats",
        task + "_label.xlsx"
    )

    # task_label_file = os.path.join(
    #     "/Users/mturja/task_labels",
    #     task + "_label.xlsx"
    # )

    # Parsing segments to generate a dictionary like:
    # {"non_emo": [(13, 38), (71, 96), (130, 155)], "fear_emo": [(42, 67), (101, 126)]}
    # skips the `rest` segments
    labels = pd.read_excel(task_label_file, header=None).iloc[1]
    left, right = 2, 2
    segment_dict = defaultdict(list)
    for i in range(3, len(labels)):
        if labels[i] == labels[i - 1]:
            right = i
        else:
            if labels[left] != "rest":
                segment_dict[labels[left]].append((left - 2, right - 2))
            left, right = i, i
    return segment_dict


def generate_segment_maps(datadir):
    segment_maps = {}
    ALL_TASKS = ["EMOTION", "GAMBLING", "LANGUAGE",
                 "MOTOR", "RELATION", "SOCIAL", "WM"]
    for task in ALL_TASKS:
        segment_maps[task] = get_segments(datadir, task)
    return segment_maps


def get_ids(datadir, task):
    task_dir = os.path.join(datadir, task + "_csv_formats")
    return [f.split("_")[0] for f in os.listdir(task_dir) if f.endswith("csv") and "LR" in f]


def get_class_to_label(segment_maps):
    # This function collects all the subclass labels from segment_maps and
    # assign an integer label to it for classification purpose.

    # For example, the segment_maps look like the following:
    # {
    # 'EMOTION': {'non_emo': [(13, 38), ... ], 'fear_emo': [(42, 67), ... ]}),
    # 'GAMBLING': {'loss_gamb': [(9, 47), ... ], 'win_gamb': [(70, 108), ... ]}) ...
    # }

    # This function will return
    # {"non_emo": 0, "fear_emo": 1, "loss_gamb": 2, "win_gamb": 3, ... }

    all_classes = [class_name for segments in segment_maps.values()
                   for class_name in segments.keys()]
    class_to_labels = {class_name: label
                       for label, class_name in enumerate(all_classes)}
    return class_to_labels


class HCPTaskDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            subject_ids: List[str],
            input_length: int,  # num of input steps
            datapath: str = None,
            jumps: int = 1,
            task_names: List[str] = None,
            shuffle: bool = False
    ):
        super(HCPTaskDataset, self).__init__()
        self.data_dir = datapath
        self.file_paths = retrieve_hcptask_filepaths(
            datapath, task_names, subject_ids)
        print("Number of files: ", len(self.file_paths))
        self.input_length = input_length
        segment_maps = generate_segment_maps(datapath)
        class_to_label = get_class_to_label(segment_maps)

        # Cache data in memory since datasize is not that big
        self.fmri_data = [np.loadtxt(file_path, delimiter="\t")
                          for file_path in self.file_paths]
        self.ts_indices = []
        self.class_labels = []

        for i in range(len(self.fmri_data)):
            filepath = self.file_paths[i]
            # Find the time segments for this file
            segments = next((segment_maps[task] for task in segment_maps.keys()
                             if task in filepath), None)
            for class_name, seg in segments.items():
                for left, right in seg:
                    for j in range(left, right - input_length + 1, jumps):
                        if len(self.fmri_data[i]) - j < input_length:
                            # Skipping short windows to make sure
                            # all samples have the same number of time points
                            continue
                        self.ts_indices.append((i, j))
                        self.class_labels.append(class_to_label[class_name])
        print(f"Total Samples: {len(self.ts_indices)}")
        # Shuffling training data segments
        if shuffle:
            random.seed(123)
            indices = np.arange(len(self.ts_indices))
            np.random.shuffle(indices)
            self.ts_indices = [self.ts_indices[idx] for idx in indices]
            self.class_labels = [self.class_labels[idx] for idx in indices]

        print("Loaded Dataset!")

    def __len__(self):
        return len(self.ts_indices)

    def __getitem__(self, idx):
        # Convert to PyTorch tensor
        i, j = self.ts_indices[idx]
        x = self.fmri_data[i][j:j + self.input_length]
        label = self.class_labels[idx]

        return torch.from_numpy(x).float(), \
               torch.LongTensor([label])


def generate_data_sets(datapath,
                       input_length,
                       jumps,
                       task_names=None,
                       mode="train",
                       random_state=1):
    assert mode in ["train", "valid", "test", "all"], \
        f"mode can only be one of train, valid, test, and, all"
    ALL_TASKS = ["EMOTION", "GAMBLING", "LANGUAGE",
                 "MOTOR", "RELATION", "SOCIAL", "WM"]
    if not task_names:
        task_names = ALL_TASKS
    subject_ids = {}

    # Reading all subject ids for each task
    for task in task_names:
        parent = os.path.join(datapath, task + "_csv_formats")
        subject_ids[task] = [f.split("_")[0] for f in os.listdir(parent)
                          if f.endswith("csv") and "LR" in f]

    # Generate subject ids that have sample for every task
    common_ids = set(subject_ids["EMOTION"])
    for k, v in subject_ids.items():
        common_ids = common_ids.intersection(set(v))

    # Creating dataset using only common_ids.
    common_ids = list(common_ids)
    common_ids.sort()
    train_ids, test_ids = train_test_split(
        common_ids,
        test_size=0.4,
        shuffle=True,
        random_state=random_state)
    valid_ids = train_ids[0:int(len(train_ids) * 0.2)]
    train_ids = train_ids[int(len(train_ids) * 0.2):]
    print(train_ids, valid_ids)
    # Create Dataset Object
    if mode == "train":
        train_set = HCPTaskDataset(
            train_ids, input_length,
            datapath,
            jumps, task_names, True)
        return train_set
    elif mode == "valid":
        valid_set = HCPTaskDataset(
            valid_ids, input_length,
            datapath,
            jumps, task_names, True)
        return valid_set
    elif mode == "test":
        test_set = HCPTaskDataset(
            test_ids, input_length,
            datapath,
            jumps, task_names, False)
        return test_set
    else:
        all_set = HCPTaskDataset(
            common_ids,
            input_length, datapath,
            jumps, task_names, False)
        return all_set


def generate_data_from_single_file(
        datapath,
        id, task, input_length,
        output_length):
    dataset = HCPTaskDataset(
        [id], input_length,
        output_length, datapath,
        1, [task], False)
    return dataset



