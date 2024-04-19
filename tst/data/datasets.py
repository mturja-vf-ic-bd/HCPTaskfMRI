import unittest
from src.data.datasets import get_segments, generate_segment_maps, generate_data_sets, store_data_into_disk
import torch


class TestGenerateSegmentMaps(unittest.TestCase):
    def testGetSegments(self):
        datadir = "/export_home/mturja/HCP_7tasks"
        task = "EMOTION"
        segment_dict = get_segments(datadir, task)
        print(segment_dict)

    def testGetSegmentMap(self):
        datadir = "/export_home/mturja/HCP_7tasks"
        segment_map = generate_segment_maps(datadir)
        print(segment_map)


class TestHCPTaskDataset(unittest.TestCase):
    def setUp(self):
        filepath = "/Users/mturja/PycharmProjects/DeepGraphKoopmanOperator/data/HCP_7tasks"
        self.datapath = filepath
        self.input_length = 16
        self.output_length = 4
        task_names = ["EMOTION"]

        self.dataset = generate_data_sets(
            input_length=self.input_length,
            output_length=self.output_length,
            datapath=self.datapath,
            jumps=1,
            task_names=task_names,
            mode="test"
        )

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), len(self.dataset.ts_indices))

    def test_sample_item(self):
        idx = 0
        sample = self.dataset.__getitem__(idx)
        num_roi = 360

        # Check if the returned tensors have the correct shapes
        x, y, label = sample
        self.assertEqual(x.shape, torch.Size([self.input_length, num_roi]))
        self.assertEqual(y.shape, torch.Size([self.output_length, num_roi]))


class TestHCPTaskStoreFunc(unittest.TestCase):
    def testStoreFunc(self):
        filepath = "/Users/mturja/PycharmProjects/DeepGraphKoopmanOperator/data/HCP_7tasks"
        writedir = "/Users/mturja/PycharmProjects/DeepGraphKoopmanOperator/data"
        input_length = 16
        output_length = 4
        task_names = ["EMOTION"]
        mode = "train"
        store_data_into_disk(filepath, input_length, output_length, 1, task_names, mode, 1, 32, writedir)
