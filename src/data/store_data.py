from src.data.datasets import store_data_into_disk

if __name__ == '__main__':
    datapath = "/home/mturja/HCP_7tasks"
    input_length = 16
    output_length = 4
    jumps = 1
    seed = 900
    for mode in ["train", "valid", "test"]:
        store_data_into_disk(
                datapath,
                input_length,
                output_length,
                jumps,
                task_names=None,
                mode=mode,
                random_state=seed,
                batch_size=512,
                writedir="/home/mturja/HCP_7task_preloaded")