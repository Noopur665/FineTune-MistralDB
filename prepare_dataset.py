import pandas as pd

import os
from typing import Union

import datasets
from datasets import load_dataset


def save_and_compress(dataset: Union[datasets.Dataset, pd.DataFrame], name: str, idx=None):
    if idx:
        path = f"{name}_{idx}.jsonl"
    else:
        path = f"{name}.jsonl"

    print("Saving to", path)
    dataset.to_json(path, force_ascii=False, orient='records', lines=True)

    print("Compressing...")
    os.system(f'xz -zkf -T0 {path}')  # -TO to use multithreading


def get_dataset_column_from_text_folder(folder_path):
    return load_dataset("text", data_dir=folder_path, sample_by="document", split='train').to_pandas()['text']


for split in ["train", "test"]:
    dfs = []
    for dataset_name in ["IN-Abs", "UK-Abs", "IN-Ext"]:
        if dataset_name == "IN-Ext" and split == "test":
            continue
        print(f"Processing {dataset_name} {split}")
        path = f"original_dataset/{dataset_name}/{split}-data"

        df = pd.DataFrame()
        df['text'] = get_dataset_column_from_text_folder(f"{path}/judgement")
        # df['dataset_name'] = dataset_name

        if dataset_name == "UK-Abs" and split == "test" or dataset_name == "IN-Ext":
            summary_full_path = f"{path}/summary/full"
        else:
            summary_full_path = f"{path}/summary"
        df['response'] = get_dataset_column_from_text_folder(summary_full_path)
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.fillna("")  # NaNs can lead to huggingface not recognizing the feature type of the column
    save_and_compress(df, f"{split}")