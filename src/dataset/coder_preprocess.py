"""
Preprocess the Hugging Face dataset "Crystalcareai/Code-feedback-sharegpt-renamed"
so that each multi-round conversation is split into multiple rows, one per round.
"""

import os
import argparse
from datasets import load_dataset, Dataset
from tqdm import tqdm

DATASET_NAME = "Crystalcareai/Code-feedback-sharegpt-renamed"
DEFAULT_OUTPUT_PATH = "./assets/processed_code_feedback"


def expand_conversations(example):
    """
    Splits a multi-round conversation into one row per round,
    where each row contains all messages up to and including that round.
    """
    conversations = example["messages"]
    expanded_rows = []

    for i in range(1, len(conversations), 2):
        current_round_slice = conversations[: i + 1]

        new_row = {
            "messages": current_round_slice,
            "id": f"{example.get('id', '0')}_round_{i // 2 + 1}",
        }
        expanded_rows.append(new_row)

    return expanded_rows


def preprocess_dataset(local_hf_path):
    """
    Loads the Code-feedback-sharegpt-renamed dataset, expands multi-round
    conversations into individual rows, and saves the result to disk.
    """
    print(f"Loading dataset: {DATASET_NAME}...")
    try:
        raw_dataset = load_dataset(DATASET_NAME, split="train")
        print(f"Total examples to process: {len(raw_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    new_data_list = []

    print("Expanding multi-round conversations...")
    for example in tqdm(raw_dataset):
        rounds = expand_conversations(example)
        new_data_list.extend(rounds)

    processed_dataset = Dataset.from_list(new_data_list)

    print(f"Original size: {len(raw_dataset)}")
    print(f"New expanded size: {len(processed_dataset)}")

    # --- Saving Logic ---
    if local_hf_path:
        print(f"Saving Hugging Face dataset to disk at {local_hf_path}...")
        if not os.path.exists(local_hf_path):
            os.makedirs(local_hf_path)
        processed_dataset.save_to_disk(local_hf_path)

    print("Successfully processed and saved the dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expand multi-round conversations in the Code-feedback dataset into individual rows."
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Local directory path to save the Hugging Face dataset format. (default: {DEFAULT_OUTPUT_PATH})",
    )

    args = parser.parse_args()
    preprocess_dataset(args.hf_path)
