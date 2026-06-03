"""
Preprocess the Hugging Face dataset "lmsys/lmsys-chat-1m" into ShareGPT format.

Steps:
  1. Filter rows whose 'language' column is not 'English'.
  2. Convert the 'conversation' field (list of {"role", "content"} dicts, where
     role is "user" or "assistant") to the ShareGPT 'messages' format
     (list of {"role", "value"} dicts, where role is "human" or "gpt").
  3. Expand each multi-round conversation into one row per round, where each row
     contains all messages up to and including that round.
"""

import os
import argparse
from datasets import load_dataset, Dataset
from tqdm import tqdm

DATASET_NAME = "lmsys/lmsys-chat-1m"
DEFAULT_OUTPUT_PATH = "./assets/processed_lmsys_chat"


# Map lmsys role names to ShareGPT role names
_ROLE_MAP = {
    "user": "human",
    "assistant": "gpt",
}


def to_sharegpt_messages(conversation):
    """
    Convert a lmsys conversation (list of {"role", "content"} dicts with
    roles "user" / "assistant") to ShareGPT messages format
    (list of {"role", "value"} dicts with roles "human" / "gpt").
    """
    return [
        {"role": _ROLE_MAP[turn["role"]], "value": turn["content"]}
        for turn in conversation
    ]


def expand_conversations(example):
    """
    Splits a multi-round conversation into one row per round,
    where each row contains all messages up to and including that round.
    """
    messages = example["messages"]
    expanded_rows = []

    for i in range(1, len(messages), 2):
        current_round_slice = messages[: i + 1]

        new_row = {
            "messages": current_round_slice,
            "id": f"{example['conversation_id']}_round_{i // 2 + 1}",
        }
        expanded_rows.append(new_row)

    return expanded_rows


def preprocess_dataset(local_hf_path):
    """
    Loads lmsys/lmsys-chat-1m, filters to English rows, converts to ShareGPT
    format, expands multi-round conversations into individual rows, and saves
    the result to disk.
    """
    print(f"Loading dataset: {DATASET_NAME}...")
    try:
        raw_dataset = load_dataset(DATASET_NAME, split="train")
        print(f"Total rows before filtering: {len(raw_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Step 1 — filter to English only
    english_dataset = raw_dataset.filter(lambda row: row["language"] == "English")
    print(f"Rows after English filter: {len(english_dataset)}")

    new_data_list = []

    # Steps 2 & 3 — convert format and expand rounds
    print("Converting to ShareGPT format and expanding multi-round conversations...")
    for example in tqdm(english_dataset):
        example["messages"] = to_sharegpt_messages(example["conversation"])
        rounds = expand_conversations(example)
        new_data_list.extend(rounds)

    processed_dataset = Dataset.from_list(new_data_list)

    print(f"Original (English) size: {len(english_dataset)}")
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
        description="Filter, convert, and expand lmsys-chat-1m into ShareGPT format."
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Local directory path to save the Hugging Face dataset format. (default: {DEFAULT_OUTPUT_PATH})",
    )

    args = parser.parse_args()
    preprocess_dataset(args.hf_path)
