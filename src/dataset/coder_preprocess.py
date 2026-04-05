"""
Preprocess huggingface dataset "Crystalcareai/Code-feedback-sharegpt-renamed"
so that multi-round conversation is split into multiple rows.
"""

from datasets import load_dataset, Dataset


def expand_conversations(example):
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


def main():
    dataset_name = "Crystalcareai/Code-feedback-sharegpt-renamed"
    save_path = "./assets/processed_code_feedback"

    print(f"Loading dataset: {dataset_name}...")
    raw_dataset = load_dataset(dataset_name, split="train")

    new_data_list = []

    print("Expanding multi-round conversations...")
    for example in raw_dataset:
        rounds = expand_conversations(example)
        new_data_list.extend(rounds)

    processed_dataset = Dataset.from_list(new_data_list)

    print(f"Original size: {len(raw_dataset)}")
    print(f"New expanded size: {len(processed_dataset)}")

    print(f"Saving dataset to: {save_path}...")
    processed_dataset.save_to_disk(save_path)


if __name__ == "__main__":
    main()
