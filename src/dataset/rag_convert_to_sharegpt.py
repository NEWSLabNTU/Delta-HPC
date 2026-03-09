import os
import argparse
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm


def convert_dataset(local_hf_path):
    """
    Loads both train and test splits of neural-bridge/rag-dataset-12000,
    concatenates them, and exports to ShareGPT format as a Hugging Face dataset.
    """
    print("Loading 'train' and 'test' splits from Hugging Face...")
    try:
        # Load both splits
        dataset_dict = load_dataset("neural-bridge/rag-dataset-12000")

        # Combine the splits into one single dataset object
        combined_dataset = concatenate_datasets(
            [dataset_dict["train"], dataset_dict["test"]]
        )
        print(f"Total rows to process: {len(combined_dataset)}")
    except Exception as e:
        print(f"Error loading dataset splits: {e}")
        return

    sharegpt_list = []

    print("Processing rows into ShareGPT format...")
    for mid, row in enumerate(tqdm(combined_dataset)):
        # 1. Concatenate context and question into a complete query prompt
        context = row.get("context", "")
        question = row.get("question", "")

        # Structure the prompt for RAG evaluation
        prompt = f"Context:\n{context}\n\nQuestion: {question}"

        # 2. Construct ShareGPT format
        # Each entry has an ID and a list of messages with 'role' and 'value'
        conversation = {
            "id": mid,
            "messages": [
                {"role": "human", "value": prompt},
                {"role": "gpt", "value": row.get("answer", "")},
            ],
        }
        sharegpt_list.append(conversation)

    # Convert the list of dictionaries to a Hugging Face Dataset Object
    hf_dataset = Dataset.from_list(sharegpt_list)

    # --- Saving Logic ---
    if local_hf_path:
        print(f"Saving Hugging Face dataset to disk at {local_hf_path}...")
        if not os.path.exists(local_hf_path):
            os.makedirs(local_hf_path)
        hf_dataset.save_to_disk(local_hf_path)

    print("Successfully processed and saved both splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RAG dataset (train+test) to ShareGPT format."
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        required=True,
        help="Local directory path to save the Hugging Face dataset format.",
    )

    args = parser.parse_args()
    convert_dataset(args.hf_path)
