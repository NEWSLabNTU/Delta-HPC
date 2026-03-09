"""
Script to generate response from a dataset with a Coder Agent LLM.

Usage:
$ python src/profiling/coder-generator.py \
--port [port] \
--model [model name] \
--dataset-dir [HF dataset directory] \
--output-dir [output directory]
"""

import argparse
import json
import httpx
import asyncio
from pathlib import Path
from datasets import load_from_disk
from typing import Dict, Optional, List, Any

# Set the number of concurrent requests here
CONCURRENCY_LIMIT = 30


async def send_request(
    client: httpx.AsyncClient, url: str, model_name: str, messages: List[str], mid: str
) -> Optional[Dict[str, Any]]:
    """
    Sends a completion request to the vLLM server with exponential backoff.
    Returns a dictionary containing the content and token usage, or None if all retries fail.
    """
    print(f"[Start] MID: {mid}")
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 4096,
    }

    # Implementation of exponential backoff
    for i in range(6):  # 0 to 5 retries
        try:
            response = await client.post(url, json=payload, timeout=300.0)
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"]

            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            return {
                "id": mid,
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        except Exception as e:
            print("[WARN] Exception occurs in send_request:", e)
            if i == 5:
                print(f"[Error] MID: {mid} failed after retries: {e}")
                return {
                    "id": mid,
                    "content": f"Error after 5 retries: {str(e)}",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            else:
                wait_time = 2**i
                await asyncio.sleep(wait_time)
    return None


async def bounded_request(sem, client, *args):
    """Wrapper to enforce the concurrency limit using a semaphore."""
    async with sem:
        return await send_request(client, *args)


async def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(
        description="Serve prompts from Ling-Coder-SFT to a vLLM server concurrently."
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Port where the vLLM server is hosted."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name as registered in vLLM."
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the directory that stores the output .jsonl file.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    output_path = (
        output_dir
        / f"{args.model.lower().split("/")[1]}-port-{args.port}-{dataset_dir.name}-generated.jsonl"
    )

    vllm_url = f"http://localhost:{args.port}/v1/chat/completions"
    print(f"Connecting to vLLM at: {vllm_url}")
    dataset = load_from_disk(dataset_dir)
    total_rows = len(dataset)
    print(f"Loading dataset with {total_rows} samples...")

    # 2. Semaphore to limit concurrency
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # 3. Process Rows concurrently
    async with httpx.AsyncClient() as client:
        with open(output_path, "w", encoding="utf-8") as f:
            tasks = []
            for row in dataset:
                mid = row.get("id", "unknown")
                messages = row.get("messages", [])

                if len(messages) >= 2:
                    openai_messages = []
                    for x in messages[:-1]:
                        match x["role"]:
                            case "human":
                                role = "user"
                            case "gpt":
                                role = "assistant"
                            case _:
                                role = "unknown"
                        openai_messages.append({"role": role, "content": x["value"]})
                    task = bounded_request(
                        sem, client, vllm_url, args.model, openai_messages, mid
                    )
                    tasks.append(task)

            # Use a counter to track actual completion order
            completed_count = 0
            for future in asyncio.as_completed(tasks):
                result_data = await future
                completed_count += 1

                if result_data is not None:
                    mid = result_data["id"]
                    tokens = result_data["completion_tokens"]
                    print(
                        f"[{completed_count}/{total_rows}] [Done] MID: {mid} | Tokens: {tokens}"
                    )

                    result_entry = {
                        "id": mid,
                        "response": result_data["content"],
                        "prompt_tokens": result_data["prompt_tokens"],
                        "completion_tokens": tokens,
                        "total_tokens": result_data["total_tokens"],
                    }
                    f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                    f.flush()

    print(
        f"\nProcessing complete. {total_rows} tasks handled. Results saved to {output_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())
