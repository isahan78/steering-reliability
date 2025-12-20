"""
Data loading utilities for JSONL prompt datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


def load_prompts(
    jsonl_path: str,
    max_prompts: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load prompts from a JSONL file.

    Expected JSONL format:
    {"id": "harm_train_0001", "prompt": "...", "meta": {"family": "...", "style": "..."}}

    Args:
        jsonl_path: Path to JSONL file
        max_prompts: Maximum number of prompts to load (None = all)

    Returns:
        List of prompt dictionaries

    Example:
        >>> prompts = load_prompts("data/prompts/harm_train.jsonl")
        >>> print(prompts[0])
        {'id': 'harm_train_0001', 'prompt': '...', 'meta': {...}}
    """
    jsonl_path = Path(jsonl_path)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {jsonl_path}")

    prompts = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if max_prompts is not None and i >= max_prompts:
                break

            line = line.strip()
            if not line:
                continue

            try:
                prompt_dict = json.loads(line)
                prompts.append(prompt_dict)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i+1} in {jsonl_path}: {e}")

    print(f"Loaded {len(prompts)} prompts from {jsonl_path}")
    return prompts


def save_prompts(prompts: List[Dict[str, Any]], jsonl_path: str):
    """
    Save prompts to a JSONL file.

    Args:
        prompts: List of prompt dictionaries
        jsonl_path: Path to save JSONL file
    """
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")

    print(f"Saved {len(prompts)} prompts to {jsonl_path}")


def prompts_to_dataframe(prompts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of prompt dictionaries to a pandas DataFrame.

    Args:
        prompts: List of prompt dictionaries

    Returns:
        DataFrame with columns: id, prompt, and flattened metadata
    """
    rows = []
    for p in prompts:
        row = {
            "id": p.get("id"),
            "prompt": p.get("prompt"),
        }
        # Flatten metadata
        if "meta" in p:
            for key, value in p["meta"].items():
                row[f"meta_{key}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


def load_all_splits(
    harm_train_path: str,
    harm_test_path: str,
    benign_path: str,
    max_prompts_per_split: Optional[int] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all three prompt splits.

    Args:
        harm_train_path: Path to harm_train.jsonl
        harm_test_path: Path to harm_test.jsonl
        benign_path: Path to benign.jsonl
        max_prompts_per_split: Max prompts per split (None = all)

    Returns:
        Dictionary with keys: harm_train, harm_test, benign
    """
    return {
        "harm_train": load_prompts(harm_train_path, max_prompts_per_split),
        "harm_test": load_prompts(harm_test_path, max_prompts_per_split),
        "benign": load_prompts(benign_path, max_prompts_per_split),
    }


def validate_prompt(prompt_dict: Dict[str, Any]) -> bool:
    """
    Validate that a prompt dictionary has the required fields.

    Args:
        prompt_dict: Prompt dictionary to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    if "id" not in prompt_dict:
        raise ValueError("Prompt missing 'id' field")
    if "prompt" not in prompt_dict:
        raise ValueError("Prompt missing 'prompt' field")
    if not isinstance(prompt_dict["prompt"], str):
        raise ValueError("Prompt 'prompt' field must be a string")

    return True
