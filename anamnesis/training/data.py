"""
Training data pipeline for Anamnesis.

Handles conversation data with signal annotations from Didymus.
Each training example is a conversation turn with:
- Input tokens (the prompt / context)
- Output tokens (Thomas's response)
- Signal health score (from Didymus scorer)
- Individual signal facets (alignment, embodiment, clarity, vitality)
- Field coherence at time of conversation

Data sources:
1. Didymus SQLite database (messages table with signal metadata)
2. Pre-exported JSON/JSONL files
3. HuggingFace datasets (for general instruction data)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset


@dataclass
class SignalAnnotatedExample:
    """A single training example with signal annotations."""
    input_text: str
    output_text: str
    signal_health: float = 0.5
    alignment: float = 0.5
    embodiment: float = 0.5
    clarity: float = 0.5
    vitality: float = 0.5
    field_coherence: float = 0.5
    person: str = ""
    timestamp: str = ""

    @property
    def signal_tensor(self) -> Tensor:
        """Signal facets as a tensor for the proxy network."""
        return torch.tensor([
            self.alignment, self.embodiment, self.clarity,
            self.vitality, self.field_coherence,
        ])


class ConversationDataset(Dataset):
    """
    Dataset of signal-annotated conversation turns.

    Loads from JSONL files where each line is:
    {
        "input": "...",
        "output": "...",
        "signal_health": 0.75,
        "alignment": 0.8,
        "embodiment": 0.7,
        "clarity": 0.6,
        "vitality": 0.9,
        "field_coherence": 0.5,
        "person": "aidan",
        "timestamp": "2026-03-24T..."
    }

    Args:
        path: Path to JSONL file or directory of JSONL files.
        tokenizer: HuggingFace tokenizer for encoding text.
        max_length: Maximum sequence length.
        min_signal: Minimum signal health to include (filter low-quality data).
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer=None,
        max_length: int = 2048,
        min_signal: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[SignalAnnotatedExample] = []

        path = Path(path)
        if path.is_dir():
            files = sorted(path.glob("*.jsonl"))
        else:
            files = [path]

        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    example = SignalAnnotatedExample(
                        input_text=data.get("input", ""),
                        output_text=data.get("output", ""),
                        signal_health=data.get("signal_health", 0.5),
                        alignment=data.get("alignment", 0.5),
                        embodiment=data.get("embodiment", 0.5),
                        clarity=data.get("clarity", 0.5),
                        vitality=data.get("vitality", 0.5),
                        field_coherence=data.get("field_coherence", 0.5),
                        person=data.get("person", ""),
                        timestamp=data.get("timestamp", ""),
                    )
                    if example.signal_health >= min_signal:
                        self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        if self.tokenizer is not None:
            # Format as chat: input → output
            text = f"<|im_start|>user\n{example.input_text}<|im_end|>\n<|im_start|>assistant\n{example.output_text}<|im_end|>"
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "signal_health": torch.tensor(example.signal_health),
                "signal_facets": example.signal_tensor,
            }
        else:
            return {
                "input_text": example.input_text,
                "output_text": example.output_text,
                "signal_health": torch.tensor(example.signal_health),
                "signal_facets": example.signal_tensor,
            }

    def get_preference_pairs(
        self,
        signal_threshold: float = 0.15,
    ) -> list[tuple[SignalAnnotatedExample, SignalAnnotatedExample]]:
        """
        Generate DPO preference pairs from the dataset.

        Finds pairs of examples with the same person/context but different
        signal scores. The higher-signal example is "chosen", the lower is "rejected".

        For simplicity, pairs any two examples where the signal difference
        exceeds the threshold (more sophisticated matching can be added).

        Args:
            signal_threshold: Minimum signal difference to form a pair.

        Returns:
            List of (chosen, rejected) example pairs.
        """
        pairs = []
        sorted_examples = sorted(self.examples, key=lambda e: -e.signal_health)

        for i, chosen in enumerate(sorted_examples):
            for rejected in sorted_examples[i + 1:]:
                if chosen.signal_health - rejected.signal_health >= signal_threshold:
                    pairs.append((chosen, rejected))
                    break  # One pair per chosen example

        return pairs

    def signal_statistics(self) -> dict[str, float]:
        """Compute signal health statistics across the dataset."""
        if not self.examples:
            return {}
        signals = [e.signal_health for e in self.examples]
        return {
            "count": len(signals),
            "mean": sum(signals) / len(signals),
            "min": min(signals),
            "max": max(signals),
            "std": (sum((s - sum(signals) / len(signals)) ** 2 for s in signals) / len(signals)) ** 0.5,
        }


def export_from_didymus(
    db_path: str | Path,
    output_path: str | Path,
    min_length: int = 50,
) -> int:
    """
    Export training data from Didymus SQLite database.

    Reads the messages table and joins with signal metadata to create
    signal-annotated conversation turns.

    Args:
        db_path: Path to Didymus SQLite database.
        output_path: Path to write JSONL output.
        min_length: Minimum message length to include.

    Returns:
        Number of examples exported.
    """
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Query messages with signal metadata
    cursor = conn.execute("""
        SELECT
            m.content,
            m.role,
            m.person,
            m.created_at,
            m.salience,
            m.metadata
        FROM messages m
        WHERE LENGTH(m.content) >= ?
        ORDER BY m.created_at ASC
    """, (min_length,))

    examples = []
    prev_user_msg = None

    for row in cursor:
        role = row["role"]
        content = row["content"]
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        if role == "user":
            prev_user_msg = content
        elif role == "assistant" and prev_user_msg:
            # Extract signal from metadata
            signal = metadata.get("signal", {})
            example = {
                "input": prev_user_msg,
                "output": content,
                "signal_health": signal.get("health", 0.5),
                "alignment": signal.get("alignment", 0.5),
                "embodiment": signal.get("embodiment", 0.5),
                "clarity": signal.get("clarity", 0.5),
                "vitality": signal.get("vitality", 0.5),
                "field_coherence": signal.get("field_coherence", 0.5),
                "person": row["person"] or "",
                "timestamp": row["created_at"] or "",
            }
            examples.append(example)
            prev_user_msg = None

    conn.close()

    # Write JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    return len(examples)
