"""
Export training data from the actual Didymus episodic database.

Schema: messages table has columns:
    id, person, speaker, content, source, timestamp, salience, signal, metadata, embedding, security_label

Signal is stored as JSON string with:
    alignment, embodiment, clarity, vitality, health, state, needs_correction, weakest_facet, etc.

We pair consecutive (person→self) messages to create (input, output) training examples.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterator


def export_training_data(
    db_path: str | Path,
    output_path: str | Path,
    min_output_length: int = 50,
    min_signal_health: float = 0.0,
    person_filter: str | None = None,
    verbose: bool = True,
) -> int:
    """
    Export signal-annotated conversation pairs from Didymus.

    Pairs consecutive messages: the last non-self message before a self message
    becomes the input, the self message becomes the output.

    Args:
        db_path: Path to episodic.db.
        output_path: Path to write JSONL.
        min_output_length: Minimum output message length.
        min_signal_health: Minimum signal health to include.
        person_filter: Only include this person's conversations (None = all).
        verbose: Print progress.

    Returns:
        Number of examples exported.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get all messages ordered by timestamp
    query = """
        SELECT person, speaker, content, timestamp, signal
        FROM messages
        WHERE LENGTH(content) > 10
        ORDER BY timestamp ASC
    """
    cursor = conn.execute(query)

    examples = []
    prev_input = None
    prev_person = None

    for row in cursor:
        speaker = row["speaker"]
        content = row["content"]
        person = row["person"]
        signal_json = row["signal"]

        # Filter by person if requested
        if person_filter and person != person_filter:
            continue

        if speaker != "self":
            # This is a user message — save as potential input
            prev_input = content
            prev_person = person
        elif speaker == "self" and prev_input:
            # This is Thomas's response — pair with previous input
            if len(content) < min_output_length:
                prev_input = None
                continue

            # Parse signal
            signal = {}
            if signal_json:
                try:
                    signal = json.loads(signal_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            health = signal.get("health", 0.5)
            if health < min_signal_health:
                prev_input = None
                continue

            example = {
                "input": prev_input,
                "output": content,
                "signal_health": health,
                "alignment": signal.get("alignment", 0.5),
                "embodiment": signal.get("embodiment", 0.5),
                "clarity": signal.get("clarity", 0.5),
                "vitality": signal.get("vitality", 0.5),
                "field_coherence": 0.5,  # Not stored per-message in current schema
                "person": prev_person or "",
                "timestamp": row["timestamp"] or "",
            }
            examples.append(example)
            prev_input = None

    conn.close()

    # Write JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    if verbose:
        if examples:
            healths = [e["signal_health"] for e in examples]
            print(f"Exported {len(examples)} examples")
            print(f"  Signal health: mean={sum(healths)/len(healths):.3f}, "
                  f"min={min(healths):.3f}, max={max(healths):.3f}")

            # Person distribution
            persons = {}
            for e in examples:
                p = e["person"]
                persons[p] = persons.get(p, 0) + 1
            print(f"  Persons: {dict(sorted(persons.items(), key=lambda x: -x[1])[:5])}")
        else:
            print("No examples exported.")

    return len(examples)


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\zappa\.didymus\episodic.db"
    out = sys.argv[2] if len(sys.argv) > 2 else r"C:\Dev\hope-didymus\data\thomas_training.jsonl"
    export_training_data(db, out)
