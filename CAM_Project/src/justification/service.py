import json
from pathlib import Path
from typing import Any

from llm import call_llm

def generate_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Generate a justification report."""
    if not payload:
        raise ValueError("Payload must not be empty.")
    system_prompt = (Path(__file__).parent / "prompts.md").read_text()
    raw = call_llm(system_prompt, json.dumps(payload, indent=2, default=str))
    return json.loads(raw)
