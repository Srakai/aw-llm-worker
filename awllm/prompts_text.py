"""Prompt templates for text-based activity classification."""

from typing import List, Tuple


def build_text_classification_prompt(
    text_content: str, topics: List[str]
) -> Tuple[str, str]:
    """Build system and user prompts for text classification.

    Args:
        text_content: The activity log text to classify
        topics: List of topic categories

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    topics_str = ", ".join(topics) if topics else "general"

    system_prompt = f"""You are an expert time management assistant. Your task is to classify the user's activity based on a list of events.
The user will provide a block of text summarizing their activity over a period of time, which may include:
- Window/app activity logs
- VS Code project and file information (summary)
- Screenshot analysis results (detailed summaries with timestamps)

You must classify this activity into one of the following categories: {topics_str}.

Additionally, identify:
1. The most likely project they are working on (if any)
2. A brief description of what they're actually doing

Use all available context (window events, VS Code activity, and screenshot summaries) to make the best classification.

Respond with a JSON object containing:
- "label": one of the categories ({topics_str})
- "confidence": 0.0-1.0
- "project": string or null (the project name if identifiable)
- "activity_description": string (brief description of what they're doing, max 50 words)

Example: {{"label": "Coding", "confidence": 0.9, "project": "aw-llm-worker", "activity_description": "Refactoring prompt templates and updating data models"}}"""

    user_prompt = f"""Here is the activity log:
---
{text_content[:3000]}
---
Classify this activity block and identify the project and what the user is doing."""

    return system_prompt, user_prompt


def build_cli_text_prompt(text_content: str, topics: List[str]) -> str:
    """Build a combined prompt for CLI text classification.

    Args:
        text_content: The activity log text to classify
        topics: List of topic categories

    Returns:
        Combined prompt string for CLI
    """
    topics_str = ", ".join(topics) if topics else "general"

    return f"""You are an expert time management assistant. Classify this activity into one of: {topics_str}.
Also identify the project being worked on and what the user is doing.

The activity log includes window events, VS Code activity summaries, and screenshot analysis results.
Use all available context to make the best classification.

Activity log:
---
{text_content[:3000]}
---

Respond with JSON containing:
- "label": category ({topics_str})
- "confidence": 0.0-1.0
- "project": project name or null
- "activity_description": brief description (max 50 words)

Example: {{"label": "Coding", "confidence": 0.9, "project": "aw-llm-worker", "activity_description": "Refactoring prompt templates"}}"""
