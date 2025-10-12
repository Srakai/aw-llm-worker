"""Prompt templates for text-based activity classification."""

from typing import List, Tuple, Dict, Any


def build_text_classification_prompt(
    text_content: str, topics: List[str], projects: List[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """Build system and user prompts for text classification.

    Args:
        text_content: The activity log text to classify
        topics: List of topic categories
        projects: List of project definitions with keywords from context.yaml

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    topics_str = ", ".join(topics) if topics else "general"

    # Build project context section
    project_context = ""
    if projects:
        project_names = [p.get("name", "") for p in projects if p.get("name")]
        if project_names:
            project_context = f"\n\nKnown projects: {', '.join(project_names)}"

    system_prompt = f"""You are an expert time management assistant. Your task is to classify the user's activity based on a list of events.
The user will provide a block of text summarizing their activity over a period of time, which may include:
- Window/app activity logs
- VS Code project and file information (summary)
- Screenshot analysis results (detailed summaries with timestamps)

You must classify this activity into one of the following categories: {topics_str}.

Additionally, identify:
1. The most likely project they are working on (if any){project_context}
2. A brief description of what they're actually doing
3. Whether you need screenshot analysis to better understand the activity (if screenshot summaries are missing or unclear)

Use all available context (window events, VS Code activity, and screenshot summaries) to make the best classification.

Respond with a JSON object containing:
- "label": one of the categories ({topics_str})
- "confidence": 0.0-1.0
- "project": string or null (the project name if identifiable)
- "activity_description": string (brief description of what they're doing, max 50 words)
- "request_screenshot_analysis": boolean (true if you need screenshots to better understand this activity)
- "screenshot_analysis_reason": string (if requesting screenshots, explain why in max 30 words)

Example: {{"label": "Coding", "confidence": 0.9, "project": "aw-llm-worker", "activity_description": "Refactoring prompt templates and updating data models", "request_screenshot_analysis": false, "screenshot_analysis_reason": null}}"""

    user_prompt = f"""Here is the activity log:
---
{text_content[:3000]}
---
Classify this activity block and identify the project and what the user is doing."""

    return system_prompt, user_prompt


def build_cli_text_prompt(
    text_content: str, topics: List[str], projects: List[Dict[str, Any]] = None
) -> str:
    """Build a combined prompt for CLI text classification.

    Args:
        text_content: The activity log text to classify
        topics: List of topic categories
        projects: List of project definitions with keywords from context.yaml

    Returns:
        Combined prompt string for CLI
    """
    topics_str = ", ".join(topics) if topics else "general"

    # Build project context section
    project_context = ""
    if projects:
        project_names = [p.get("name", "") for p in projects if p.get("name")]
        if project_names:
            project_context = f"\n\nKnown projects: {', '.join(project_names)}"

    return f"""You are an expert time management assistant. Classify this activity into one of: {topics_str}.
Also identify the project being worked on and what the user is doing.{project_context}
If you need screenshot analysis to better understand the activity, indicate this in the response.

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
- "request_screenshot_analysis": boolean (true if you need screenshots)
- "screenshot_analysis_reason": string or null (why you need screenshots, max 30 words)

Example: {{"label": "Coding", "confidence": 0.9, "project": "aw-llm-worker", "activity_description": "Refactoring prompt templates", "request_screenshot_analysis": false, "screenshot_analysis_reason": null}}"""


def build_screenshot_refinement_prompt(
    initial_classification: Dict[str, Any],
    window_content: str,
    screenshot_descriptions: List[str],
    topics: List[str],
    projects: List[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Build prompts for refining a classification using screenshot analysis.

    Args:
        initial_classification: The initial LLM classification result
        window_content: The original window text content
        screenshot_descriptions: List of screenshot analysis results
        topics: List of topic categories
        projects: List of project definitions

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    topics_str = ", ".join(topics) if topics else "general"

    project_context = ""
    if projects:
        project_names = [p.get("name", "") for p in projects if p.get("name")]
        if project_names:
            project_context = f"\n\nKnown projects: {', '.join(project_names)}"

    system_prompt = f"""You are an expert time management assistant. You previously classified an activity block, and now you have additional screenshot evidence to refine your analysis.

Your task is to review the initial classification and enrich the activity description using the screenshot analysis.

Valid categories: {topics_str}{project_context}

Respond with a refined JSON object containing:
- "label": category (same or updated if screenshots reveal different activity)
- "confidence": 0.0-1.0 (should be higher with screenshot evidence)
- "project": project name or null (refined based on screenshots)
- "activity_description": enriched description incorporating screenshot details (max 80 words)
- "screenshot_insights": list of key insights from screenshots (3-5 bullet points)

Focus on making the description specific and actionable based on what you see in the screenshots."""

    screenshots_text = "\n".join(f"- {desc}" for desc in screenshot_descriptions)

    user_prompt = f"""Initial Classification:
- Label: {initial_classification.get('label')}
- Confidence: {initial_classification.get('confidence')}
- Project: {initial_classification.get('project')}
- Description: {initial_classification.get('activity_description')}

Window Activity Log (summary):
{window_content[:1000]}

Screenshot Analysis Results:
{screenshots_text}

Please refine the classification with a richer, more specific description based on the screenshot evidence."""

    return system_prompt, user_prompt
