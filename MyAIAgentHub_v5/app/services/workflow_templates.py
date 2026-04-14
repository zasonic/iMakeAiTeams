"""
services/workflow_templates.py

Five pre-built workflow templates that bypass the coordinator and add tasks
directly, providing instant reliable workflow scaffolding for common goals.

Each template is a dict with:
  id:          short identifier used by the frontend
  name:        display name
  description: one-line summary
  icon:        emoji for the gallery card
  tasks:       list of {name, agent_role, depends_on (names), description}

plan_from_template() resolves depends_on names → UUIDs (same as plan_workflow)
and writes the workflow + tasks to the DB, then returns workflow_id.
"""

import json
import logging
import uuid as _uuid
from datetime import datetime, timezone

import db

log = logging.getLogger("MyAIEnv.templates")


# ── Template definitions ──────────────────────────────────────────────────────

WORKFLOW_TEMPLATES = [
    {
        "id": "research_report",
        "name": "Research Report",
        "description": "Search, synthesise, and write a structured report on any topic.",
        "icon": "🔬",
        "tasks": [
            {
                "name": "gather_sources",
                "agent_role": "Researcher",
                "depends_on": [],
                "description": (
                    "Research the topic provided in the workflow goal. "
                    "Identify at least 5 key facts, statistics, or findings. "
                    "Return a JSON object: {\"topic\": str, \"findings\": [str], \"key_questions\": [str]}"
                ),
            },
            {
                "name": "deep_analysis",
                "agent_role": "Researcher",
                "depends_on": ["gather_sources"],
                "description": (
                    "Using the initial findings, perform a deeper analysis. "
                    "Identify patterns, contradictions, and gaps. "
                    "Return JSON: {\"analysis\": str, \"insights\": [str], \"gaps\": [str]}"
                ),
            },
            {
                "name": "write_report",
                "agent_role": "Writer",
                "depends_on": ["gather_sources", "deep_analysis"],
                "description": (
                    "Write a well-structured report (Introduction, Findings, Analysis, "
                    "Conclusion) using the research and analysis above. "
                    "Include specific numbers and sources. "
                    "Return JSON: {\"title\": str, \"report\": str}"
                ),
            },
            {
                "name": "review_report",
                "agent_role": "Reviewer",
                "depends_on": ["write_report"],
                "description": (
                    "Review the report for accuracy, completeness, and quality. "
                    "Return JSON: {\"verdict\": \"pass\"|\"revise\", \"issues\": [str], \"summary\": str}"
                ),
            },
        ],
    },
    {
        "id": "code_review",
        "name": "Code Review",
        "description": "Analyse code for bugs, security issues, and improvement opportunities.",
        "icon": "🔍",
        "tasks": [
            {
                "name": "static_analysis",
                "agent_role": "Code Helper",
                "depends_on": [],
                "description": (
                    "Perform static analysis on the code or codebase described in the goal. "
                    "Look for: syntax errors, undefined variables, obvious bugs, bad practices. "
                    "Return JSON: {\"language\": str, \"issues\": [{\"line\": str, \"severity\": str, \"description\": str}]}"
                ),
            },
            {
                "name": "security_scan",
                "agent_role": "Code Helper",
                "depends_on": [],
                "description": (
                    "Review the code for security vulnerabilities: SQL injection, XSS, "
                    "hardcoded secrets, insecure dependencies, improper error handling. "
                    "Return JSON: {\"vulnerabilities\": [{\"type\": str, \"severity\": \"high\"|\"medium\"|\"low\", \"description\": str}]}"
                ),
            },
            {
                "name": "performance_review",
                "agent_role": "Code Helper",
                "depends_on": [],
                "description": (
                    "Review the code for performance issues: N+1 queries, unnecessary loops, "
                    "missing indexes, large memory allocations, blocking I/O. "
                    "Return JSON: {\"performance_issues\": [{\"location\": str, \"impact\": str, \"suggestion\": str}]}"
                ),
            },
            {
                "name": "summary_report",
                "agent_role": "Writer",
                "depends_on": ["static_analysis", "security_scan", "performance_review"],
                "description": (
                    "Synthesise the findings from all three reviews into a clean code review report. "
                    "Group by severity. Include actionable recommendations. "
                    "Return JSON: {\"summary\": str, \"critical\": [str], \"recommended\": [str], \"minor\": [str]}"
                ),
            },
        ],
    },
    {
        "id": "content_outline",
        "name": "Content Outline",
        "description": "Generate a detailed content outline with headings, key points, and SEO notes.",
        "icon": "📝",
        "tasks": [
            {
                "name": "audience_research",
                "agent_role": "Researcher",
                "depends_on": [],
                "description": (
                    "Research the target audience and key questions they have about this topic. "
                    "Identify search intent and common pain points. "
                    "Return JSON: {\"audience\": str, \"pain_points\": [str], \"search_intent\": str}"
                ),
            },
            {
                "name": "competitor_analysis",
                "agent_role": "Researcher",
                "depends_on": [],
                "description": (
                    "Analyse what content already exists on this topic. "
                    "What angles are overrepresented? What gaps exist? "
                    "Return JSON: {\"common_angles\": [str], \"gaps\": [str], \"differentiators\": [str]}"
                ),
            },
            {
                "name": "create_outline",
                "agent_role": "Writer",
                "depends_on": ["audience_research", "competitor_analysis"],
                "description": (
                    "Create a detailed content outline using the audience research and gap analysis. "
                    "Include: title, meta description, H2/H3 headings, key points per section, "
                    "word count target, and calls to action. "
                    "Return JSON: {\"title\": str, \"meta_description\": str, \"sections\": [{\"heading\": str, \"key_points\": [str]}], \"word_count_target\": int}"
                ),
            },
        ],
    },
    {
        "id": "document_qa",
        "name": "Document Q&A",
        "description": "Extract key information, answer questions, and summarise indexed documents.",
        "icon": "📄",
        "tasks": [
            {
                "name": "extract_structure",
                "agent_role": "Researcher",
                "depends_on": [],
                "description": (
                    "Search the indexed documents for content relevant to the stated goal. "
                    "Identify main topics, key entities (names, dates, numbers), and document structure. "
                    "Return JSON: {\"main_topics\": [str], \"entities\": {\"names\": [str], \"dates\": [str], \"numbers\": [str]}, \"doc_count\": int}"
                ),
            },
            {
                "name": "answer_questions",
                "agent_role": "Researcher",
                "depends_on": ["extract_structure"],
                "description": (
                    "Based on the document structure, answer the specific questions in the goal. "
                    "Cite the source for each answer. Flag any questions the documents don't answer. "
                    "Return JSON: {\"answers\": [{\"question\": str, \"answer\": str, \"source\": str}], \"unanswered\": [str]}"
                ),
            },
            {
                "name": "executive_summary",
                "agent_role": "Summarizer",
                "depends_on": ["extract_structure", "answer_questions"],
                "description": (
                    "Write a concise executive summary of the documents and their relevance to the goal. "
                    "Include key findings, answered questions, and any important caveats. "
                    "Return JSON: {\"summary\": str, \"key_findings\": [str], \"caveats\": [str]}"
                ),
            },
        ],
    },
    {
        "id": "competitive_analysis",
        "name": "Competitive Analysis",
        "description": "Compare competitors, identify strengths/weaknesses, and find strategic opportunities.",
        "icon": "⚔️",
        "tasks": [
            {
                "name": "identify_competitors",
                "agent_role": "Researcher",
                "depends_on": [],
                "description": (
                    "Given the product or market described in the goal, identify 3–5 key competitors. "
                    "For each, note their positioning, target market, and main offering. "
                    "Return JSON: {\"competitors\": [{\"name\": str, \"positioning\": str, \"target\": str, \"offering\": str}]}"
                ),
            },
            {
                "name": "feature_comparison",
                "agent_role": "Researcher",
                "depends_on": ["identify_competitors"],
                "description": (
                    "Compare the competitors' features, pricing, strengths, and weaknesses. "
                    "Return JSON: {\"comparison\": [{\"competitor\": str, \"strengths\": [str], \"weaknesses\": [str], \"pricing\": str}]}"
                ),
            },
            {
                "name": "swot_analysis",
                "agent_role": "Researcher",
                "depends_on": ["identify_competitors", "feature_comparison"],
                "description": (
                    "Produce a SWOT analysis for the product/market described in the goal "
                    "relative to the identified competitors. "
                    "Return JSON: {\"strengths\": [str], \"weaknesses\": [str], \"opportunities\": [str], \"threats\": [str]}"
                ),
            },
            {
                "name": "strategic_recommendations",
                "agent_role": "Writer",
                "depends_on": ["feature_comparison", "swot_analysis"],
                "description": (
                    "Write a strategic recommendations report using all prior analysis. "
                    "Include: positioning recommendations, feature priorities, and market entry tactics. "
                    "Return JSON: {\"recommendations\": [str], \"quick_wins\": [str], \"long_term\": [str], \"risks\": [str]}"
                ),
            },
        ],
    },
]

# Index by id for fast lookup
_TEMPLATE_INDEX = {t["id"]: t for t in WORKFLOW_TEMPLATES}


def list_templates() -> list[dict]:
    """Return all templates as lightweight gallery cards (no task details)."""
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "description": t["description"],
            "icon": t["icon"],
            "task_count": len(t["tasks"]),
        }
        for t in WORKFLOW_TEMPLATES
    ]


def get_template(template_id: str) -> dict | None:
    return _TEMPLATE_INDEX.get(template_id)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def plan_from_template(
    template_id: str,
    goal: str,
    workflow_name: str = "",
) -> str:
    """
    Create a workflow from a pre-built template.

    The `goal` string is prepended to each task's description so agents
    know the user's specific target.  No coordinator call is needed —
    the task graph is fully predefined.

    Returns workflow_id.
    Raises ValueError if template_id is unknown.
    """
    template = _TEMPLATE_INDEX.get(template_id)
    if not template:
        raise ValueError(f"Unknown template id: {template_id!r}")

    name = workflow_name or f"{template['name']} — {goal[:40]}"

    wf_id = str(_uuid.uuid4())
    db.execute(
        "INSERT INTO workflows (id, name, status, created_at, updated_at) "
        "VALUES (?, ?, 'pending', ?, ?)",
        (wf_id, name, _now(), _now()),
    )

    # Pre-assign UUIDs so depends_on names can be resolved in one pass
    name_to_id: dict[str, str] = {t["name"]: str(_uuid.uuid4()) for t in template["tasks"]}

    for task_def in template["tasks"]:
        task_id = name_to_id[task_def["name"]]
        dep_ids = [name_to_id[dep] for dep in task_def.get("depends_on", [])]
        # Prepend the goal to the task description so agents have full context
        enriched_description = f"Goal: {goal}\n\n{task_def['description']}"
        db.execute(
            """
            INSERT INTO tasks
                (id, workflow_id, name, agent_role, status, depends_on,
                 input_data, output_data, attempt_count, max_attempts, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'pending', ?, ?, '{}', 0, 3, ?, ?)
            """,
            (
                task_id, wf_id,
                task_def["name"],
                task_def.get("agent_role", "General Assistant"),
                json.dumps(dep_ids),
                json.dumps({"description": enriched_description}),
                _now(), _now(),
            ),
        )

    db.commit()
    log.info("Created workflow %s from template '%s' (goal: %s…)",
             wf_id[:8], template_id, goal[:40])
    return wf_id
