from typing import TypedDict


class AgentState(TypedDict, total=False):
    workspace_path: str
    user_message: str

    intent: str  # "chat" | "task"
    reply: str  # what gets shown to the user in the chat panel

    plan: str
    target_file: str
    change_description: str

    original_content: str | None
    new_content: str

    test_passed: bool | None
    test_output: str

    review_status: str  # "APPROVED" | "CHANGES_REQUESTED"
    review_feedback: str
    review_risk: str
    review_round: int

    branch_name: str
    base_branch: str
    pr_url: str
    pr_error: str

    approval_response: str
