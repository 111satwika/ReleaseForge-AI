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

    readiness_score: int
    readiness_recommendation: str  # "GO" | "NO-GO"
    readiness_breakdown: list[str]

    docker_image_tag: str

    deployment_strategy: str  # "Recreate" | "Rolling" | "Canary" | "Blue-Green"
    deployment_strategy_reason: str
    k8s_namespace: str
    k8s_app_name: str
    k8s_service_name: str
    k8s_deploy_target: str  # the actual Deployment name just rolled out (app / app-canary / app-green / app-blue)
    deploy_color: str  # only meaningful for Blue-Green: "blue" | "green" (the NEW color just deployed)
    deploy_applied: bool | None
    deploy_log: str
    deploy_started_at: str  # ISO timestamp stamped at the top of deploy() — the piece a real MTTR metric needed

    validation_passed: bool | None
    monitor_summary: str

    recovery_analysis: str
    rollback_performed: bool | None

    approval_response: str

    # How this task ended, e.g. "declined_at_plan", "docker_built" — set by whichever
    # terminal node the graph reaches, and persisted to release memory in main.py.
    outcome: str
