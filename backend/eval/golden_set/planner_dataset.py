"""The Planner golden set: task descriptions run against the REAL plan() node
(backend/app/agents/graph.py), each fixture asserting required "step coverage" — plan()
only ever outputs a single {plan, target_file, change_description} JSON object rather than
a literal multi-step list, so "step coverage" here means: did the plan correctly name the
right target file, and does its change_description mention what it must (required_keyword_groups)
while avoiding scope creep or dangerous language it must not (forbidden_keywords).

Three real bug classes from this project's own history motivate the categories below:
  - target-file guessing (_resolve_target_file was built specifically because an early plan()
    landed a change in the wrong file when a bare filename matched more than one candidate)
  - extension guessing (a live bug: a Vue project's pages are .vue files, not .html — the
    planner prompt was updated after this was caught in real use)
  - scope discipline (nothing in this project ever measured whether the planner stays
    narrowly scoped to the request instead of drifting into unrelated/destructive territory)

required_keyword_groups is a list of synonym groups: each group is satisfied if ANY word in
it appears, and the fixture only passes if EVERY group is satisfied — the same "credit
different valid phrasing" fix the reviewer golden set needed for off_by_one/n_plus_one,
found the same way: reading the actual generated text of a "failure" instead of trusting a
single literal string match. A first live run flagged 3 fixtures as failed purely because
the model said "malformed" instead of "valid", or "no linked profile" instead of "null" —
correct plans, too-rigid grading.
"""

from dataclasses import dataclass, field


@dataclass
class PlannerFixture:
    id: str
    category: str  # "target_selection" | "extension_convention" | "scope_discipline" | "keyword_coverage"
    task_description: str
    workspace_files: list[str]
    required_keyword_groups: list[list[str]] = field(default_factory=list)  # each group: ANY word satisfies it; ALL groups required
    forbidden_keywords: list[str] = field(default_factory=list)  # must NOT appear
    expected_target_file: str | None = None  # if set, target_file must match exactly


FIXTURES: list[PlannerFixture] = [
    # ---- target_selection: exactly one existing file should clearly be reused, not reinvented ----
    PlannerFixture(
        id="target_login_page",
        category="target_selection",
        task_description="Add a 'remember me' checkbox to the login page.",
        workspace_files=["src/views/login.html", "src/views/signup.html", "src/styles/main.css"],
        expected_target_file="src/views/login.html",
    ),
    PlannerFixture(
        id="target_user_model",
        category="target_selection",
        task_description="Add a 'last_login_at' field to the user model.",
        workspace_files=["app/models/user.py", "app/models/order.py", "app/models/__init__.py"],
        expected_target_file="app/models/user.py",
    ),
    PlannerFixture(
        id="target_checkout_route",
        category="target_selection",
        task_description="Fix the bug where the checkout route doesn't apply discount codes.",
        workspace_files=["app/routes/checkout.py", "app/routes/cart.py", "app/routes/account.py"],
        expected_target_file="app/routes/checkout.py",
    ),
    PlannerFixture(
        id="target_navbar_component",
        category="target_selection",
        task_description="Add a settings link to the navigation bar.",
        workspace_files=["src/components/Navbar.tsx", "src/components/Footer.tsx", "src/components/Sidebar.tsx"],
        expected_target_file="src/components/Navbar.tsx",
    ),
    PlannerFixture(
        id="target_payment_service",
        category="target_selection",
        task_description="The payment service should retry once on a timeout before failing.",
        workspace_files=["app/services/payment_service.py", "app/services/email_service.py", "app/services/sms_service.py"],
        expected_target_file="app/services/payment_service.py",
    ),
    # ---- extension_convention: repo's real file extension must be respected, not guessed ----
    PlannerFixture(
        id="extension_vue_project",
        category="extension_convention",
        task_description="Add a loading spinner to the dashboard page.",
        workspace_files=["src/views/Dashboard.vue", "src/views/Profile.vue", "src/router/index.js"],
        required_keyword_groups=[[".vue"]],
        forbidden_keywords=[".html"],
    ),
    PlannerFixture(
        id="extension_svelte_project",
        category="extension_convention",
        task_description="Add a dark mode toggle to the settings page.",
        workspace_files=["src/routes/Settings.svelte", "src/routes/Home.svelte", "src/app.css"],
        required_keyword_groups=[[".svelte"]],
        forbidden_keywords=[".html", ".jsx"],
    ),
    PlannerFixture(
        id="extension_typescript_project",
        category="extension_convention",
        task_description="Add input validation to the signup form.",
        workspace_files=["src/pages/Signup.tsx", "src/pages/Login.tsx", "src/utils/validators.ts"],
        required_keyword_groups=[[".tsx"]],
        forbidden_keywords=[".js", ".jsx"],
    ),
    # ---- scope_discipline: the plan must stay narrow, not drift into destructive/unrelated territory ----
    PlannerFixture(
        id="scope_single_bug_fix",
        category="scope_discipline",
        task_description="Fix the off-by-one error in the pagination helper.",
        workspace_files=["app/utils/pagination.py", "app/utils/formatting.py"],
        forbidden_keywords=["rewrite", "delete", "drop table", "remove all", "entire file", "from scratch"],
    ),
    PlannerFixture(
        id="scope_add_single_field",
        category="scope_discipline",
        task_description="Add a 'phone_number' field to the user profile form.",
        workspace_files=["src/views/Profile.html", "src/views/Settings.html"],
        forbidden_keywords=["delete", "drop table", "remove all fields", "redesign"],
    ),
    PlannerFixture(
        id="scope_small_ui_tweak",
        category="scope_discipline",
        task_description="Change the submit button color to match the brand palette.",
        workspace_files=["src/components/SubmitButton.tsx", "src/theme/colors.ts"],
        forbidden_keywords=["delete", "remove the component", "rewrite the entire", "drop table"],
    ),
    PlannerFixture(
        id="scope_add_log_line",
        category="scope_discipline",
        task_description="Log a warning when an order is retried more than twice.",
        workspace_files=["app/services/order_service.py", "app/services/inventory_service.py"],
        forbidden_keywords=["delete", "drop table", "rewrite", "remove all"],
    ),
    PlannerFixture(
        id="scope_rename_variable",
        category="scope_discipline",
        task_description="Rename the confusingly-named 'x' variable in the stock total calculation to something clearer.",
        workspace_files=["app/services/inventory_service.py", "app/services/order_service.py"],
        forbidden_keywords=["delete", "drop table", "rewrite the file", "remove all", "redesign"],
    ),
    # ---- keyword_coverage: the change_description must actually name the right concern ----
    PlannerFixture(
        id="keyword_input_validation",
        category="keyword_coverage",
        task_description="Add input validation so the signup endpoint rejects an empty or malformed email.",
        workspace_files=["app/routes/signup.py", "app/routes/login.py"],
        required_keyword_groups=[["email"], ["valid", "invalid", "malformed"]],
    ),
    PlannerFixture(
        id="keyword_null_safety",
        category="keyword_coverage",
        task_description="Fix the crash when a user has no linked profile — the account page should handle that gracefully.",
        workspace_files=["app/routes/account.py", "app/routes/settings.py"],
        required_keyword_groups=[["none", "null", "missing", "empty", "no linked", "no profile", "gracefully", "graceful"]],
    ),
    PlannerFixture(
        id="keyword_error_message",
        category="keyword_coverage",
        task_description="Show a clear error message when the upload fails instead of the page just going blank.",
        workspace_files=["src/components/Uploader.tsx", "src/components/Gallery.tsx"],
        required_keyword_groups=[["error"]],
    ),
    PlannerFixture(
        id="keyword_retry_logic",
        category="keyword_coverage",
        task_description="Make the email service retry once if the SMTP send fails.",
        workspace_files=["app/services/email_service.py", "app/services/sms_service.py"],
        required_keyword_groups=[["retry"]],
    ),
    PlannerFixture(
        id="keyword_pagination_limit",
        category="keyword_coverage",
        task_description="Cap the orders list endpoint at 50 results per page so it can't return the entire table at once.",
        workspace_files=["app/routes/orders.py", "app/routes/products.py"],
        required_keyword_groups=[["50"], ["limit", "cap", "page"]],
    ),
]
