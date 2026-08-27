"""The Review Agent golden set: 42 diffs with a deliberately injected, real defect (6 each
across 7 CWE-shaped categories) plus 16 clean diffs as negatives, used by
run_reviewer_golden_set.py to measure whether review() (backend/app/agents/graph.py)
actually catches what it's supposed to.

Each defect fixture is built from a small per-category template, substituted across 6
different domains (user/order/payment/inventory/session/invoice) so the 6 variants in a
category are genuinely different code, not the same snippet copy-pasted — the same
"parameterized fixture family" technique benchmarks like OWASP Benchmark use for CWE
coverage. Every fixture still gets reviewed by literally calling the real review() node
against real before/after content — nothing here is graded by string-matching the code
itself, only by what the LLM's actual review response says about it.

catch_keywords: if ANY of these substrings appears (case-insensitive) in the reviewer's
combined critical+warning text, the defect counts as caught. These are deliberately broad
(e.g. "sql injection", "inject", "parameteriz", "sanitiz" for SQLi) so a reviewer that
names the problem in different words still gets credit — the goal is measuring whether the
*defect* was caught, not whether the reviewer used one specific phrase.
"""

from dataclasses import dataclass, field

DOMAINS = ["user", "order", "payment", "inventory", "session", "invoice"]


@dataclass
class ReviewFixture:
    id: str
    defect_type: str | None  # None for a clean (negative) fixture
    severity: str | None  # "critical" | "major" | "minor" | None for clean
    target_file: str
    change_description: str
    original_content: str
    new_content: str
    catch_keywords: list[str] = field(default_factory=list)
    test_passed: bool | None = True


# ---------------------------------------------------------------------------
# 1. SQL injection (critical) — string-built query instead of a parameterized one
# ---------------------------------------------------------------------------
def _sql_injection(domain: str) -> ReviewFixture:
    table = f"{domain}s"
    return ReviewFixture(
        id=f"sqli_{domain}",
        defect_type="sql_injection",
        severity="critical",
        target_file=f"app/repositories/{domain}_repository.py",
        change_description=f"Add a lookup function that fetches a {domain} row by its id, filterable by name.",
        original_content=f"def get_{domain}(cursor):\n    return None\n",
        new_content=(
            f"def get_{domain}_by_name(cursor, name):\n"
            f'    query = "SELECT * FROM {table} WHERE name = \'" + name + "\'"\n'
            f"    cursor.execute(query)\n"
            f"    return cursor.fetchone()\n"
        ),
        catch_keywords=["sql injection", "sql-injection", "injection", "parameteriz", "sanitiz", "unsafe query", "string concatenation"],
    )


# ---------------------------------------------------------------------------
# 2. Hardcoded secret (critical) — a real-looking credential committed as a literal
# ---------------------------------------------------------------------------
def _hardcoded_secret(domain: str) -> ReviewFixture:
    return ReviewFixture(
        id=f"secret_{domain}",
        defect_type="hardcoded_secret",
        severity="critical",
        target_file=f"app/services/{domain}_client.py",
        change_description=f"Wire up the {domain} service's API client so it can authenticate outbound requests.",
        original_content=f"class {domain.capitalize()}Client:\n    def __init__(self):\n        self.api_key = None\n",
        new_content=(
            f"class {domain.capitalize()}Client:\n"
            f'    def __init__(self):\n'
            f'        self.api_key = "sk_live_51Hc9f3KzT9r2mNq8vXpL4jY7wRbA2eD6"\n'
            f"        self.base_url = \"https://api.{domain}s.internal\"\n"
        ),
        catch_keywords=["hardcoded", "hard-coded", "secret", "credential", "api key", "env var", "environment variable", "exposed key"],
    )


# ---------------------------------------------------------------------------
# 3. Missing null check (major) — attribute/index access on a value that can be None
# ---------------------------------------------------------------------------
def _missing_null_check(domain: str) -> ReviewFixture:
    return ReviewFixture(
        id=f"nullcheck_{domain}",
        defect_type="missing_null_check",
        severity="major",
        target_file=f"app/services/{domain}_service.py",
        change_description=f"Add a function that returns the owner's email for a given {domain}.",
        original_content=f"def find_{domain}(store, {domain}_id):\n    return store.get({domain}_id)\n",
        new_content=(
            f"def get_owner_email(store, {domain}_id):\n"
            f"    record = store.get({domain}_id)\n"
            f"    return record.owner.email\n"
        ),
        catch_keywords=["null", "none check", "nonetype", "attributeerror", "missing check", "not checked", "could be none", "may be none"],
    )


# ---------------------------------------------------------------------------
# 4. Off-by-one (major) — a boundary error in a loop/range/slice
# ---------------------------------------------------------------------------
def _off_by_one(domain: str) -> ReviewFixture:
    return ReviewFixture(
        id=f"offbyone_{domain}",
        defect_type="off_by_one",
        severity="major",
        target_file=f"app/utils/{domain}_pagination.py",
        change_description=f"Add a helper that returns the last N items from a list of {domain} records.",
        original_content=f"def first_n(items, n):\n    return items[:n]\n",
        new_content=(
            f"def last_n(items, n):\n"
            f"    result = []\n"
            f"    for i in range(1, n):\n"
            f"        result.append(items[len(items) - i])\n"
            f"    return result\n"
        ),
        catch_keywords=[
            "off-by-one", "off by one", "boundary", "index error", "skips the last", "misses the last",
            "range(1", "wrong index", "does not actually return", "does not correctly", "incorrect result",
            "logic is incorrect", "logic error", "wrong result",
        ],
    )


# ---------------------------------------------------------------------------
# 5. Unhandled exception (minor) — a call that can raise, with no try/except
# ---------------------------------------------------------------------------
def _unhandled_exception(domain: str) -> ReviewFixture:
    return ReviewFixture(
        id=f"unhandled_{domain}",
        defect_type="unhandled_exception",
        severity="minor",
        target_file=f"app/importers/{domain}_import.py",
        change_description=f"Add a function that loads a {domain} record from a JSON string sent by the client.",
        original_content=f"def parse_{domain}(raw):\n    return {{}}\n",
        new_content=(
            f"import json\n\n"
            f"def parse_{domain}(raw):\n"
            f"    data = json.loads(raw)\n"
            f"    return {{\"id\": data[\"id\"], \"name\": data[\"name\"]}}\n"
        ),
        catch_keywords=["unhandled", "exception", "try/except", "try-except", "crash", "malformed", "invalid json", "jsondecodeerror", "keyerror"],
    )


# ---------------------------------------------------------------------------
# 6. N+1 query (major) — one DB query per loop iteration instead of a batch query
# ---------------------------------------------------------------------------
def _n_plus_one(domain: str) -> ReviewFixture:
    # The query itself must be otherwise safe (parameterized) — the ONLY defect this
    # fixture should contain is the N+1 loop shape. An earlier version of this template
    # built the query with an f-string, which is ALSO a SQL-injection shape; that
    # confounded 3 of 6 live runs, where the reviewer (correctly) reported the injection
    # instead of the N+1 pattern it was meant to isolate — a dataset bug, not a model miss.
    return ReviewFixture(
        id=f"n_plus_one_{domain}",
        defect_type="n_plus_one_query",
        severity="major",
        target_file=f"app/services/{domain}_list_service.py",
        change_description=f"Add a function that returns a list of {domain} records with each one's owner name attached.",
        original_content=f"def list_{domain}s(db, ids):\n    return [db.get({domain}_id) for {domain}_id in ids]\n",
        new_content=(
            f"def list_{domain}s_with_owner(db, ids):\n"
            f"    results = []\n"
            f"    for {domain}_id in ids:\n"
            f"        record = db.query(\"SELECT * FROM {domain}s WHERE id = %s\", ({domain}_id,))\n"
            f"        owner = db.query(\"SELECT name FROM users WHERE id = %s\", (record.owner_id,))\n"
            f"        results.append((record, owner))\n"
            f"    return results\n"
        ),
        catch_keywords=["n+1", "n + 1", "query per", "one query", "multiple queries", "loop", "batch", "join instead", "inefficient"],
    )


# ---------------------------------------------------------------------------
# 7. Race condition (critical) — check-then-act on shared state with no lock
# ---------------------------------------------------------------------------
def _race_condition(domain: str) -> ReviewFixture:
    return ReviewFixture(
        id=f"race_{domain}",
        defect_type="race_condition",
        severity="critical",
        target_file=f"app/services/{domain}_reservation.py",
        change_description=f"Add a function that reserves the next available {domain} slot from a shared counter, callable concurrently by multiple requests.",
        original_content=f"_available_{domain}_slots = 10\n\ndef slots_left():\n    return _available_{domain}_slots\n",
        new_content=(
            f"_available_{domain}_slots = 10\n\n"
            f"async def reserve_{domain}_slot():\n"
            f"    global _available_{domain}_slots\n"
            f"    if _available_{domain}_slots > 0:\n"
            f"        await notify_reserved()\n"
            f"        _available_{domain}_slots -= 1\n"
            f"        return True\n"
            f"    return False\n"
        ),
        catch_keywords=["race condition", "race", "concurren", "lock", "atomic", "thread-safe", "thread safe", "check-then-act", "shared state"],
    )


_DEFECT_BUILDERS = [
    _sql_injection,
    _hardcoded_secret,
    _missing_null_check,
    _off_by_one,
    _unhandled_exception,
    _n_plus_one,
    _race_condition,
]


def _build_defect_fixtures() -> list[ReviewFixture]:
    fixtures = []
    for builder in _DEFECT_BUILDERS:
        for domain in DOMAINS:
            fixtures.append(builder(domain))
    return fixtures


# ---------------------------------------------------------------------------
# Clean fixtures (negatives) — 16 genuinely distinct, correct changes. No shared template:
# a false-positive rate is only meaningful if these are as varied as real PRs, not 16
# copies of the same safe pattern.
# ---------------------------------------------------------------------------
def _build_clean_fixtures() -> list[ReviewFixture]:
    return [
        ReviewFixture(
            id="clean_add_field_user",
            defect_type=None, severity=None,
            target_file="app/models/user.py",
            change_description="Add an optional 'display_name' field to the User model.",
            original_content="class User:\n    def __init__(self, email):\n        self.email = email\n",
            new_content="class User:\n    def __init__(self, email, display_name=None):\n        self.email = email\n        self.display_name = display_name\n",
        ),
        ReviewFixture(
            id="clean_parameterized_query",
            defect_type=None, severity=None,
            target_file="app/repositories/order_repository.py",
            change_description="Add a lookup function that fetches an order by id, using a parameterized query.",
            original_content="def get_order(cursor):\n    return None\n",
            new_content="def get_order_by_id(cursor, order_id):\n    cursor.execute(\"SELECT * FROM orders WHERE id = %s\", (order_id,))\n    return cursor.fetchone()\n",
        ),
        ReviewFixture(
            id="clean_env_var_secret",
            defect_type=None, severity=None,
            target_file="app/services/payment_client.py",
            change_description="Wire up the payment service's API client, reading its key from the environment.",
            original_content="class PaymentClient:\n    def __init__(self):\n        self.api_key = None\n",
            new_content="import os\n\nclass PaymentClient:\n    def __init__(self):\n        self.api_key = os.environ[\"PAYMENT_API_KEY\"]\n",
        ),
        ReviewFixture(
            id="clean_null_checked",
            defect_type=None, severity=None,
            target_file="app/services/invoice_service.py",
            change_description="Add a function that returns the owner's email for a given invoice, or None if not found.",
            original_content="def find_invoice(store, invoice_id):\n    return store.get(invoice_id)\n",
            new_content="def get_owner_email(store, invoice_id):\n    record = store.get(invoice_id)\n    if record is None or record.owner is None:\n        return None\n    return record.owner.email\n",
        ),
        ReviewFixture(
            id="clean_correct_slice",
            defect_type=None, severity=None,
            target_file="app/utils/session_pagination.py",
            change_description="Add a helper that returns the last N items from a list of session records.",
            original_content="def first_n(items, n):\n    return items[:n]\n",
            new_content="def last_n(items, n):\n    if n <= 0:\n        return []\n    return items[-n:]\n",
        ),
        ReviewFixture(
            id="clean_try_except_import",
            defect_type=None, severity=None,
            target_file="app/importers/inventory_import.py",
            change_description="Add a function that loads an inventory record from a JSON string, handling malformed input.",
            original_content="def parse_inventory(raw):\n    return {}\n",
            new_content="import json\n\ndef parse_inventory(raw):\n    try:\n        data = json.loads(raw)\n    except json.JSONDecodeError:\n        return None\n    return {\"id\": data.get(\"id\"), \"name\": data.get(\"name\")}\n",
        ),
        ReviewFixture(
            id="clean_batched_query",
            defect_type=None, severity=None,
            target_file="app/services/user_list_service.py",
            change_description="Add a function that returns a list of user records with each one's owner name attached, using a single join.",
            original_content="def list_users(db, ids):\n    return [db.get(user_id) for user_id in ids]\n",
            new_content="def list_users_with_owner(db, ids):\n    return db.query(\n        \"SELECT u.*, o.name FROM users u JOIN owners o ON u.owner_id = o.id WHERE u.id IN %s\",\n        (tuple(ids),),\n    )\n",
        ),
        ReviewFixture(
            id="clean_lock_protected_counter",
            defect_type=None, severity=None,
            target_file="app/services/order_reservation.py",
            change_description="Add a function that reserves the next available order slot from a shared counter, safe for concurrent calls.",
            original_content="_available_order_slots = 10\n\ndef slots_left():\n    return _available_order_slots\n",
            new_content="import asyncio\n\n_available_order_slots = 10\n_lock = asyncio.Lock()\n\nasync def reserve_order_slot():\n    async with _lock:\n        global _available_order_slots\n        if _available_order_slots > 0:\n            _available_order_slots -= 1\n            return True\n        return False\n",
        ),
        ReviewFixture(
            id="clean_add_docstring",
            defect_type=None, severity=None,
            target_file="app/utils/formatting.py",
            change_description="Document the existing currency formatting helper.",
            original_content="def format_currency(amount):\n    return f\"${amount:.2f}\"\n",
            new_content="def format_currency(amount):\n    \"\"\"Formats a numeric amount as a USD string, e.g. 12.5 -> '$12.50'.\"\"\"\n    return f\"${amount:.2f}\"\n",
        ),
        ReviewFixture(
            id="clean_rename_variable",
            defect_type=None, severity=None,
            target_file="app/services/inventory_service.py",
            change_description="Rename a confusingly-named local variable for clarity, no behavior change.",
            original_content="def total_stock(items):\n    x = 0\n    for i in items:\n        x += i.quantity\n    return x\n",
            new_content="def total_stock(items):\n    total = 0\n    for item in items:\n        total += item.quantity\n    return total\n",
        ),
        ReviewFixture(
            id="clean_add_type_hints",
            defect_type=None, severity=None,
            target_file="app/services/session_service.py",
            change_description="Add type hints to the existing session-expiry check function.",
            original_content="def is_expired(session, now):\n    return session.expires_at < now\n",
            new_content="from datetime import datetime\n\ndef is_expired(session: \"Session\", now: datetime) -> bool:\n    return session.expires_at < now\n",
        ),
        ReviewFixture(
            id="clean_extract_constant",
            defect_type=None, severity=None,
            target_file="app/services/payment_limits.py",
            change_description="Extract a magic number into a named constant for readability.",
            original_content="def is_over_limit(amount):\n    return amount > 10000\n",
            new_content="MAX_TRANSACTION_AMOUNT = 10000\n\ndef is_over_limit(amount):\n    return amount > MAX_TRANSACTION_AMOUNT\n",
        ),
        ReviewFixture(
            id="clean_add_logging",
            defect_type=None, severity=None,
            target_file="app/services/order_service.py",
            change_description="Add a log line when an order is cancelled, for observability.",
            original_content="def cancel_order(order):\n    order.status = \"cancelled\"\n    return order\n",
            new_content="import logging\n\nlogger = logging.getLogger(__name__)\n\ndef cancel_order(order):\n    order.status = \"cancelled\"\n    logger.info(\"order %s cancelled\", order.id)\n    return order\n",
        ),
        ReviewFixture(
            id="clean_input_validation",
            defect_type=None, severity=None,
            target_file="app/api/user_routes.py",
            change_description="Add basic input validation to the user-creation endpoint.",
            original_content="def create_user(payload):\n    return User(payload[\"email\"])\n",
            new_content="def create_user(payload):\n    email = payload.get(\"email\", \"\").strip()\n    if not email or \"@\" not in email:\n        raise ValueError(\"a valid email is required\")\n    return User(email)\n",
        ),
        ReviewFixture(
            id="clean_add_index_field",
            defect_type=None, severity=None,
            target_file="app/models/invoice.py",
            change_description="Add a 'created_at' timestamp field to the Invoice model, defaulted at construction time.",
            original_content="class Invoice:\n    def __init__(self, amount):\n        self.amount = amount\n",
            new_content="from datetime import datetime, timezone\n\nclass Invoice:\n    def __init__(self, amount):\n        self.amount = amount\n        self.created_at = datetime.now(timezone.utc)\n",
        ),
        ReviewFixture(
            id="clean_test_only_change",
            defect_type=None, severity=None,
            target_file="app/utils/retry.py",
            change_description="Reduce the default retry count from 5 to 3 for a flaky external call.",
            original_content="def with_retries(fn, retries=5):\n    for attempt in range(retries):\n        try:\n            return fn()\n        except Exception:\n            continue\n    raise RuntimeError(\"exhausted retries\")\n",
            new_content="def with_retries(fn, retries=3):\n    for attempt in range(retries):\n        try:\n            return fn()\n        except Exception:\n            continue\n    raise RuntimeError(\"exhausted retries\")\n",
        ),
    ]


def load_golden_set() -> list[ReviewFixture]:
    return _build_defect_fixtures() + _build_clean_fixtures()
