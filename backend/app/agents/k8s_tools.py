import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml

CLUSTER_NAME = "releaseforge"
APP_LABEL = "app"
VERSION_LABEL = "version"


class KubeToolMissingError(Exception):
    """Raised when `kind` or `kubectl` isn't on PATH — checked once up front so every
    later step can assume the tool exists, instead of every subprocess call needing its
    own FileNotFoundError handling."""


def _which_or_raise(name: str) -> None:
    if shutil.which(name) is None:
        raise KubeToolMissingError(f"`{name}` isn't installed, or isn't on PATH.")


def _run(args: list[str], timeout: int, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        # Explicit utf-8, not text=True's platform-default decode: kind/kubectl print
        # unicode progress glyphs (checkmarks, spinners) that aren't valid cp1252 —
        # Windows' default console encoding — and text=True crashed the reader thread
        # outright the first time this ran for real against `kind create cluster`.
        encoding="utf-8",
        errors="replace",
        input=input_text,
        stdin=subprocess.DEVNULL if input_text is None else None,
        timeout=timeout,
    )


def _output(result: subprocess.CompletedProcess) -> str:
    return ((result.stdout or "") + (result.stderr or "")).strip()


def sanitize_dns_label(name: str) -> str:
    """Kubernetes names (namespaces, Deployments, Services) must be lowercase RFC 1123
    DNS labels: [a-z0-9-], no leading/trailing hyphen, max 63 chars."""
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return slug[:63].strip("-") or "app"


def detect_container_port(workspace_path: str, default: int = 8080) -> int:
    """Reads the first `EXPOSE <port>` line from the repo's Dockerfile — the same
    honest-signal principle as everywhere else in this project: read what the repo
    actually declares rather than guess a framework-typical port."""
    dockerfile = Path(workspace_path) / "Dockerfile"
    if not dockerfile.exists():
        return default
    match = re.search(r"^\s*EXPOSE\s+(\d+)", dockerfile.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
    return int(match.group(1)) if match else default


# ---------- cluster bootstrap ----------


def ensure_kind_cluster(cluster_name: str = CLUSTER_NAME, timeout: int = 180) -> tuple[bool, str]:
    """Idempotent: creates the shared local `kind` cluster only if it doesn't already
    exist. One cluster for every project this tool manages, not one per repo — separated
    by Kubernetes namespace instead, so it doesn't accumulate a new VM-equivalent per
    project."""
    _which_or_raise("kind")
    existing = _run(["kind", "get", "clusters"], timeout=30)
    if cluster_name in existing.stdout.split():
        return True, f"kind cluster '{cluster_name}' already exists"

    result = _run(["kind", "create", "cluster", "--name", cluster_name], timeout=timeout)
    return result.returncode == 0, _output(result)


def kind_load_image(image_tag: str, cluster_name: str = CLUSTER_NAME, timeout: int = 120) -> tuple[bool, str]:
    """`kind` clusters don't see the host's Docker images automatically — a locally built
    image has to be explicitly loaded into the cluster's node, since there's no registry
    in this local setup for it to be pulled from."""
    _which_or_raise("kind")
    result = _run(["kind", "load", "docker-image", image_tag, "--name", cluster_name], timeout=timeout)
    return result.returncode == 0, _output(result)


def ensure_namespace(namespace: str, timeout: int = 30) -> tuple[bool, str]:
    _which_or_raise("kubectl")
    check = _run(["kubectl", "get", "namespace", namespace], timeout=timeout)
    if check.returncode == 0:
        return True, f"namespace '{namespace}' already exists"
    result = _run(["kubectl", "create", "namespace", namespace], timeout=timeout)
    return result.returncode == 0, _output(result)


# ---------- manifest generation (deterministic — no LLM near cluster-changing YAML) ----------


def build_deployment_manifest(
    name: str,
    namespace: str,
    image: str,
    port: int,
    replicas: int,
    strategy_type: str,
    extra_labels: dict[str, str] | None = None,
    container_name: str | None = None,
) -> dict:
    """container_name defaults to `name` (the Deployment's own resource name) — correct
    for Recreate/Rolling, where the Deployment name already *is* the shared app name.
    It's wrong for Blue-Green/Canary, whose Deployment name is a color/canary-suffixed
    variant: a real live test caught this exact gap — promote_and_finish()'s Canary
    branch always calls set_deployment_image(container_name=app_name), which needs
    every strategy's container to actually be named that, not whatever this Deployment
    resource happens to be called. graph.py passes container_name=app_name explicitly
    for Blue-Green/Canary to fix that, rather than changing the assumption on the
    calling side."""
    labels = {APP_LABEL: name, **(extra_labels or {})}
    spec: dict = {
        "replicas": replicas,
        "selector": {"matchLabels": labels},
        "template": {
            "metadata": {"labels": labels},
            "spec": {
                "containers": [
                    {
                        "name": container_name or name,
                        "image": image,
                        "ports": [{"containerPort": port}],
                        "imagePullPolicy": "IfNotPresent",
                    }
                ]
            },
        },
    }
    if strategy_type in ("Recreate", "RollingUpdate"):
        spec["strategy"] = {"type": strategy_type}
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": spec,
    }


def build_service_manifest(name: str, namespace: str, port: int, selector_labels: dict[str, str]) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "selector": selector_labels,
            "ports": [{"port": port, "targetPort": port}],
            "type": "ClusterIP",
        },
    }


def manifests_to_yaml(manifests: list[dict]) -> str:
    return "---\n".join(yaml.safe_dump(m, sort_keys=False) for m in manifests)


# ---------- apply / rollout ----------


def apply_manifests(namespace: str, manifests: list[dict], timeout: int = 60) -> tuple[bool, str]:
    _which_or_raise("kubectl")
    result = _run(
        ["kubectl", "apply", "-n", namespace, "-f", "-"],
        timeout=timeout,
        input_text=manifests_to_yaml(manifests),
    )
    return result.returncode == 0, _output(result)


def rollout_status(namespace: str, deployment_name: str, timeout: int = 90) -> tuple[bool, str]:
    _which_or_raise("kubectl")
    result = _run(
        ["kubectl", "rollout", "status", f"deployment/{deployment_name}", "-n", namespace, f"--timeout={timeout}s"],
        timeout=timeout + 15,
    )
    return result.returncode == 0, _output(result)


def deployment_exists(namespace: str, deployment_name: str, timeout: int = 30) -> bool:
    _which_or_raise("kubectl")
    result = _run(["kubectl", "get", "deployment", deployment_name, "-n", namespace], timeout=timeout)
    return result.returncode == 0


# ---------- validation / monitoring ----------


def get_pod_restart_total(namespace: str, label_selector: str, timeout: int = 30) -> tuple[int, str]:
    """Sum of restart counts across every container in every pod matching the selector —
    the simplest real signal that something is crash-looping, not a simulated check."""
    _which_or_raise("kubectl")
    result = _run(
        ["kubectl", "get", "pods", "-n", namespace, "-l", label_selector, "-o", "json"],
        timeout=timeout,
    )
    if result.returncode != 0:
        return 0, _output(result)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0, result.stdout
    total = 0
    statuses = []
    for pod in data.get("items", []):
        pod_name = pod.get("metadata", {}).get("name", "?")
        for cs in pod.get("status", {}).get("containerStatuses", []):
            total += cs.get("restartCount", 0)
        phase = pod.get("status", {}).get("phase", "Unknown")
        statuses.append(f"{pod_name}: {phase}")
    return total, "; ".join(statuses)


def pod_logs(namespace: str, label_selector: str, tail: int = 100, timeout: int = 30) -> str:
    _which_or_raise("kubectl")
    result = _run(
        ["kubectl", "logs", "-n", namespace, "-l", label_selector, f"--tail={tail}", "--all-containers"],
        timeout=timeout,
    )
    return _output(result)


def describe_pods(namespace: str, label_selector: str, timeout: int = 30) -> str:
    _which_or_raise("kubectl")
    result = _run(["kubectl", "describe", "pods", "-n", namespace, "-l", label_selector], timeout=timeout)
    return _output(result)[-4000:]


# ---------- rollback / cleanup ----------


def rollout_undo(namespace: str, deployment_name: str, timeout: int = 60) -> tuple[bool, str]:
    _which_or_raise("kubectl")
    result = _run(["kubectl", "rollout", "undo", f"deployment/{deployment_name}", "-n", namespace], timeout=timeout)
    return result.returncode == 0, _output(result)


def delete_deployment(namespace: str, deployment_name: str, timeout: int = 60) -> tuple[bool, str]:
    _which_or_raise("kubectl")
    result = _run(
        ["kubectl", "delete", "deployment", deployment_name, "-n", namespace, "--ignore-not-found"], timeout=timeout
    )
    return result.returncode == 0, _output(result)


def get_service_selector(namespace: str, service_name: str, timeout: int = 30) -> dict[str, str] | None:
    _which_or_raise("kubectl")
    result = _run(["kubectl", "get", "service", service_name, "-n", namespace, "-o", "json"], timeout=timeout)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout).get("spec", {}).get("selector", {})
    except json.JSONDecodeError:
        return None


def patch_service_selector(namespace: str, service_name: str, selector: dict[str, str], timeout: int = 30) -> tuple[bool, str]:
    """Uses a JSON Patch "replace" (--type=json), not a JSON Merge Patch (--type=merge).
    A merge patch recursively merges nested objects — patching selector `{"app": "x"}`
    onto an existing `{"app": "x", "version": "green"}` would leave "version" sitting
    there untouched, since merge patch only overwrites keys it names, it doesn't clear
    the ones it doesn't. A JSON Patch replace swaps the whole field wholesale, which is
    what "make the selector exactly this" actually needs — found by live-testing a
    Canary deploy right after a Blue-Green promotion: the Service kept routing only to
    the promoted color's pods, silently, because the leftover `version` key survived
    an earlier merge-patch attempt at this same fix."""
    _which_or_raise("kubectl")
    patch = json.dumps([{"op": "replace", "path": "/spec/selector", "value": selector}])
    result = _run(
        ["kubectl", "patch", "service", service_name, "-n", namespace, "-p", patch, "--type=json"], timeout=timeout
    )
    return result.returncode == 0, _output(result)


def set_deployment_image(namespace: str, deployment_name: str, container_name: str, image: str, timeout: int = 30) -> tuple[bool, str]:
    _which_or_raise("kubectl")
    result = _run(
        ["kubectl", "set", "image", f"deployment/{deployment_name}", f"{container_name}={image}", "-n", namespace],
        timeout=timeout,
    )
    return result.returncode == 0, _output(result)


def check_cluster_reachable(timeout: int = 15) -> dict:
    """The real infra-readiness signal release_readiness() was missing: is a Kubernetes
    cluster actually reachable right now? Deliberately cluster-level, not namespace-
    specific — this runs before select_deployment_strategy() has even computed this
    app's namespace, so there's nothing more specific to check yet at this point in the
    pipeline. Raises KubeToolMissingError (not a False result) when kubectl itself isn't
    on PATH, same as every other function here — callers already know how to turn that
    into an honest "couldn't run this check" rather than a fabricated pass or fail."""
    _which_or_raise("kubectl")
    result = _run(["kubectl", "cluster-info"], timeout=timeout)
    reachable = result.returncode == 0
    detail = "A Kubernetes cluster is reachable." if reachable else _output(result)[:500]
    return {"reachable": reachable, "detail": detail}


# ---------- strategy selection (deterministic, pure — no subprocess, no LLM) ----------


def decide_deployment_strategy(has_existing_deployment: bool, risk: str, readiness_score: int) -> tuple[str, str]:
    """Same principle as release_readiness() in graph.py: picks from real signals this
    tool actually has (review risk, readiness score, whether a previous version of this
    app is even running yet), never an LLM guess — a wrong deployment-strategy guess has
    real consequences (traffic, downtime), which is exactly the kind of high-risk call
    the design doc's Section 13 says needs a deterministic, explainable basis, not a
    model's judgment call."""
    if not has_existing_deployment:
        return "Recreate", "First deployment for this app — no existing version to roll from or run a canary against."

    if risk == "High":
        return (
            "Blue-Green",
            f"Review risk is High (readiness {readiness_score}/100) — deploy a full parallel environment "
            "and only switch traffic after it's confirmed healthy, so a bad release never touches live traffic.",
        )
    if risk == "Medium":
        return (
            "Canary",
            f"Review risk is Medium (readiness {readiness_score}/100) — send a small slice of traffic to the "
            "new version first and promote only if it stays healthy.",
        )
    return (
        "Rolling",
        f"Review risk is Low (readiness {readiness_score}/100) — a standard rolling update is proportionate.",
    )
