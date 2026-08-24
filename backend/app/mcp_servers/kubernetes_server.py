"""MCP server exposing local Kubernetes (`kind`/`kubectl`) operations — the
"Kubernetes" server from the design doc's Section 5 MCP architecture.

Deliberately does NOT expose the pure, no-subprocess helpers from k8s_tools.py
(build_deployment_manifest, decide_deployment_strategy, sanitize_dns_label, ...) as MCP
tools. Those aren't actions on an external system — they're plain application logic
that happens to live next to the real tools, the same reason a filesystem server
wouldn't expose "capitalize this string" as a tool. Only the things that actually touch
the cluster are here.
"""

from mcp.server.mcpserver import MCPServer

from app.agents import k8s_tools

server = MCPServer(
    name="releaseforge-kubernetes",
    instructions="Deploys to, and inspects, a local `kind` Kubernetes cluster.",
)


@server.tool()
def check_cluster_reachable() -> dict:
    """Real infra-readiness signal — closes the third of the three gaps
    release_readiness() used to name as "not built yet." Cluster-level, not
    namespace-specific: this runs before select_deployment_strategy() has even computed
    the target namespace."""
    return k8s_tools.check_cluster_reachable()


@server.tool()
def ensure_kind_cluster() -> dict:
    """Creates the shared local `kind` cluster if it doesn't already exist. Idempotent."""
    ok, message = k8s_tools.ensure_kind_cluster()
    return {"ok": ok, "message": message}


@server.tool()
def kind_load_image(image_tag: str) -> dict:
    """Loads a locally built Docker image into the kind cluster's node."""
    ok, message = k8s_tools.kind_load_image(image_tag)
    return {"ok": ok, "message": message}


@server.tool()
def ensure_namespace(namespace: str) -> dict:
    ok, message = k8s_tools.ensure_namespace(namespace)
    return {"ok": ok, "message": message}


@server.tool()
def apply_manifests(namespace: str, manifests: list[dict]) -> dict:
    """Applies a list of Kubernetes manifest objects (Deployment, Service, ...) via
    `kubectl apply`."""
    ok, message = k8s_tools.apply_manifests(namespace, manifests)
    return {"ok": ok, "message": message}


@server.tool()
def rollout_status(namespace: str, deployment_name: str) -> dict:
    """Waits (bounded) for a Deployment's rollout to finish — the real readiness check."""
    ok, message = k8s_tools.rollout_status(namespace, deployment_name)
    return {"ok": ok, "message": message}


@server.tool()
def deployment_exists(namespace: str, deployment_name: str) -> bool:
    return k8s_tools.deployment_exists(namespace, deployment_name)


@server.tool()
def get_pod_restart_total(namespace: str, label_selector: str) -> dict:
    """Sum of container restarts across every pod matching the selector, plus a short
    per-pod status summary — the real signal used to catch a crash-looping deploy."""
    total, statuses = k8s_tools.get_pod_restart_total(namespace, label_selector)
    return {"total_restarts": total, "pod_statuses": statuses}


@server.tool()
def pod_logs(namespace: str, label_selector: str, tail: int = 100) -> str:
    return k8s_tools.pod_logs(namespace, label_selector, tail=tail)


@server.tool()
def describe_pods(namespace: str, label_selector: str) -> str:
    return k8s_tools.describe_pods(namespace, label_selector)


@server.tool()
def rollout_undo(namespace: str, deployment_name: str) -> dict:
    ok, message = k8s_tools.rollout_undo(namespace, deployment_name)
    return {"ok": ok, "message": message}


@server.tool()
def delete_deployment(namespace: str, deployment_name: str) -> dict:
    ok, message = k8s_tools.delete_deployment(namespace, deployment_name)
    return {"ok": ok, "message": message}


@server.tool()
def get_service_selector(namespace: str, service_name: str) -> dict:
    """Returns the Service's current label selector, or {"selector": null} if the
    Service doesn't exist yet (a first-ever deploy for this app)."""
    selector = k8s_tools.get_service_selector(namespace, service_name)
    return {"selector": selector}


@server.tool()
def patch_service_selector(namespace: str, service_name: str, selector: dict[str, str]) -> dict:
    """Replaces a Service's selector wholesale (JSON Patch replace, not a merge patch —
    see k8s_tools.py's docstring for exactly why that distinction matters here)."""
    ok, message = k8s_tools.patch_service_selector(namespace, service_name, selector)
    return {"ok": ok, "message": message}


@server.tool()
def set_deployment_image(namespace: str, deployment_name: str, container_name: str, image: str) -> dict:
    ok, message = k8s_tools.set_deployment_image(namespace, deployment_name, container_name, image)
    return {"ok": ok, "message": message}


if __name__ == "__main__":
    server.run()
