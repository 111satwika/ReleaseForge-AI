import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

SERVICE_NAME = "releaseforge-backend"

_configured = False


def setup_tracing() -> None:
    """Configures a real OTLP exporter pointed at a trace backend (Jaeger's all-in-one
    image, run locally via Docker, ingests OTLP natively on :4318 — see the Phase 8
    build-plan notes for the exact `docker run` command). Overridable via
    RELEASEFORGE_OTEL_ENDPOINT for a different collector, or disabled entirely via
    RELEASEFORGE_TRACING_ENABLED=false.

    Deliberately does not fail startup if the collector is unreachable: the OTLP
    exporter already degrades to logging a warning and dropping the batch on export
    failure, the same "optional enhancement, not a dependency" posture as RAG's
    embedding calls in Phase 6 — a backend with no trace collector running should still
    serve chat/tasks normally, just without traces landing anywhere."""
    global _configured
    if _configured:
        return

    if os.getenv("RELEASEFORGE_TRACING_ENABLED", "true").lower() == "false":
        _configured = True
        return

    endpoint = os.getenv("RELEASEFORGE_OTEL_ENDPOINT", "http://localhost:4318/v1/traces")
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer():
    return trace.get_tracer(SERVICE_NAME)


def record_exception(span, exc: Exception) -> None:
    """Marks a span as failed with the real exception, not just a boolean flag — matches
    what an OTel-aware backend (Jaeger, an APM tool) expects to render an error trace."""
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
