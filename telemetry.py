# telemetry.py
import os
import json
import logging
from functools import wraps
from opentelemetry import trace, metrics
from opentelemetry.trace import StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

def setup_telemetry(service_name: str = "trip-planner-agent"):
    """Call this once at the startup of your application."""
    resource = Resource.create({"service.name": service_name})

    # 1. Traces
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint="http://127.0.0.1:6006/v1/traces"))
    )
    trace.set_tracer_provider(tracer_provider)

    # 2. Metrics
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint="http://127.0.0.1:4318/v1/metrics")
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # 3. Auto-Instrumentors (The Plug & Play Magic)
    LangChainInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    
    logging.info(f"Telemetry initialized for {service_name}")

def trace_node(node_name: str, span_kind=OpenInferenceSpanKindValues.CHAIN.value):
    """
    A plug-and-play decorator to automatically trace LangGraph nodes.
    It captures the incoming state, outgoing state, and any errors automatically.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(node_name) as span:
                span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, span_kind)
                
                # Auto-log incoming state
                try:
                    span.set_attribute("agent.state.incoming", json.dumps(state, default=str))
                except Exception:
                    span.set_attribute("agent.state.incoming", str(state))

                try:
                    # Run actual business logic
                    result = func(state, *args, **kwargs)
                    
                    # Auto-log outgoing result
                    try:
                        span.set_attribute("agent.state.outgoing", json.dumps(result, default=str))
                    except Exception:
                        span.set_attribute("agent.state.outgoing", str(result))
                        
                    return result
                
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR, str(e))
                    raise
        return wrapper
    return decorator