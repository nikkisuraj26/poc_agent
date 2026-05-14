import os
import json
from typing import TypedDict, List

from fastapi import FastAPI
import time
import logging
import uvicorn

# OpenTelemetry / OpenInference / Phoenix imports
from opentelemetry import trace
from opentelemetry import metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
#These enable distributed tracing:
# capture every step of the agent
# send traces to Phoenix UI for visualization and analysis
# LangChain / LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv # Added for loading API key from .env
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# --- 1. Observability Configuration ---
# Send traces directly to native Arize Phoenix (started via 'python -m phoenix.server.main serve')
# --- 1. Observability Configuration ---
# Send traces directly to native Arize Phoenix (started via 'python -m phoenix.server.main serve')
OTLP_ENDPOINT = "http://127.0.0.1:6006/v1/traces"

# Send Metrics and Logs to our running Docker Compose OTel Collector
OTLP_METRICS_ENDPOINT = "http://127.0.0.1:4318/v1/metrics"
OTLP_LOGS_ENDPOINT = "http://127.0.0.1:4318/v1/logs"

resource = Resource.create({"service.name": "trip-planner-agent"})

# -- Traces Setup --
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT)))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

# -- Metrics Setup --
metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=OTLP_METRICS_ENDPOINT))
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(__name__)

# Define custom metrics for dashboarding/alerting
request_counter = meter.create_counter("agent_requests_total", description="Total number of agent requests")
error_counter = meter.create_counter("agent_errors_total", description="Total number of agent errors")
duration_histogram = meter.create_histogram("agent_request_duration_seconds", description="Agent request duration")

# -- Logs Setup --
logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=OTLP_LOGS_ENDPOINT)))
set_logger_provider(logger_provider)

# Set up python logging to route through OTel (automatically correlated with traces)
handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
logger = logging.getLogger("trip-planner-logger")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Automatically capture LangChain LLM calls with OpenInference semantics
# This will capture token counts, model names, and prompts as child spans
LangChainInstrumentor().instrument()

# --- 2. Agent Logic & Graph ---
class AgentState(TypedDict):
  user_request: str
  parsed_request: dict
  itinerary: List[dict] # Changed to List[dict] for clarity
  final_response: str



if os.path.exists("env"):
  load_dotenv(dotenv_path="env")
else:
  load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
  raise ValueError("GOOGLE_API_KEY not found. Please ensure your 'env' file contains GOOGLE_API_KEY=your_key")

print(f"Initializing Agent with model: gemini-2.5-flash")
# Explicitly using the stable model name
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)

# --- Mock Tool for Tracking External API / Search Usage ---
@tool
def search_city_info(city: str, topic: str) -> str:
  """Mock tool to simulate searching an external service for city information."""
  logger.info(f"Using external search tool for {topic} in {city}...")
  # Here you would normally use Tavily, Google Search, Wikipedia, or an SQL Database
  time.sleep(0.5) # Simulate network latency
  return f"Found detailed external info: The best {topic} experiences in {city} are highly rated by locals."

def parse_request_node(state: AgentState):
  with tracer.start_as_current_span("parse_request") as span: # we are saying track everything that is happening inside this function as part of the "parse_request" step in our agent's execution
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
    logger.info(f"Parsing user request: {state['user_request']}")
    span.add_event("Executing parse_request LLM call")
    
    prompt = f"Extract trip details from: '{state['user_request']}'. Return JSON: city, days (int), budget, interests (list), pace."
    response = llm.invoke([SystemMessage(content="You are a parser."), HumanMessage(content=prompt)])
    # Basic cleaning of LLM output for JSON parsing
    content = response.content.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(content)
    span.set_attribute("output.value", json.dumps(parsed))
    span.add_event("Parsing complete", attributes={"parsed_keys": str(list(parsed.keys()))})
    return {"parsed_request": parsed}

def planner_node(state: AgentState):
  with tracer.start_as_current_span("planner_node") as span:
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
    req = state['parsed_request']
    logger.info(f"Generating itinerary plan for {req.get('city')}")
    span.add_event("Generating itinerary plan")
    prompt = f"Plan {req.get('days')} days in {req.get('city')} for {req.get('interests')}. Return JSON: {{\"days\": [{{\"day\": 1, \"activities\": [\"act1\"]}}]}}"
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.replace("```json", "").replace("```", "").strip()
    plan = json.loads(content)
    # Ensure we have a valid list of days
    if "days" not in plan or not plan["days"]:
      raise ValueError("LLM failed to generate a structured itinerary.")
    return {"itinerary": plan['days']}

def enricher_node(state: AgentState):
  with tracer.start_as_current_span("enricher_node") as span:
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
    logger.info("Enriching itinerary activities with descriptions")
    enriched = []
    for day_data in state['itinerary']:
      day_plan = {"day": day_data['day'], "activities": []}
      for act in day_data['activities']:
        # 1. First, use our tool to "search" for real information (This gets traced!)
        external_info = search_city_info.invoke({"city": state['parsed_request']['city'], "topic": act})
        
        # 2. Provide the external info to the LLM
        # Child LLM spans are automatically nested under 'enricher_node'
        desc = llm.invoke([HumanMessage(content=f"Based on this info: '{external_info}', write a 1 line description for {act}.")]).content
        day_plan['activities'].append({"name": act, "details": desc})
      enriched.append(day_plan)
    return {"itinerary": enriched}

def validator_node(state: AgentState):
  with tracer.start_as_current_span("validator_node") as span:
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
    logger.info("Validating and formatting final response")
    itinerary = state['itinerary']
    city = state['parsed_request']['city']
    output = f"### Final Itinerary for {city} ###\n"
    for d in itinerary:
      output += f"\nDay {d['day']}:\n" + "\n".join([f"- {a['name']}: {a['details']}" for a in d['activities']])
    span.set_attribute("output.value", output)
    return {"final_response": output}

# Assemble the LangGraph
builder = StateGraph(AgentState)
builder.add_node("parse_request", parse_request_node)
builder.add_node("planner", planner_node)
builder.add_node("enricher", enricher_node)
builder.add_node("validator", validator_node)

builder.set_entry_point("parse_request")
builder.add_edge("parse_request", "planner")
builder.add_edge("planner", "enricher")
builder.add_edge("enricher", "validator")
builder.add_edge("validator", END)
agent = builder.compile()
# User → Parse → Plan → Enrich → Validate → Output
# --- 3. FastAPI App ---
app = FastAPI()

@app.get("/plan")
async def plan_trip(query: str):
  start_time = time.time()
  request_counter.add(1, {"endpoint": "/plan"})
  
  # Root span for the transaction
  with tracer.start_as_current_span("agent_trip_planner") as span:
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
    span.set_attribute("input.value", query)
    
    logger.info(f"Received /plan request: {query}")
    try:
      result = agent.invoke({"user_request": query})
      span.set_attribute("output.value", result["final_response"])
      logger.info("Successfully processed /plan request")
      return {"plan": result["final_response"]}
    except Exception as e:
      error_counter.add(1, {"endpoint": "/plan", "error_type": type(e).__name__})
      logger.error(f"Error processing trip plan: {e}", exc_info=True)
      span.record_exception(e)
      raise
    finally:
      duration = time.time() - start_time
      duration_histogram.record(duration, {"endpoint": "/plan"})

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)


  #python -m phoenix.server.main serve
  #run agent_poc.py
  #curl http://localhost:8000/plan?query=Plan a 1 day trip to warangal which covers food and historical places give simple answer
