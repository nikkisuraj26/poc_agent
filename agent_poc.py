import os
import json
import time
import uuid
import logging
from typing import TypedDict, List
from datetime import datetime, timedelta

from fastapi import FastAPI, Header
import uvicorn
import requests
from dotenv import load_dotenv

# --- OpenTelemetry / OpenInference Imports ---
from opentelemetry import trace, metrics
from opentelemetry.trace import StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

# Auto-Instrumentors (The Global Catch-All Nets)
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# --- LangGraph / LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# ==========================================
# 1. OBSERVABILITY BOOTSTRAP
# ==========================================
load_dotenv()

OTLP_ENDPOINT = "http://127.0.0.1:6006/v1/traces"
OTLP_METRICS_ENDPOINT = "http://127.0.0.1:4318/v1/metrics"
OTLP_LOGS_ENDPOINT = "http://127.0.0.1:4318/v1/logs"

resource = Resource.create({"service.name": "trip-planner-agent"})

# -- Traces --
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT)))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

# -- Metrics --
metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=OTLP_METRICS_ENDPOINT))
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(__name__)

request_counter = meter.create_counter("agent_requests_total", description="Total number of agent requests")
error_counter = meter.create_counter("agent_errors_total", description="Total number of agent errors")
duration_histogram = meter.create_histogram("agent_request_duration_seconds", description="Agent request duration")

# -- Logs --
logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=OTLP_LOGS_ENDPOINT)))
set_logger_provider(logger_provider)

handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
logger = logging.getLogger("trip-planner-logger")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -- Activate the Glass Box Instrumentors --
LangChainInstrumentor().instrument()
RequestsInstrumentor().instrument() # Catches standard sync APIs (OpenWeather)
HTTPXClientInstrumentor().instrument() # Catches async LLM packets (Google Gemini)

# ==========================================
# 2. AGENT LOGIC & GRAPH STATE
# ==========================================
class AgentState(TypedDict):
    user_request: str
    parsed_request: dict
    weather_forecast: str 
    itinerary: List[dict]
    final_response: str

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")

logger.info("Initializing Agent with model: gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)

# --- Tools ---
# @tool
# def search_city_info(city: str, topic: str) -> str:
#     """Mock tool to simulate searching an external service for city information."""
#     logger.info(f"Using external search tool for {topic} in {city}...")
#     time.sleep(0.5) 
#     return f"Found detailed external info: The best {topic} experiences in {city} are highly rated by locals."

@tool
def get_weather_info(city: str, date: str) -> str:
    """Retrieves weather information for a city on a specific date using OpenWeatherMap API."""
    logger.info(f"Using OpenWeatherMap API for weather in {city} on {date}...")
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        logger.error("OPENWEATHER_API_KEY not found.")
        return "Weather information is unavailable due to missing API key."

    base_url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city, "appid": api_key, "units": "metric"}

    # GLASS BOX: Grab the existing LangChain span to inject our payload directly into it
    current_span = trace.get_current_span()
    current_span.set_attribute("http.url", base_url)
    current_span.set_attribute("weather.city_requested", city)
    current_span.set_attribute("weather.date_requested", date)

    try:
        response = requests.get(base_url, params=params)
        current_span.set_attribute("http.status_code", response.status_code)
        response.raise_for_status() 
        data = response.json()

        # Dump raw API response directly into the Events tab of the Tool span
        current_span.add_event("Raw API Payload Received", attributes={"payload_json": json.dumps(data)})

        if data.get("cod") != "200":
            error_msg = data.get('message', 'Unknown error')
            current_span.set_attribute("error.message", error_msg)
            return f"Could not retrieve weather for {city}. Reason: {error_msg}."

        forecast_list = data.get("list", [])
        if not forecast_list:
            return f"No forecast data available for {city}."

        target_date = datetime.strptime(date, '%Y-%m-%d').date()
        closest_forecast = None
        min_time_diff = timedelta.max

        for forecast_entry in forecast_list:
            forecast_dt_str = forecast_entry.get("dt_txt")
            if forecast_dt_str:
                forecast_datetime = datetime.strptime(forecast_dt_str, '%Y-%m-%d %H:%M:%S')
                if forecast_datetime.date() == target_date:
                    time_diff = abs(forecast_datetime - datetime.combine(target_date, datetime.min.time()))
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_forecast = forecast_entry
        
        if closest_forecast:
            main_data = closest_forecast.get("main", {})
            weather_data = closest_forecast.get("weather", [{}])[0]
            temp = main_data.get("temp")
            feels_like = main_data.get("feels_like")
            description = weather_data.get("description")
            
            final_result = f"The weather in {city} on {date} is expected to be {description} with a temperature of {temp}°C (feels like {feels_like}°C)."
            current_span.set_attribute("weather.parsed_result", final_result)
            return final_result
        else:
            return f"No specific forecast found for {city} on {date}."

    except Exception as e:
        current_span.record_exception(e)
        current_span.set_status(StatusCode.ERROR, str(e))
        logger.error(f"Error fetching weather for {city}: {e}")
        return "Weather information is currently unavailable due to an error."

# --- Nodes ---
def parse_request_node(state: AgentState):
    with tracer.start_as_current_span("parse_request") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute("agent.state.incoming", state['user_request']) # State Transparency
        
        logger.info(f"Parsing user request: {state['user_request']}")
        prompt = f"Extract trip details from: '{state['user_request']}'. Return JSON: city, days (int), budget, interests (list), pace, date (YYYY-MM-DD format, if specified, otherwise null)."
        
        response = llm.invoke([SystemMessage(content="You are a parser."), HumanMessage(content=prompt)])
        content = response.content.replace("```json", "").replace("```", "").strip()
        
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            # Deep Error Tracing for LLM Hallucinations
            span.record_exception(e)
            span.set_attribute("error.raw_llm_output", content)
            span.set_status(StatusCode.ERROR, "LLM failed to output valid JSON")
            raise

        if parsed.get('date') is None:
            parsed['date'] = datetime.now().strftime('%Y-%m-%d')
        
        span.set_attribute("agent.state.outgoing", json.dumps(parsed))
        return {"parsed_request": parsed}

def planner_node(state: AgentState):
    with tracer.start_as_current_span("planner_node") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
        span.set_attribute("agent.state.incoming_parsed", json.dumps(state.get('parsed_request', {})))

        req = state['parsed_request']
        logger.info(f"Generating itinerary plan for {req.get('city')}")
        
        # GLASS BOX: Force Chain-of-Thought (CoT) reasoning
        prompt = (
            f"Plan {req.get('days')} days in {req.get('city')} for someone interested in {req.get('interests')}. "
            f"Pace should be {req.get('pace', 'moderate')}. "
            "You MUST return a JSON object with two keys: \n"
            "1. 'reasoning': A detailed paragraph explaining WHY you chose this layout of days, how it fits the pace, and how you grouped geography.\n"
            "2. 'days': The actual itinerary array.\n"
            "Format: {\"reasoning\": \"...\", \"days\": [{\"day\": 1, \"activities\": [\"act1\"]}]}"
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        try:
            content = response.content.replace("```json", "").replace("```", "").strip()
            plan = json.loads(content)
            
            # Extract the LLM's thought process and bind it directly to the trace
            span.set_attribute("agent.llm_reasoning", plan.get("reasoning", "No reasoning provided"))
            
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("error.raw_llm_output", response.content)
            span.set_status(StatusCode.ERROR, "Failed to parse itinerary JSON")
            raise

        if "days" not in plan or not plan["days"]:
            span.set_status(StatusCode.ERROR, "Invalid itinerary structure")
            raise ValueError("LLM failed to generate a structured itinerary.")
            
        span.set_attribute("agent.state.outgoing_day_count", len(plan['days']))
        return {"itinerary": plan['days']}

def enricher_node(state: AgentState):
    with tracer.start_as_current_span("enricher_node") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        span.set_attribute("agent.state.incoming_itinerary_length", len(state.get('itinerary', [])))
        logger.info("Enriching itinerary activities with descriptions")
        
        enriched = []
        city = state['parsed_request']['city']
        trip_date_str = state['parsed_request'].get('date')

        if trip_date_str:
            weather_forecast = get_weather_info.invoke({"city": city, "date": trip_date_str})
            state['weather_forecast'] = weather_forecast
            span.add_event("Weather Tool Executed", attributes={"weather_result": weather_forecast})

        for day_data in state['itinerary']:
            day_plan = {"day": day_data['day'], "activities": []}
            for act in day_data['activities']:
                
                # --- THE FIX: Sanitize LLM Output ---
                # If the LLM generated a dict instead of a string, extract the text
                if isinstance(act, dict):
                    # Try to grab common keys LLMs use, otherwise convert the whole dict to a string
                    topic_str = act.get('name', act.get('activity', act.get('title', str(act))))
                else:
                    topic_str = str(act)
                # ------------------------------------

                external_info = search_city_info.invoke({"city": city, "topic": topic_str})
                desc = llm.invoke([HumanMessage(content=f"Based on this info: '{external_info}', write a 1 line description for {topic_str}.")]).content
                day_plan['activities'].append({"name": topic_str, "details": desc})
            enriched.append(day_plan)
            
        return {"itinerary": enriched}

def validator_node(state: AgentState):
    with tracer.start_as_current_span("validator_node") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        logger.info("Validating and formatting final response")
        
        itinerary = state['itinerary']
        city = state['parsed_request']['city']
        weather_forecast = state.get('weather_forecast')
        
        output = f"### Final Itinerary for {city} ###\n"
        if weather_forecast:
            output += f"**Weather forecast:** {weather_forecast}\n"
        output += "\n"
        for d in itinerary:
            output += f"\n**Day {d['day']}**:\n" + "\n".join([f"- **{a['name']}**: {a['details']}" for a in d['activities']])
            
        span.set_attribute("output.value", output)
        return {"final_response": output}

# --- Assemble LangGraph ---
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

# ==========================================
# 3. FASTAPI ENDPOINTS
# ==========================================
app = FastAPI()

@app.get("/plan")
async def plan_trip(
    query: str, 
    user_id: str = Header(default="anonymous_user"),
    session_id: str = Header(default_factory=lambda: str(uuid.uuid4()))
):
    start_time = time.time()
    request_counter.add(1, {"endpoint": "/plan"})
  
    with tracer.start_as_current_span("agent_trip_planner") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
        
        # GLASS BOX: Bind context to the root trace
        span.set_attribute("user.id", user_id)
        span.set_attribute("session.id", session_id)
        span.set_attribute("input.value", query)
        
        logger.info(f"Received /plan request from {user_id} in session {session_id}")
        
        try:
            result = agent.invoke({"user_request": query})
            span.set_attribute("output.value", result["final_response"])
            logger.info("Successfully processed /plan request")
            return {"plan": result["final_response"]}
        except Exception as e:
            error_counter.add(1, {"endpoint": "/plan", "error_type": type(e).__name__})
            logger.error(f"Error processing trip plan: {e}", exc_info=True)
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            raise
        finally:
            duration = time.time() - start_time
            duration_histogram.record(duration, {"endpoint": "/plan"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    #python -m phoenix.server.main serve
  #run agent_poc.py
  #curl http://localhost:8000/plan?query=Plan a 1 day trip to vishakapatnam which covers food and historical places give simple answer