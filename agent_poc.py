import os
import json
from typing import TypedDict, List

from fastapi import FastAPI
import uvicorn

# OpenTelemetry / OpenInference / Phoenix imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

# LangChain / LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv # Added for loading API key from .env
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# --- 1. Observability Configuration ---
# The Phoenix OTLP collector typically listens on port 6006
OTLP_ENDPOINT = "http://localhost:6006/v1/traces"

resource = Resource.create({"service.name": "trip-planner-agent"})
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT)))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

# Automatically capture LangChain LLM calls with OpenInference semantics
# This will capture token counts, model names, and prompts as child spans
LangChainInstrumentor().instrument()

# --- 2. Agent Logic & Graph ---
class AgentState(TypedDict):
    user_request: str
    parsed_request: dict
    itinerary: List[dict] # Changed to List[dict] for clarity
    final_response: str


# Ensure OPENAI_API_KEY is set in your environment
# Try loading from 'env' first, then fallback to standard '.env'
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

def parse_request_node(state: AgentState):
    with tracer.start_as_current_span("parse_request") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        prompt = f"Extract trip details from: '{state['user_request']}'. Return JSON: city, days (int), budget, interests (list), pace."
        response = llm.invoke([SystemMessage(content="You are a parser."), HumanMessage(content=prompt)])
        # Basic cleaning of LLM output for JSON parsing
        content = response.content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)
        span.set_attribute("output.value", json.dumps(parsed))
        return {"parsed_request": parsed}

def planner_node(state: AgentState):
    with tracer.start_as_current_span("planner_node") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
        req = state['parsed_request']
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
        enriched = []
        for day_data in state['itinerary']:
            day_plan = {"day": day_data['day'], "activities": []}
            for act in day_data['activities']:
                # Child LLM spans are automatically nested under 'enricher_node'
                desc = llm.invoke([HumanMessage(content=f"One line description for {act} in {state['parsed_request']['city']}.")]).content
                day_plan['activities'].append({"name": act, "details": desc})
            enriched.append(day_plan)
        return {"itinerary": enriched}

def validator_node(state: AgentState):
    with tracer.start_as_current_span("validator_node") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
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

# --- 3. FastAPI App ---
app = FastAPI()

@app.get("/plan")
async def plan_trip(query: str):
    # Root span for the transaction
    with tracer.start_as_current_span("agent_trip_planner") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value)
        span.set_attribute("input.value", query)
        result = agent.invoke({"user_request": query})
        span.set_attribute("output.value", result["final_response"])
        return {"plan": result["final_response"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)