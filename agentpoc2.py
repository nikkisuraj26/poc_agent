# agent_poc.py
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

# --- Import our Plug & Play Telemetry ---
from telemetry import setup_telemetry, trace_node

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# Initialize Telemetry ONCE
setup_telemetry("trip-planner-agent")

load_dotenv()
logger = logging.getLogger("trip-planner-logger")
logger.setLevel(logging.INFO)

class AgentState(TypedDict):
    user_request: str
    parsed_request: dict
    weather_forecast: str 
    itinerary: List[dict]
    final_response: str

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)

# --- Nodes (Notice how clean these are now!) ---

@trace_node("parse_request")
def parse_request_node(state: AgentState):
    logger.info(f"Parsing user request: {state['user_request']}")
    prompt = f"Extract trip details from: '{state['user_request']}'. Return JSON: city, days (int), budget, interests (list), pace, date (YYYY-MM-DD format, if specified, otherwise null)."
    
    response = llm.invoke([SystemMessage(content="You are a parser."), HumanMessage(content=prompt)])
    content = response.content.replace("```json", "").replace("```", "").strip()
    
    parsed = json.loads(content)
    if parsed.get('date') is None:
        parsed['date'] = datetime.now().strftime('%Y-%m-%d')
    
    return {"parsed_request": parsed}

@trace_node("planner_node")
def planner_node(state: AgentState):
    req = state['parsed_request']
    logger.info(f"Generating itinerary plan for {req.get('city')}")
    
    prompt = (
        f"Plan {req.get('days')} days in {req.get('city')} for someone interested in {req.get('interests')}. "
        f"Pace should be {req.get('pace', 'moderate')}. "
        "You MUST return a JSON object with two keys: \n"
        "1. 'reasoning': A detailed paragraph explaining WHY...\n"
        "2. 'days': The actual itinerary array.\n"
        "Format: {\"reasoning\": \"...\", \"days\": [{\"day\": 1, \"activities\": [\"act1\"]}]}"
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.replace("```json", "").replace("```", "").strip()
    plan = json.loads(content)
        
    return {"itinerary": plan['days']}

@trace_node("enricher_node")
def enricher_node(state: AgentState):
    logger.info("Enriching itinerary activities with descriptions")
    enriched = []
    city = state['parsed_request']['city']

    for day_data in state['itinerary']:
        day_plan = {"day": day_data['day'], "activities": []}
        for act in day_data['activities']:
            topic_str = act.get('name', act.get('activity', act.get('title', str(act)))) if isinstance(act, dict) else str(act)
            
            # Using LLM directly to simulate enrichment instead of separate tool for brevity
            desc = llm.invoke([HumanMessage(content=f"Write a 1 line description for {topic_str} in {city}.")]).content
            day_plan['activities'].append({"name": topic_str, "details": desc})
        enriched.append(day_plan)
        
    return {"itinerary": enriched}

@trace_node("validator_node")
def validator_node(state: AgentState):
    logger.info("Validating and formatting final response")
    itinerary = state['itinerary']
    city = state['parsed_request']['city']
    
    output = f"### Final Itinerary for {city} ###\n\n"
    for d in itinerary:
        output += f"\n**Day {d['day']}**:\n" + "\n".join([f"- **{a['name']}**: {a['details']}" for a in d['activities']])
        
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

# --- FASTAPI Endpoints ---
app = FastAPI()

@app.get("/plan")
async def plan_trip(query: str):
    # LangChainInstrumentor automatically catches the execution of `agent.invoke()` 
    # and all the nodes inside it because we decorated them!
    result = agent.invoke({"user_request": query})
    return {"plan": result["final_response"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)