
import os
import json
import re
import sqlite3
import pandas as pd
from datetime import datetime, date
from graph_agent import GraphPlottingAgent
#graph_agent = GraphPlottingAgent()

from dotenv import load_dotenv
from typing import TypedDict, Any
from typing import Dict, Any, Optional, TypedDict, Literal
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
import uuid
import base64

from graph_agent import GraphPlottingAgent

# LangGraph + LangChain
from langchain_core.output_parsers import JsonOutputParser
#from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# Groq LLM
from langchain_groq import ChatGroq

#------blocking Dr0p truncate etc queries-------
DANGEROUS_TOKENS = [
    r"\bDROP\s+TABLE\b",
    r"\bDELETE\s+FROM\b",
    r"\bUPDATE\s+\w+",
    r"\bALTER\s+TABLE\b",
    r"\bTRUNCATE\b",
    r";\s*--",   # comment injection
    r"/\*.*?\*/"  # block comments
]

def is_sql_safe(sql: str) -> tuple[bool, str]:
    for pattern in DANGEROUS_TOKENS:
        if re.search(pattern, sql, re.I):
            return False, f"Disallowed SQL operation detected: {pattern}"
    return True, "OK"

#------JSON extraction more robust-----
def extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end+1])
    except Exception:
        # Optional: try shrinking from the end
        for i in range(end, start, -1):
            try:
                return json.loads(text[start:i+1])
            except Exception:
                continue
    return None
##---avoid token limit---
def safe_llm_view(data):
    """
    Do NOT truncate data used in graphs or final calculations.
    Only reduce what is sent to LLM to prevent token overload.
    """
    if not isinstance(data, list):
        return data

    MAX_ROWS_TO_SEND = 50

    if len(data) <= MAX_ROWS_TO_SEND:
        return data

    preview = data[:MAX_ROWS_TO_SEND]
    preview.append({
        "__note__": f"... {len(data) - MAX_ROWS_TO_SEND} more rows hidden from LLM"
    })
    return preview

load_dotenv()

def extract_sql_from_llm(text: str) -> str:
    # 1) Try fenced ```sql``` block
    m = re.search(r"```sql\s*(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()

    # 2) Any fenced ``` ``` block, then find SELECT ... ;
    m = re.search(r"```(.*?)```", text, re.S)
    if m:
        content = m.group(1)
        m2 = re.search(r"(SELECT[\s\S]*?;)", content, re.I)
        if m2:
            return m2.group(1).strip()

    # 3) Directly in the whole text
    m = re.search(r"(SELECT[\s\S]*?;)", text, re.I)
    if m:
        return m.group(1).strip()

    # 4) Fallback: return as-is
    return text.strip()

##added to take latest dates from DATABASE  not current dates
def get_database_max_date(table="EMS_Daily"):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(f'SELECT MAX("time") FROM "{table}"')
        result = cursor.fetchone()[0]

    # If DB is empty or NULL
    if not result:
        return date.today()

    # Convert string -> datetime.date object
    return datetime.strptime(result.split(" ")[0], "%Y-%m-%d").date()

# ================== CONFIG ==================
DB_PATH = "steel.db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"   # You can also try: mixtral-8x7b

STATIC_IMAGES_DIR = "static/images"
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

with open("db_metadata.json", "r") as f:
    metadata = json.load(f)
if "columns_metadata" in metadata and "db_schema_metadata" not in metadata:
    metadata["db_schema_metadata"] = metadata["columns_metadata"]

# Filter tables based on what actually exists in DB
def get_existing_tables():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            r = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            return {row[0] for row in r}
    except:
        return set()

EXISTING_TABLES = get_existing_tables()

# Filter metadata so planner gets ONLY real tables
metadata["table_metadata"] = [
    tbl for tbl in metadata.get("table_metadata", [])
    if tbl["table_name"] in EXISTING_TABLES
]

CURRENT_DATE = get_database_max_date()
CURRENT_YEAR = CURRENT_DATE.year

graph_agent = GraphPlottingAgent(api_key=GROQ_API_KEY)

# ================== STATE ==================
class Text2SQLState(TypedDict):
    question: str
    username: str
    structured_intent: Any
    sql_query: str | None
    query_result: Any
    post_processing_required: bool | None
    python_code: str | None
    final_result: Any
    explanation: dict | None
    graph_output: Any | None   # 👈 ADD THIS

#=======Graph agent node function=========
def graph_agent_node(state: Text2SQLState):
    exp = state.get("explanation") or {}

    agent_result = graph_agent.analyze_and_plot(
        user_question=state["question"],
        explanation=exp.get("Explanation", ""),
        table_data=exp.get("Table", "NA"),
        graph_suggestion=exp.get("Graph", "")
    )
    #  # Store graph inside workflow state
    # state["graph_output"] = agent_result.get("Graph", {})
    # # Return state so next node receives updated values
    # return state
    return {
        "graph_output": agent_result.get("Graph", {"plot_created": False})
    }

# ================== DATABASE ==================
def execute_sql_query(query: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(query.replace("%", "%%"))

            if cur.description:
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                result = [dict(zip(columns, row)) for row in rows]
            else:
                result = {"message": "Query executed."}

        return result
    except Exception as e:
       return {"message": "SQL execution failed", "error": str(e)}

#===================graph saving function===========
def generate_graph_image(data):
    import matplotlib.pyplot as plt
    import uuid

    df = pd.DataFrame(data)

    if df.empty or len(df.columns) < 2:
        return "NA"

    filename = f"graph_{uuid.uuid4().hex}.png"
    filepath = os.path.join("static/images", filename)

    df.plot()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return f"http://127.0.0.1:8000/static/images/{filename}"

# ================== METADATA ==================
def get_metadata_context():
    schema = "\n".join(
        [f"- {col['column_name']}: {col['description']} ({col['column_datatype']})"
         for col in metadata.get("db_schema_metadata", [])]
    )

    tables = "\n".join(
        [f"- {tbl['table_name']}" for tbl in metadata.get("table_metadata", [])]
    )

    return schema, tables

DB_SCHEMA, TABLES = get_metadata_context()

# ================== LLM ==================
def get_llm(temp=0):
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=temp
    )

# ================== PROMPTS ==================

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are the Planner Agent of a Text2SQL system for a steel plant database.

Your job is to convert a natural language question into a structured JSON plan.

Available tables:
{TABLES}

Available columns:
{DB_SCHEMA}

Current Date: {CURRENT_DATE}

STRICT RULES:
- NEVER hallucinate table names or columns.
- ONLY use tables from the provided list.
- Choose the best table ONLY from the available list.
- If query is not answerable from database, set action="refuse".
- When applying date filters, ALWAYS use DATE("time").
RULE FOR "last N rows":
- If user asks for "last N" or "latest N" or "recent N rows":
    * DO NOT aggregate
    * DO NOT group
    * MUST set: "limit_rows": N     
 Additional Chart Rules:
- If the question mentions trend, variation, over time, history, monitor, pattern, or increasing/decreasing → action MUST be "chart".
- If any time or date column is required (e.g., "time", "date", "yesterday", "last week", "today", "last 7 days"), also treat as "chart" unless question explicitly demands an aggregate number only.
- "chart" means SQL should return raw data with time on x-axis and numeric column on y-axis (no aggregation unless specifically asked). 
- If the question requires filtering or selecting TOP-N (e.g., "top 5 machines", "highest", "lowest", "compare first"), treat the action as "query" even if a chart is implied. Post-processing will handle chartability.        
***When filtering by date, ALWAYS use:
DATE("time") between start_date and end_date
Never compare raw "time" column directly to a date string.
Return ONLY valid JSON in this exact format:

{{ 
  "action": "query | chart | metadata | refuse",
  "data_source": "EMS_Daily",
  "filters": {{}},
  "time_range": {{
      "start_date": null,
      "end_date": null
  }},
  "columns": [],
  "aggregation": {{
      "group_by": null,
      "method": null
  }},
  "metadata_type": null
}}

Example:

User Question: Show average spindle speed yesterday

Return:
{{
  "action": "query",
  "data_source": "EMS_Daily",
  "filters": {{}},
  "time_range": {{
      "start_date": "2025-09-29",
      "end_date": "2025-09-29"
  }},
  "columns": ["NCH20"],
  "aggregation": {{
      "group_by": "MachineName",
      "method": "avg"
  }},
  "metadata_type": null
}}
"""),

    ("human", "{question}")
])






generator_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an SQLite SQL Generator Agent for a steel plant Text2SQL system.

STRICT RULES:
- Use ONLY valid SQLite syntax
- ALWAYS wrap table and column names in double quotes
- NEVER hallucinate columns or tables
- Only use schema and tables given below
- If query is impossible, return: -- NOT POSSIBLE
- DO NOT explain anything
- Return ONLY pure SQL

***When applying date filters:
Use this format:
WHERE DATE("time") BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
Never use direct "time" = 'YYYY-MM-DD'

Tables:
""" + TABLES + """

Schema:
""" + DB_SCHEMA + """

For "chart" actions, NEVER aggregate values unless explicitly asked.
Always return raw time-series values using:
SELECT "time", "<column>"
FROM "<table>"
WHERE DATE("time") BETWEEN ...
ORDER BY "time" ASC;

Example 1:
User: Show last 10 rows of spindle speed
SQL:
SELECT "time", "NCH20"
FROM "EMS_Daily"
ORDER BY "time" DESC
LIMIT 10;

Example 2:
User: Average spindle current by machine
SQL:
SELECT "MachineName", AVG("NCH18") AS avg_spindle_current
FROM "EMS_Daily"
GROUP BY "MachineName";
"""
    ),
    (
        "human",
        "User Question: {question}\nStructured Intent: {structured_intent}"
    )
])





validator_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an SQLite SQL Validator Agent.

Fix errors if any.
Ensure only allowed tables and columns are used.

Tables:
{TABLES}

Columns:
{DB_SCHEMA}

Return corrected query inside ```sql``` block.
"""),
    ("human", """
Question: {question}
Intent: {structured_intent}
SQL: {sql_query}
""")
])





decision_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a Decision Agent that decides whether Python post-processing is required
for the SQL query result.

You MUST respond with a JSON object with a single key "decision".
The value of "decision" must be either:
- "POST_PROCESSING_REQUIRED"
- "NO_POST_PROCESSING_REQUIRED"

Guidelines:
- If the raw SQL result already directly answers the user's question in a clean way,
  choose "NO_POST_PROCESSING_REQUIRED".
- If the user asks for derived metrics, comparisons, rankings, percentages,
  thresholds, or any non-trivial transformation, choose "POST_PROCESSING_REQUIRED".

Return ONLY the JSON object. No explanations, no extra text.
"""
    ),
    (
        "human",
        """
User question:
{question}

SQL result (as JSON-like data):
{query_result}
"""
    ),
])




code_generator_prompt = ChatPromptTemplate.from_messages([
    ("system", """
Generate Python pandas code for post-processing.
No plotting code.
Only processing + final data.
"""),
    ("human", """
Question: {question}
SQL Result: {query_result}
""")
])




interpreter_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are the Result Interpreter Agent for a Text2SQL system.

Your job is to:
- Explain the final result to the user in clear language.
- Optionally provide a small tabular view of the data.
- Optionally suggest if a graph would be useful.

You MUST return a JSON object with these keys:
- "Explanation": a clear natural language explanation (string)
- "Table": either:
    - a list of row objects suitable for display in a table, or
    - the string "NA" if no table is needed
- "Graph": either:
    - a short description of what kind of graph would be useful
      (for example: "line chart of NCH20 over time by MachineName"), or
    - the string "NA" if no visualization is suggested

The "Explanation" should be understandable to an engineer in the steel plant.
Do not return any extra keys. Do not add text outside the JSON object.
"""
    ),
    (
        "human",
        """
User question:
{question}

Final processed result:
{final_result}
"""
    ),
])


# ================== AGENTS ==================

def planner(state: Text2SQLState):
    llm = get_llm()
    chain = planner_prompt | llm

    result = chain.invoke({
        "question": state["question"],
        "TABLES": TABLES,
        "DB_SCHEMA": DB_SCHEMA,
        "CURRENT_DATE": CURRENT_DATE.isoformat()
    })

    # raw = result.content if hasattr(result, "content") else str(result)

    # print("\n=== PLANNER RAW OUTPUT ===\n", raw, "\n========================\n")

    # intent = None
    # try:
    #     start = raw.find("{")
    #     end = raw.rfind("}")
    #     if start != -1 and end != -1:
    #         json_str = raw[start:end+1]
    #         intent = json.loads(json_str)
    # except Exception as e:
    #     print("Planner JSON extraction failed:", e)

    # # Absolute fallback so it NEVER crashes
    # if intent is None:
    #     intent = {
    #         "action": "metadata",
    #         "data_source": "EMS_Daily",
    #         "filters": {},
    #         "time_range": {
    #             "start_date": None,
    #             "end_date": None
    #         },
    #         "columns": [],
    #         "aggregation": {
    #             "group_by": None,
    #             "method": None
    #         },
    #         "metadata_type": "schema_overview"
    #     }

    # return {"structured_intent": intent}
    raw = result.content if hasattr(result, "content") else str(result)
    print("\n=== PLANNER RAW OUTPUT ===\n", raw, "\n========================\n")

    intent = extract_json_object(raw)
    if isinstance(intent, dict) and "limit_rows" in intent:
        try:
            intent["limit_rows"] = int(intent["limit_rows"])
        except:
            intent["limit_rows"] = None

    if intent is None:
        intent = {
            "action": "metadata",
            "data_source": "EMS_Daily",
            "filters": {},
            "time_range": {
                "start_date": None,
                "end_date": None
            },
            "columns": [],
            "aggregation": {
                "group_by": None,
                "method": None
            },
            "metadata_type": "schema_overview"
        }
    # ⚠️ Fallback: Prevent large SELECT * returning huge data to LLM
    if intent.get("columns") == [] or intent.get("columns") is None:
    # Use only commonly meaningful default columns for raw preview
    # Time + top metrics (Edit if required)
        intent["columns"] = ["time", "NCH20", "NCH23"]
    return {"structured_intent": intent}



def generator(state: Text2SQLState):
    llm = get_llm()
    chain = generator_prompt | llm

    result = chain.invoke({
        "question": state["question"],
        "structured_intent": state["structured_intent"]
    })

    raw = result.content if hasattr(result, "content") else str(result)
    sql_query = extract_sql_from_llm(raw)
    return {"sql_query": sql_query}


def validator(state: Text2SQLState):
    llm = get_llm()
    chain = validator_prompt | llm

    result = chain.invoke({
        "question": state["question"],
        "structured_intent": state["structured_intent"],
        "sql_query": state["sql_query"],
        "TABLES": TABLES,
        "DB_SCHEMA": DB_SCHEMA
    })

    raw = result.content if hasattr(result, "content") else str(result)
    final_sql = extract_sql_from_llm(raw)

    safe, reason = is_sql_safe(final_sql)
    if not safe:
        return {
            "sql_query": final_sql,
            "query_result": {"error": reason}
        }

    query_result = execute_sql_query(final_sql)
    return {"sql_query": final_sql, "query_result": query_result}


def decision_maker(state: Text2SQLState):
    llm = get_llm()
    chain = decision_prompt | llm | JsonOutputParser()

    result = safe_llm_view(state["query_result"])
    # Skip post-processing for raw time-series chart
    intent = state.get("structured_intent", {})
    if intent.get("action") == "chart" and intent.get("aggregation", {}).get("method") is None:
         return {
        "post_processing_required": False,
        "final_result": state["query_result"]
    }

    # Reduce token load
    if isinstance(result, list) and len(result) > 50:
        result = result[:50]

    decision = chain.invoke({
        "question": state["question"],
        "query_result": result
    })

    required = decision["decision"] == "POST_PROCESSING_REQUIRED"
    return {
        "post_processing_required": required,
        "final_result": state["query_result"] if not required else None
    }


def code_generator(state: Text2SQLState):
    llm = get_llm()
    chain = code_generator_prompt | llm

    raw = state.get("query_result")

    # Use safe_llm_view so we never blow the token limit,
    # but keep the *full* raw result for computation & graphs.
    preview = safe_llm_view(raw)

    # Optional: give LLM explicit schema to make code more general
    schema = []
    if isinstance(raw, list) and raw:
        schema = list(raw[0].keys())

    result = chain.invoke({
        "question": state["question"],
        "query_result": {
            "schema": schema,
            "sample": preview
        }
    })

    return {"python_code": result.content}


# Interpreter
def interpreter(state: Text2SQLState):
    llm = get_llm()
    chain = interpreter_prompt | llm | JsonOutputParser()

    # Always pass final_result from SQL or post-processing
    if state.get("final_result") is None and state.get("query_result"):
        state["final_result"] = state["query_result"]

    final = safe_llm_view(state["final_result"])

    # Trim big outputs for LLM reasoning
    trimmed = final
    if isinstance(trimmed, list) and len(trimmed) > 50:
        trimmed = trimmed[:50]

    explanation = chain.invoke({
        "question": state["question"],
        "final_result": trimmed
    })

    # ❗ Always preserve the actual table data if available
    if explanation.get("Table") == "NA" and isinstance(final, list) and len(final) > 0:
        explanation["Table"] = final

    # ❗ Don’t suppress suggested Graph if LLM generated it
    if explanation.get("Graph") == "NA" and isinstance(final, list) and len(final) > 0:
        explanation["Graph"] = "auto"

    return {"explanation": explanation}


# ================== LANGGRAPH ==================

graph = StateGraph(Text2SQLState)

graph.add_node("planner", planner)
graph.add_node("generator", generator)
graph.add_node("validator", validator)
graph.add_node("decision", decision_maker)
graph.add_node("code_generator", code_generator)
graph.add_node("interpreter", interpreter)

# ✅ NEW — Graph Agent Node
graph.add_node("graph_agent", graph_agent_node)

graph.add_edge("planner", "generator")
graph.add_edge("generator", "validator")
graph.add_edge("validator", "decision")
def route_after_decision(state):
    intent = state.get("structured_intent", {})
    if intent.get("action") == "chart" and intent.get("aggregation", {}).get("method") is None:
        return "interpreter"
    return "code_generator" if state["post_processing_required"] else "interpreter"
graph.add_conditional_edges(
    "decision",
    route_after_decision,
    {
        "code_generator": "code_generator",
        "interpreter": "interpreter"
    }
)

graph.add_edge("code_generator", "interpreter")

# ❗ Old: interpreter → END  
# ❗ New: interpreter → graph_agent → END
#graph.add_node("graph_agent", graph_agent_node)
graph.add_edge("interpreter", "graph_agent")
graph.add_edge("graph_agent", END)

graph.set_entry_point("planner")

workflow = graph.compile()


# ================== FASTAPI ==================

class QueryRequest(BaseModel):
    user_question: str
    username: str = "guest"

class QueryResponse(BaseModel):
    SQL: str
    Explanation: str
    Table: Any
    Graph: str
    

@app.post("/query")
async def handle_query(query: QueryRequest):

    state: Text2SQLState = {
        "question": query.user_question,
        "username": query.username,
        "structured_intent": None,
        "sql_query": None,
        "query_result": None,
        "post_processing_required": None,
        "python_code": None,
        "final_result": None,
        "explanation": None,
        "graph_output": None
    }

    final_state = workflow.invoke(state)
    explanation = final_state["explanation"]

    graph_output = final_state.get("graph_output", None)   # from graph agent

    return {
    "SQL": final_state.get("sql_query", "NA"),
    "Planner": final_state.get("structured_intent", {}),
    "ValidatedSQL": final_state.get("sql_query", "NA"),   # same for now
    "Explanation": explanation.get("Explanation", "NA"),
    "Table": explanation.get("Table", "NA"),
    "Graph": graph_output if graph_output else {"plot_created": False},
    "RawQueryResult": final_state.get("query_result", {}),
    "PostProcessingRequired": final_state.get("post_processing_required", False),
    "GeneratedPython": final_state.get("python_code", ""),"InterpreterJSON": explanation,
    "GraphAgentOutput": graph_output

}

# ================== RUN ==================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
