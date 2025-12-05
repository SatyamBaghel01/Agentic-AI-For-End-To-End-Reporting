import streamlit as st
import requests
import base64
import pandas as pd

BACKEND_URL = "http://127.0.0.1:8000/query"

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Steel Plant Text2SQL",
    layout="wide",
)

# ------------------ THEME ------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

chosen = st.sidebar.radio(
    "Theme", ["Dark", "Light"], key="theme_toggle"
)

if chosen == "Dark":
    st.session_state.theme = "dark"
    st.markdown("""
        <style>
        body { background-color:#0d0d0d; }
        .stApp { background-color:#0d0d0d; color:white; }
        </style>
    """, unsafe_allow_html=True)

else:
    st.session_state.theme = "light"
    st.markdown("""
        <style>
        body { background-color:white; }
        .stApp { background-color:white; color:black; }
        </style>
    """, unsafe_allow_html=True)


# ------------------ SIDEBAR QUICK ACTIONS ------------------
st.sidebar.markdown("### ⚡ Quick Actions")

suggested_queries = [
    "Compare average spindle temperature (NCH21) across all machines for the last 7 days.",
    "Show power consumption (NCH23) trend of each Machine in the last 10 days.",
    "Plot axis motor temperature (NCH13) vs spindle load (NCH19) for machine HR_Mill_01.",
    "Which machine had the highest spindle RPM (NCH20) in the last 5 days?",
    "Show daily average spindle load (NCH19) for each division over the last 15 days.",
    "Plot energy usage (NCH23) vs production speed (NCH20) scatter chart.",
    "Show alarm distribution (NCH25) for the entire plant.",
    "Compare feedrate (NCH17) across machines for last 48 hours.",
    "Plot hourly temperature variation (NCH21) for SM_Conv_02 over last 2 days.",
    "Trend of voltage stability (NCH22) for all machines over last week."
]
for i, q in enumerate(suggested_queries):
    if st.sidebar.button(q, key=f"suggest{i}"):
        st.session_state.user_question = q

# st.sidebar.markdown("#### Graph Test Queries")
# for i, q in enumerate(graph_test_queries):
#     if st.sidebar.button(q, key=f"graph_suggest{i}"):
#         st.session_state.user_question = q


# ------------------ MAIN HEADER ------------------
st.title("Agentic AI For End-To-End Reporting")
st.caption("Ask production, machine, or spindle related questions")

# ------------------ USER INPUT ------------------
user_question = st.text_input(
    "Ask your question:",
    value=st.session_state.get("user_question", "")
)

run_btn = st.button("Run Query", key="run_query_btn")


# ------------------ EXECUTION ------------------
if run_btn and user_question.strip():
    with st.spinner("Running query through AI pipeline..."):

        payload = {
            "user_question": user_question,
            "username": "streamlit_user"
        }

        try:
            response = requests.post(BACKEND_URL, json=payload, timeout=120)
        except Exception as e:
            st.error("Backend connection failed.")
            st.text(str(e))
            st.stop()

        if response.status_code != 200:
            st.error(f"Backend Error {response.status_code}")
            st.text(response.text)
            st.stop()

        data = response.json()


        # ------------------ GPT-STYLE TABS ------------------
        tab_sql, tab_expl, tab_table, tab_graph, tab_logs = st.tabs(
            ["🧠 SQL Query", "📘 Explanation", "📊 Table", "📈 Graph", "📄 Agent Logs"]
        )


        # ------------------ SQL TAB ------------------
        with tab_sql:
            st.subheader("Generated SQL")
            sql_query = data.get("SQL", "NA")
            if isinstance(sql_query, str):
                st.code(sql_query, language="sql")
        # If multiple queries returned (dictionary from compiler)
            elif isinstance(sql_query, dict):
                for name, query in sql_query.items():
                    st.markdown(f"#### 🔹 {name.capitalize()} Query")
                    st.code(query, language="sql")
            else:
                st.info("No SQL generated.")


        # ------------------ EXPLANATION TAB ------------------
        with tab_expl:
            st.subheader("Explanation")
            st.markdown(data.get("Explanation", "NA"))


        # ------------------ TABLE TAB ------------------
        with tab_table:
            st.subheader("Result Table")

            table = data.get("Table")
            if table == "NA":
                st.info("No table output.")
            else:
                try:
                    df = pd.DataFrame(table)
                    st.dataframe(df, width="stretch")
                except:
                    st.write(table)


        # ------------------ GRAPH TAB ------------------
        with tab_graph:
            st.subheader("Graph Output")

            graph = data.get("Graph")

            if isinstance(graph, dict) and graph.get("plot_created"):
                try:
                    img_bytes = base64.b64decode(graph["image_base64"])
                    st.image(img_bytes, caption=graph.get("title", "Generated Graph"), use_container_width=True)
                except Exception as e:
                    st.error(f"Graph render failed: {str(e)}")
            else:
                st.info("No graph generated.")


        # ------------------ AGENT LOGS ------------------
        with tab_logs:
            st.subheader("📄 Agent Logs (Full Pipeline)")

            st.markdown("### 🔮 Planner Output")
            st.json(data.get("Planner", {}))

            st.markdown("### 🏗️ Raw SQL Generator Output")
            sql_log = data.get("SQL", "NA")
            if isinstance(sql_log, dict):
                for name, query in sql_log.items():
                    st.markdown(f"##### 🔹 {name.capitalize()} Query")
                    st.code(query, language="sql")
            else:
                st.code(sql_log, language="sql")

            st.markdown("### 🧪 Validator Executed SQL")
            v_sql = data.get("ValidatedSQL", "NA")
            if isinstance(v_sql, dict):
                for name, query in v_sql.items():
                    st.markdown(f"##### 🔹 {name.capitalize()} (Validated)")
                    st.code(query, language="sql")
            else:
                st.code(v_sql, language="sql")


            st.markdown("### 📥 Raw Query Result (Before Processing)")
            st.json(data.get("RawQueryResult", {}))

            st.markdown("### 🔍 Post-Processing Required?")
            st.write(data.get("PostProcessingRequired", False))

            st.markdown("### 🧮 Generated Python Code (if any)")
            st.code(data.get("GeneratedPython", ""), language="python")

            st.markdown("### 📘 Interpreter JSON Output")
            st.json(data.get("InterpreterJSON", {}))

            st.markdown("### 📊 Graph Agent Output")
            st.json(data.get("GraphAgentOutput", {}))

            st.info("🚀 All Agent Stages are now visible end-to-end!")