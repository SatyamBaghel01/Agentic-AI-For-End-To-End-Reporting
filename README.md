<h1 align="center">🧠 Agentic AI Text2SQL with Autonomous Plotting</h1>

<p align="center">
🚀 <b>Natural Language → SQL → Insights → Charts</b>  
</p>
<p align="center">
Built using <b>FastAPI · LangGraph · Llama3 · Pandas · Matplotlib</b>
</p>

---

## 📚 Table of Contents
- [🔎 Overview](#-overview)
- [💡 Why This Matters](#-why-this-matters)
- [🤖 Multi-Agent Workflow](#-multi-agent-workflow)
- [🛠️ Tech Stack](#-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 How to Run](#-how-to-run)
- [💬 Ask Questions Like](#-ask-questions-like)
- [🧪 API Usage](#-api-usage)
- [🔐 Security & Safety](#-security--safety)
- [🔮 Future Enhancements](#-future-enhancements)
- [⭐ Support](#-support)
- [👨‍💻 Maintainer](#-maintainer)

---

## 🔎 Overview
This system transforms **natural language questions into analytics** using an autonomous **LLM + multi-agent workflow** that:

✔ Understands user intent  
✔ Generates **safe SQL**  
✔ Executes optimized queries  
✔ Applies **post-processing (Pandas)**  
✔ Produces **tables + explanations + visual charts automatically**

> Works with **industrial sensor data, machine KPIs, finance metrics, business logs, telemetry streams, SQL operational data**, and more.

---

## 💡 Why This Matters

Traditional BI & dashboards require:

- ⛔ Manual SQL
- ⛔ Static, predefined charts  
- ⛔ Developer dependency  

📌 **With this system**, analysts can simply ask:

> _“Compare machine efficiency over the last week”_

and instantly receive:

- Generated SQL  
- Computed results  
- 📊 Auto-selected visualization  
- 💬 Natural language insight  

> 👉 **No SQL knowledge required.**

---

## 🤖 Multi-Agent Workflow

| Agent | Task |
|-------|-----|
| 🧠 Planner Agent | Extracts user intent → generates structured JSON plan |
| 🏗 SQL Generator | Converts plan into safe, SQLite-compliant SQL |
| 🔍 SQL Validator | Fixes queries & blocks unsafe SQL (DROP, UPDATE, etc.) |
| 🧮 Decision Agent | Checks if post-processing is required |
| 📊 Pandas Processor | Computes metrics, ranks, comparisons, aggregations |
| 💬 Interpreter Agent | Converts results into readable explanations |
| 🎨 Graph Agent | Auto-detects and plots best visualization |

---

## 🛠️ Tech Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI |
| AI/LLM | Groq (Llama 3.x) |
| Agents | LangGraph + LangChain |
| Database | SQLite *(extendable to MySQL/PostgreSQL)* |
| Processing | Pandas |
| Charts | Matplotlib |
| Frontend | Streamlit *(optional UI)* |

---

## 📁 Project Structure

```bash
Text2SQL_agents/
│
├── backend/
│   ├── main.py                # Multi-agent pipeline
│   ├── graph_agent.py         # Intelligent plotting
│   ├── db_helper.py           # DB utilities
│   ├── populate_*.py          # Sample data loading scripts
│   ├── requirements.txt
│   └── static/images/         # Generated charts
│
├── frontend/
│   ├── app.py                 # Streamlit UI (optional)
│   └── config.toml
│
├── .env                       # LLM keys (ignored in Git)
└── .gitignore                 # Safety rules
```
---
###

## 🚀 How to Run

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/SatyamBaghel01/Agentic-AI-For-End-To-End-Reporting.git
cd Agentic-AI-For-End-To-End-Reporting
```
2️⃣ Create a virtual environment  
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```
3️⃣ Install dependencies  
```bash
pip install -r backend/requirements.txt
```

4️⃣ Add your `.env`  
```
GROQ_API_KEY=your_key_here
```

5️⃣ Start the backend  
```bash
uvicorn backend.main:app --reload
```

6️⃣ (Optional) Start UI  
```bash
streamlit run frontend/app.py
```

---

### 🧪 Ask Questions

```
Show efficiency trend for the last 7 days
Compare motor load between two lines
Give me top 5 machines with highest downtime
Show last 40 sensor entries
Average temperature by shift
```

---

### 🔐 Security & Safety
- Rejects harmful SQL (DROP, UPDATE, TRUNCATE…)
- LLM **never executes queries**
- Sanitized planner + validator workflow

---

### 🔮 Future Enhancements
- Multi-DB support (PostgreSQL, MySQL, SQL Server)
- Real-time streaming + live charts
- Vector memory to learn user patterns
- RAG metadata for ambiguity reduction
- Role-based secured analytics

---

### ⭐ Like This Project?

> If you find it helpful, ⭐ **star the repo** and contribute!

---

### 💡 Maintained By
**Satyam Singh Baghel**  
Gen AI Engineer | LLM + Autonomous Agents

---
