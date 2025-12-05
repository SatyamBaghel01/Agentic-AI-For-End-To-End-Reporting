import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
import io
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import base64


from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate   # ✅ FIXED
from langchain_core.output_parsers import JsonOutputParser

# 🔁 Groq instead of OpenAI
from langchain_groq import ChatGroq

load_dotenv()


class GraphPlottingAgent:
    """
    Agent that analyzes user question, explanation, table data,
    and graph suggestion to generate visualizations.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=self.api_key,
            temperature=0.1
        )
        
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
            sns.set_palette("viridis")
        except:
            pass    


    def analyze_and_plot(
        self,
        user_question: str,
        explanation: str,
        table_data: Any,
        graph_suggestion: str
    ) -> Dict[str, Any]:

        try:
            df = self._prepare_dataframe(table_data)

            if df is None or df.empty:
                return {
                    "success": False,
                    "error": "No valid data available for plotting",
                    "plot_analysis": {"plot_type": "none"},
                    "plot_result": {"plot_created": False, "message": "No data"}
                }

            plot_analysis = self._analyze_plot_requirements(
                user_question, explanation, df, graph_suggestion
            )

            if "plot_type" not in plot_analysis:
                plot_analysis["plot_type"] = "none"

            plot_result = self._generate_plot(df, plot_analysis)

            return {
                "Graph": plot_result, 
                "Analysis": plot_analysis,
                "success": True,
                # "plot_analysis": plot_analysis,
                # "plot_result": plot_result,
                # "dataframe_info": {
                #     "shape": df.shape,
                #     "columns": list(df.columns),
                #     "sample_data": df.head(3).to_dict("records")
                # }
            }

        except Exception as e:
            print("Graph Agent Error:", e)
            return {
                "success": False,
                "error": str(e),
                "plot_analysis": {"plot_type": "error"},
                "plot_result": {"plot_created": False, "error": str(e)}
            }

    # ------------------ DataFrame Preparation ------------------

    def _prepare_dataframe(self, table_data: Any) -> Optional[pd.DataFrame]:
        if table_data == "NA" or table_data is None:
            return None

        if isinstance(table_data, list) and all(isinstance(row, dict) for row in table_data):
            df = pd.DataFrame(table_data)
        elif isinstance(table_data, dict):
            df = pd.DataFrame([table_data])
        elif isinstance(table_data, pd.DataFrame):
            df = table_data
        else:
            return None

        df.columns = df.columns.astype(str).str.strip()

        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        return df

    # ------------------ LLM Plot Analysis ------------------

    def _analyze_plot_requirements(self, user_question, explanation, df, graph_suggestion):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a visualization expert for industrial machine data.

Return JSON:
{{
 "plot_type": "bar|line|scatter|pie|histogram|box|area|heatmap|none",
 "x_column": null,
 "y_column": null,
 "group_by": null,
 "aggregation_needed": false,
 "aggregation_method": "",
 "title": "",
 "x_label": "",
 "y_label": ""
}}

Only use columns from this list:
{columns}
"""),
            ("human", """
User Question: {user_question}
Explanation: {explanation}
Graph Suggestion: {graph_suggestion}

Columns: {columns}
Data Types: {dtypes}
Sample: {sample_data}
""")
        ])

        try:
            chain = prompt | self.llm | JsonOutputParser()

            result = chain.invoke({
                "user_question": user_question[:200],
                "explanation": explanation[:300],
                "graph_suggestion": graph_suggestion[:200],
                "columns": list(df.columns),
                "dtypes": {},
                "sample_data": df.head(2).to_dict("records") 
            })
            print("\n--- PLOT ANALYSIS ---")
            print(result, "\n")
            return result

        except Exception as e:
            print("LLM Plot Analysis Failed:", e)
            return {
                "plot_type": "line",
                "x_column": next((c for c in df.columns if "time" in c.lower()), None),
                "y_column": (
                    df.select_dtypes(include=[np.number]).columns[0]
                    if len(df.select_dtypes(include=[np.number]).columns)
                    else None
                ),
                "title": "Fallback Plot",
                "x_label": "Time",
                "y_label": "Value"
            }
    
    # ------------------ Plot Creation ------------------

    def _generate_plot(self, df, plot_analysis):
        plot_type = plot_analysis.get("plot_type")
        print("\n--- DF BEING PLOTTED ---")
        print(df.head(), "\n")
        if plot_type == "none":
            return {"plot_created": False, "message": "No suitable plot"}

        try:
            fig = plt.figure(figsize=(10, 5))
            ax = plt.gca()

            x = plot_analysis.get("x_column")
            y = plot_analysis.get("y_column")

            # ----------------- PLOT TYPE HANDLING -----------------
            
            # --- FIX: validate x-axis ---
            if x not in df.columns:
                x = next((c for c in df.columns if "time" in c.lower()), None)

            # --- FIX: validate y-axis (numeric fallback) ---
            if y not in df.columns or df[y].dtype not in ["float64", "int64", "int32", "float32"]: 
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) == 0: 
                    return {"plot_created": False, "message": "No numeric column available"}
                y = numeric_cols[0]  

            if plot_type == "line":
                # Multi-series support if user didn't specify y_column clearly
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if x and x in df.columns:
                    for col in numeric_cols:
                        if col != x:
                            ax.plot(df[x], df[col], marker="o", label=col)
                    ax.legend()
                else:
                    df.reset_index()[numeric_cols].plot(ax=ax, marker="o")
                    ax.legend()

            elif plot_type == "bar":
                grouped = df.groupby(x)[y].mean().reset_index()
                sns.barplot(x=grouped[x], y=grouped[y], ax=ax)

            elif plot_type == "scatter":
                sns.scatterplot(x=df[x], y=df[y], ax=ax)

            elif plot_type == "histogram":
                sns.histplot(df[y], kde=True, ax=ax)

            elif plot_type == "box":
                sns.boxplot(x=df[x] if x else None, y=df[y], ax=ax)

            elif plot_type == "heatmap":
                sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="viridis", ax=ax)

            elif plot_type == "area":
                df.set_index(x)[y].plot.area(ax=ax)

            # -------- NEW 🥧 PIE CHART SUPPORT --------
            elif plot_type == "pie":
                numeric_cols = df.select_dtypes(include=["number"]).columns
                
                # Case 1: Both x & y available
                if x and x in df.columns and y and y in df.columns:
                    grouped = df.groupby(x)[y].sum().reset_index()
                    ax.pie(grouped[y], labels=grouped[x], autopct="%1.1f%%")

                # Case 2: Only numeric available (value counts pie)
                elif len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    value_counts = df[col].round(1).value_counts()
                    ax.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%")

                # Case 3: Fallback → No numeric? No pie possible
                else:
                    return {"plot_created": False, "message": "No numeric column to plot pie chart"}


            ax.set_title(plot_analysis.get("title"))
            ax.set_xlabel(plot_analysis.get("x_label"))
            ax.set_ylabel(plot_analysis.get("y_label"))

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            plt.close(fig)

            image_base64 = base64.b64encode(buffer.getvalue()).decode()

            return {
                "plot_created": True,
                "plot_type": plot_type,
                "image_base64": image_base64,
                "title": plot_analysis.get("title") or "Generated Visualization"
            }

        except Exception as e:
            return {
                "plot_created": False,
                 "message": str(e)
            }
