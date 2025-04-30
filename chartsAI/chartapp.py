import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import re
import os

OPENAI_API_KEY = "<API_KEY>"

import plotly.io as pio
pio.templates.default = "plotly_white"

def suggest_chart_type(df: pd.DataFrame) -> str:
    chat = ChatOpenAI(temperature=0.2, model="gpt-4", openai_api_key=OPENAI_API_KEY)

    preview = df.head(5).to_markdown()

    prompt = f"""
You are a data visualization assistant. Based on the following table preview, suggest the most suitable chart type from: bar, line, scatter, histogram, pie, area, box.

Only respond with the name of the most suitable chart type. Do not write any code.

Data preview:
{preview}
"""

    messages = [
        SystemMessage(content="You are a helpful data visualization assistant."),
        HumanMessage(content=prompt),
    ]

    result = chat(messages)
    chart_type = result.content.strip().lower()
    return chart_type

def main():
    st.title("📊 Smart Charts Plotter with AI-Assisted Model")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Let AI suggest chart type based on preview
        chart_type = suggest_chart_type(df)
        st.success(f"💡 AI suggests: {chart_type} chart")

        # User selects X and Y axes for plotting
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        all_cols = df.columns.tolist()

        x_axis = st.selectbox("Select X-axis", options=all_cols)
        y_axis = st.selectbox("Select Y-axis", options=numeric_cols)

        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        color_group = None
        if cat_cols:
            color_group = st.selectbox("Color grouping (optional)", options=[None] + cat_cols)

        st.subheader("📈 Generated Chart")

        # Generate chart based on AI suggestion
        fig = None

        try:
            if chart_type == "bar":
                fig = px.bar(df, x=x_axis, y=y_axis, color=color_group, title=f"{y_axis} vs {x_axis}")
            elif chart_type == "line":
                fig = px.line(df, x=x_axis, y=y_axis, color=color_group, title=f"{y_axis} over {x_axis}")
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_group, title=f"{y_axis} vs {x_axis}")
            elif chart_type == "histogram":
                fig = px.histogram(df, x=x_axis, color=color_group, title=f"Distribution of {x_axis}")
            elif chart_type == "pie":
                if color_group and y_axis:
                    fig = px.pie(df, names=color_group, values=y_axis, title=f"{y_axis} by {color_group}")
                else:
                    st.warning("Pie chart requires a categorical column for 'names' and a numeric column for values.")
            elif chart_type == "box":
                fig = px.box(df, x=x_axis, y=y_axis, color=color_group, title=f"Box plot of {y_axis} by {x_axis}")
            elif chart_type == "area":
                fig = px.area(df, x=x_axis, y=y_axis, color=color_group, title=f"{y_axis} over {x_axis}")
            else:
                st.warning("Unsupported chart type suggested.")
        except Exception as ex:
            st.error(f"Error generating chart: {ex}")

        if fig:
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
