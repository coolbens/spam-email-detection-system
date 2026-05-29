from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from app.ui import (
        inject_custom_css,
        metric_card,
        render_bulk_table,
        render_keyword_pills,
        render_prediction_result,
        section_title,
    )
except ModuleNotFoundError:
    from ui import (  # type: ignore
        inject_custom_css,
        metric_card,
        render_bulk_table,
        render_keyword_pills,
        render_prediction_result,
        section_title,
    )

from model.predict import load_artifacts, predict_many, predict_text
from model.train import train_models
from nlp.preprocess import preprocess_tokens
from utils.helpers import ARTIFACT_DIR

st.set_page_config(
    page_title="Spam Email Detection System",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
COLORWAY = ["#8b5cf6", "#06b6d4", "#22c55e", "#f59e0b", "#ef4444", "#ec4899", "#3b82f6", "#14b8a6"]


def bootstrap_if_needed() -> None:
    metrics_path = ARTIFACT_DIR / "metrics.json"
    vectorizer_path = ARTIFACT_DIR / "vectorizer.pkl"
    models_dir = ARTIFACT_DIR / "models"
    if not metrics_path.exists() or not vectorizer_path.exists() or not models_dir.exists():
        with st.spinner("Training model artifacts for first launch..."):
            train_models()


def extract_texts_from_excel(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    preferred_cols = ["text", "message", "email", "content", "body"]
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for key in preferred_cols:
        if key in normalized:
            series = df[normalized[key]].dropna().astype(str).str.strip()
            return [value for value in series.tolist() if value]

    for col in df.columns:
        series = df[col].dropna().astype(str).str.strip()
        values = [value for value in series.tolist() if value]
        if values:
            return values
    return []


def build_bulk_insights(texts: list[str], results: list[dict]) -> dict:
    paired = []
    for i, (text, result) in enumerate(zip(texts, results), start=1):
        paired.append(
            {
                "row_number": i,
                "text": text,
                "prediction": result["label"],
                "confidence": float(result["confidence"]),
                "spam_probability": float(result["spam_probability"]),
                "ham_probability": float(result["ham_probability"]),
            }
        )

    df = pd.DataFrame(paired)
    spam_texts = df.loc[df["prediction"] == "spam", "text"].tolist() if not df.empty else []
    ham_texts = df.loc[df["prediction"] == "ham", "text"].tolist() if not df.empty else []

    def top_terms(messages: list[str], n: int = 10) -> list[tuple[str, int]]:
        counter = Counter()
        for message in messages:
            counter.update(preprocess_tokens(message))
        return counter.most_common(n)

    return {
        "dataframe": df,
        "summary": {
            "total_rows": int(len(df)),
            "spam_count": int((df["prediction"] == "spam").sum()) if not df.empty else 0,
            "ham_count": int((df["prediction"] == "ham").sum()) if not df.empty else 0,
            "avg_confidence": float(df["confidence"].mean()) if not df.empty else 0.0,
            "avg_spam_probability": float(df["spam_probability"].mean()) if not df.empty else 0.0,
            "avg_ham_probability": float(df["ham_probability"].mean()) if not df.empty else 0.0,
            "spam_terms": top_terms(spam_texts),
            "ham_terms": top_terms(ham_texts),
        },
    }


def styled_figure(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.75)",
        font=dict(color="#e2e8f0"),
        colorway=COLORWAY,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_bulk_graphs(insights: dict, key_prefix: str = "bulk") -> None:
    df = insights["dataframe"]
    summary = insights["summary"]

    c1, c2 = st.columns(2)
    with c1:
        counts_df = pd.DataFrame({
            "Prediction": ["Spam", "Ham"],
            "Count": [summary["spam_count"], summary["ham_count"]],
        })
        fig_counts = px.bar(
            counts_df,
            x="Prediction",
            y="Count",
            color="Prediction",
            title="Prediction distribution",
            text="Count",
            color_discrete_map={"Spam": "#ef4444", "Ham": "#22c55e"},
        )
        st.plotly_chart(styled_figure(fig_counts), use_container_width=True, key=f"{key_prefix}_prediction_distribution")

    with c2:
        donut_df = pd.DataFrame({
            "Class": ["Spam Probability", "Ham Probability"],
            "Average": [summary["avg_spam_probability"] * 100, summary["avg_ham_probability"] * 100],
        })
        fig_prob = px.pie(
            donut_df,
            names="Class",
            values="Average",
            hole=0.55,
            title="Average class probability",
            color="Class",
            color_discrete_map={"Spam Probability": "#f59e0b", "Ham Probability": "#06b6d4"},
        )
        st.plotly_chart(styled_figure(fig_prob), use_container_width=True, key=f"{key_prefix}_average_class_probability")

    if not df.empty:
        c3, c4 = st.columns(2)
        with c3:
            trend_df = df.copy()
            trend_df["confidence_percent"] = (trend_df["confidence"] * 100).round(2)
            fig_line = px.line(
                trend_df,
                x="row_number",
                y="confidence_percent",
                color="prediction",
                markers=True,
                title="Confidence by uploaded row",
                color_discrete_map={"spam": "#ec4899", "ham": "#3b82f6"},
            )
            fig_line.update_xaxes(title="Row")
            fig_line.update_yaxes(title="Confidence %")
            st.plotly_chart(styled_figure(fig_line), use_container_width=True, key=f"{key_prefix}_confidence_by_row")

        with c4:
            fig_scatter = px.scatter(
                df,
                x="spam_probability",
                y="ham_probability",
                color="prediction",
                size="confidence",
                hover_data=["row_number"],
                title="Spam vs ham probability spread",
                color_discrete_map={"spam": "#a855f7", "ham": "#14b8a6"},
            )
            fig_scatter.update_xaxes(title="Spam probability")
            fig_scatter.update_yaxes(title="Ham probability")
            st.plotly_chart(styled_figure(fig_scatter), use_container_width=True, key=f"{key_prefix}_probability_spread")


def render_term_bar(title: str, terms: list[tuple[str, int]], color: str, key: str | None = None) -> None:
    if not terms:
        return
    terms_df = pd.DataFrame(terms, columns=["term", "count"])
    fig = px.bar(terms_df, x="count", y="term", orientation="h", text="count", title=title)
    fig.update_traces(marker_color=color)
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(styled_figure(fig), use_container_width=True, key=key or title.lower().replace(" ", "_"))


def main() -> None:
    bootstrap_if_needed()
    _, _, metrics = load_artifacts()
    available_models = metrics.get("available_models") or list(metrics.get("results", {}).keys()) or [metrics.get("best_model", "naive_bayes")]

    if "bulk_insights" not in st.session_state:
        st.session_state.bulk_insights = None
    if "active_bulk_model" not in st.session_state:
        st.session_state.active_bulk_model = None

    st.sidebar.markdown("## Controls")
    selected_model = st.sidebar.selectbox(
        "Choose model",
        options=available_models,
        index=available_models.index(metrics.get("best_model")) if metrics.get("best_model") in available_models else 0,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    selected_metrics = metrics.get("results", {}).get(selected_model, {}).get("metrics", {})
    st.sidebar.caption("Switch models to compare predictions and evaluation metrics.")
  

    section_title(
        "Spam Email Detection System","AI-Powered NLP System for Intelligent Spam Email Classification"
        
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Selected model", selected_model.replace("_", " ").title(), "Use the sidebar dropdown to switch models")
    with col2:
        metric_card("Accuracy", f"{selected_metrics.get('accuracy', 0):.2%}", "Evaluation on test split")
    with col3:
        metric_card("F1 Score", f"{selected_metrics.get('f1_score', 0):.2%}", "Spam class performance")

    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Bulk Prediction", "Model Insights"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        message = st.text_area(
            "Enter email or message text",
            height=180,
            placeholder="Example: Congratulations! You won a free gift card. Click here to claim now...",
        )
        if st.button("Predict", type="primary", use_container_width=True):
            if not message.strip():
                st.warning("Please enter a message first.")
            else:
                result = predict_text(message, model_name=selected_model)
                result["highlights"] = []
                render_prediction_result(result, show_highlights=False)
        st.caption("Single prediction keeps model insights at zero. Upload an Excel file in Bulk Prediction to unlock the insights graphs.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Upload an Excel file (.xlsx or .xls). Best practice: use a column named text, message, email, content, or body.")
        uploaded_file = st.file_uploader("Upload Excel file for bulk prediction", type=["xlsx", "xls"])
        if uploaded_file is not None:
            try:
                input_df = pd.read_excel(uploaded_file)
                lines = extract_texts_from_excel(input_df)
                if lines:
                    results = predict_many(lines, model_name=selected_model)
                    insights = build_bulk_insights(lines, results)
                    st.session_state.bulk_insights = insights
                    st.session_state.active_bulk_model = selected_model
                    df = render_bulk_table(results, lines)
                    output = Path(ARTIFACT_DIR / "bulk_predictions.xlsx")
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False, sheet_name="predictions")
                    st.download_button(
                        "Download predictions as Excel",
                        data=output.read_bytes(),
                        file_name="bulk_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                    
                else:
                    st.info("No usable text rows were found in the Excel file.")
            except Exception as exc:
                st.error(f"Could not read the Excel file: {exc}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Bulk-only model insights**")
        insights = st.session_state.bulk_insights
        if not insights or st.session_state.active_bulk_model != selected_model:
            z1, z2, z3, z4 = st.columns(4)
            with z1:
                metric_card("Uploaded Rows", "0", "Upload bulk Excel to unlock insights")
            with z2:
                metric_card("Spam Rows", "0", "No bulk file uploaded yet")
            with z3:
                metric_card("Ham Rows", "0", "No bulk file uploaded yet")
            with z4:
                metric_card("Average Confidence", "0%", "No bulk file uploaded yet")
            if st.session_state.active_bulk_model and st.session_state.active_bulk_model != selected_model:
                st.info("Upload the bulk Excel again after switching the model so the insights match the selected model.")
            else:
                st.info("Model insights stay at zero until you upload a bulk Excel file in the Bulk Prediction tab.")
            fig_zero = go.Figure(data=[go.Bar(x=["Spam", "Ham"], y=[0, 0], marker_color=["#ef4444", "#22c55e"])])
            fig_zero.update_layout(title="Prediction distribution")
            st.plotly_chart(styled_figure(fig_zero), use_container_width=True, key="insights_zero_prediction_distribution")
            render_keyword_pills("Top spam keywords", [], positive=True)
            render_keyword_pills("Top ham keywords", [], positive=False)
        else:
            summary = insights["summary"]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Uploaded Rows", str(summary["total_rows"]), "Rows analyzed from Excel")
            with c2:
                metric_card("Spam Rows", str(summary["spam_count"]), "Predicted as spam")
            with c3:
                metric_card("Ham Rows", str(summary["ham_count"]), "Predicted as not spam")
            with c4:
                metric_card("Average Confidence", f"{summary['avg_confidence']:.2%}", "Across all uploaded rows")

            render_bulk_graphs(insights, key_prefix="insights_tab")
            spam_terms = [term for term, _ in summary["spam_terms"]]
            ham_terms = [term for term, _ in summary["ham_terms"]]
            col_spam, col_ham = st.columns(2)
            with col_spam:
                render_keyword_pills("Top spam keywords", spam_terms, positive=True)
                render_term_bar("Top spam keywords frequency", summary["spam_terms"], "#ef4444", key="insights_spam_term_frequency")
            with col_ham:
                render_keyword_pills("Top ham keywords", ham_terms, positive=False)
                render_term_bar("Top ham keywords frequency", summary["ham_terms"], "#22c55e", key="insights_ham_term_frequency")
        st.markdown("</div>", unsafe_allow_html=True)

    


if __name__ == "__main__":
    main()
