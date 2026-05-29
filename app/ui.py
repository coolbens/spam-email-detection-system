# Aron mosupport sa modern Python type hints ug mas compatible sa lain-laing Python versions.
from __future__ import annotations

# Gigamit para sa type hints.
# Dict = dictionary
# List = list
from typing import Dict, List

# Pandas library para sa paghimo ug pagdumala sa DataFrame (tables).
import pandas as pd

# Streamlit library para sa web application UI.
import streamlit as st


# Nagbutang ug custom CSS aron nindot tan-awon ang Streamlit app.
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* Main background sa application */
        .stApp {
            background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
            color: #e2e8f0;
        }

        /* Main container sa page */
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Reusable card design */
        .card {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            padding: 1.2rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        }

        /* General badge style */
        .badge {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
        }

        /* Spam badge (pula) */
        .badge-spam {
            background: rgba(239, 68, 68, 0.15);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }

        /* Ham / Not Spam badge (berde) */
        .badge-ham {
            background: rgba(34, 197, 94, 0.14);
            color: #86efac;
            border: 1px solid rgba(34, 197, 94, 0.35);
        }

        /* Gagmay nga muted text */
        .small-muted {
            color: #94a3b8;
            font-size: 0.92rem;
        }

        /* Main title */
        .title-hero {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }

        /* Subtitle */
        .subtitle-hero {
            color: #cbd5e1;
            font-size: 1rem;
            margin-bottom: 1.1rem;
        }

        /* Style sa keyword pills */
        .word-pill-spam, .word-pill-ham {
            display:inline-block;
            margin: 0.2rem 0.25rem 0.2rem 0;
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        /* Spam keyword pill */
        .word-pill-spam {
            background: rgba(239,68,68,0.15);
            color: #fda4af;
        }

        /* Ham keyword pill */
        .word-pill-ham {
            background: rgba(34,197,94,0.15);
            color: #86efac;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Nagpakita ug main title ug optional subtitle.
def section_title(title: str, subtitle: str = "") -> None:

    # I-display ang main title.
    st.markdown(f'<div class="title-hero">{title}</div>', unsafe_allow_html=True)

    # Kung naay subtitle, ipakita usab.
    if subtitle:
        st.markdown(
            f'<div class="subtitle-hero">{subtitle}</div>',
            unsafe_allow_html=True
        )


# Reusable card para sa metrics/statistics.
def metric_card(title: str, value: str, hint: str = "") -> None:

    # Nag-render ug custom HTML card.
    st.markdown(
        f"""
        <div class="card">
            <div class="small-muted">{title}</div>
            <div style="font-size:1.65rem;font-weight:800;margin-top:0.35rem;">{value}</div>
            <div class="small-muted" style="margin-top:0.35rem;">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Nagbalik ug HTML badge depende sa prediction.
def result_badge(label: str) -> str:

    # Kung spam ang prediction.
    if label.lower() == "spam":
        return '<span class="badge badge-spam">Spam</span>'

    # Kung dili spam.
    return '<span class="badge badge-ham">Not Spam</span>'


# Nag-display sa result sa usa ka prediction.
def render_prediction_result(
    result: Dict,
    show_highlights: bool = True
) -> None:

    # Sugod sa result card.
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Prediction label ug badge.
    st.markdown(
        f"### Prediction {result_badge(result['label'])}",
        unsafe_allow_html=True
    )

    # Confidence score.
    st.write(f"**Confidence:** {result['confidence']:.2%}")

    # Probability nga spam.
    st.write(f"**Spam Probability:** {result['spam_probability']:.2%}")

    # Probability nga ham/not spam.
    st.write(f"**Ham Probability:** {result['ham_probability']:.2%}")


# Naghimo ug DataFrame para sa bulk predictions.
def render_bulk_table(
    results: List[Dict],
    texts: List[str]
) -> pd.DataFrame:

    # Himo ug DataFrame gikan sa prediction results.
    df = pd.DataFrame(
        {
            # Original nga text.
            "text": texts,

            # Prediction label.
            "prediction": [r["label"] for r in results],

            # Confidence percentage.
            "confidence": [
                round(r["confidence"] * 100, 2)
                for r in results
            ],

            # Spam probability percentage.
            "spam_probability": [
                round(r["spam_probability"] * 100, 2)
                for r in results
            ],

            # Ham probability percentage.
            "ham_probability": [
                round(r["ham_probability"] * 100, 2)
                for r in results
            ],
        }
    )

    # I-display ang table.
    st.dataframe(df, use_container_width=True)

    # Ibalik ang DataFrame aron magamit pa sa ubang functions.
    return df


# Nag-display ug keyword pills.
def render_keyword_pills(
    title: str,
    words: List[str],
    positive: bool = True
) -> None:

    # Section title.
    st.write(f"**{title}**")

    # Kung walay words.
    if not words:
        st.caption("0")
        return

    # Pilion kung spam style ba o ham style.
    css = "word-pill-spam" if positive else "word-pill-ham"

    # Himoon nga HTML pill ang matag keyword.
    html = "".join(
        [f'<span class="{css}">{w}</span>' for w in words]
    )

    # I-display ang pills.
    st.markdown(html, unsafe_allow_html=True)