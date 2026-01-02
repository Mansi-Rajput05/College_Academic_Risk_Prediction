# =========================
# IMPORTS (ONLY IMPORTS ABOVE set_page_config)
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================
# PATH SETUP (CRITICAL FOR STREAMLIT CLOUD)
# =========================
ROOT_DIR = Path(__file__).resolve().parent.parent


# =========================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# =========================
st.set_page_config(
    page_title="College Academic Risk Predictor",
    layout="centered"
)


# =========================
# TITLE & INTRO
# =========================
st.title("🎓 College Academic Risk Prediction System")
st.write(
    """
    This application predicts a college student's **final academic score**
    and identifies their **academic risk level** using a machine learning model.
    """
)

st.divider()


# =========================
# LOAD DATA & MODELS
# =========================
@st.cache_resource
def load_models():
    reg_model = joblib.load(
        ROOT_DIR / "models" / "final_score_model.pkl"
    )
    cls_model = joblib.load(
        ROOT_DIR / "models" / "risk_classifier.pkl"
    )
    return reg_model, cls_model


@st.cache_data
def load_data():
    return pd.read_csv(
        ROOT_DIR / "data" / "processed" / "student_featured.csv"
    )


try:
    reg_model, cls_model = load_models()
    df = load_data()
except Exception as e:
    st.error("❌ Error loading models or data")
    st.exception(e)
    st.stop()


# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("📝 Student Academic Inputs")

attendance = st.sidebar.slider("Attendance Percentage (%)", 40, 100, 75)
study_hours = st.sidebar.slider("Daily Self Study Hours", 1, 10, 3)
internal_1 = st.sidebar.slider("Internal Marks 1", 0, 20, 12)
internal_2 = st.sidebar.slider("Internal Marks 2", 0, 20, 13)
backlogs = st.sidebar.selectbox("Backlog Count", [0, 1, 2, 3])


# =========================
# FEATURE ENGINEERING (SAME AS TRAINING)
# =========================
performance_index = attendance * internal_2
consistency_score = abs(internal_2 - internal_1)

input_df = pd.DataFrame(
    [[
        attendance,
        study_hours,
        internal_1,
        internal_2,
        backlogs,
        performance_index,
        consistency_score
    ]],
    columns=[
        "attendance_percentage",
        "self_study_hours",
        "internal_marks_1",
        "internal_marks_2",
        "backlog_count",
        "performance_index",
        "consistency_score"
    ]
)


# =========================
# PREDICTION
# =========================
predicted_score = reg_model.predict(input_df)[0]
risk_prediction = cls_model.predict(input_df)[0]


# =========================
# RESULTS DISPLAY
# =========================
st.subheader("📊 Prediction Results")

col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Final Score", f"{predicted_score:.2f} / 20")

with col2:
    if risk_prediction == "At Risk":
        st.error("🔴 At Risk")
    elif risk_prediction == "Average":
        st.warning("🟡 Average")
    else:
        st.success("🟢 Top Performer")


# =========================
# EXPLANATION SECTION
# =========================
st.subheader("🔍 Explanation")

st.write(
    f"""
    **Key contributing factors:**
    - Attendance: **{attendance}%**
    - Internal Performance: **{internal_1}, {internal_2}**
    - Study Effort: **{study_hours} hrs/day**
    - Backlogs: **{backlogs}**
    """
)


# =========================
# RECOMMENDATIONS
# =========================
st.subheader("✅ Recommendations")

if risk_prediction == "At Risk":
    st.write(
        """
        - Increase attendance to **80%+**
        - Maintain consistency in internal exams
        - Reduce backlog count
        - Increase daily self-study time
        """
    )
elif risk_prediction == "Average":
    st.write(
        """
        - Improve internal marks
        - Maintain attendance above **75%**
        - Focus on weak subjects
        """
    )
else:
    st.write(
        """
        - Maintain current performance
        - Consider mentoring peers
        - Prepare for advanced academic opportunities
        """
    )


# =========================
# FOOTER
# =========================
st.divider()
st.caption("© 2025 | College Academic Risk Prediction System | Developed by Mansi Rajput")
