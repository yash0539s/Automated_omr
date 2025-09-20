import streamlit as st
import requests
import pandas as pd
import sqlite3

st.title("Automated OMR Evaluation Dashboard")

# Upload OMR sheet
st.subheader("Upload Student OMR Sheet")
student_name = st.text_input("Student Name")
sheet_version = st.selectbox("Sheet Version", ["version_1", "version_2"])
uploaded_file = st.file_uploader("Choose OMR Sheet", type=["jpg","png"])

if st.button("Upload & Evaluate") and uploaded_file and student_name:
    response = requests.post("http://127.0.0.1:8000/upload/",
                             files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                             data={"student_name": student_name, "sheet_version": sheet_version})
    if response.status_code == 200:
        result = response.json()
        st.success(f"Evaluation Completed! Total Score: {result['total_score']}")
        st.write("Subject-wise Scores:", result["subject_scores"])
    else:
        st.error("Evaluation failed.")

# Aggregate stats
st.subheader("Aggregate Results")
if st.button("Show Aggregate"):
    conn = sqlite3.connect("omr_results.db")
    df = pd.read_sql_query("SELECT student_name, total_score FROM results", conn)
    st.dataframe(df)
    st.bar_chart(df.set_index("student_name")["total_score"])
