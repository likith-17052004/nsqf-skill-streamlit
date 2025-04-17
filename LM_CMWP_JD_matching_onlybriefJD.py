import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import pandas as pd
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import ast

# --- Setup ---
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
COLLECTION_NAME = "JD_without_title"

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@st.cache_data
def load_job_data():
    df = pd.read_json("JD_without_title_embeddings.json")
    if 'embedding' in df.columns:
        df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

def search_similar_jobs_from_id(input_job_id, job_df, top_k):
    try:
        input_vector = job_df.loc[job_df['id'] == input_job_id, 'embedding'].values[0]
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=input_vector,
            limit=top_k + 1
        )
        return [r for r in results if r.id != input_job_id][:top_k]
    except Exception as e:
        st.error(f"Error in search: {e}")
        return []

st.set_page_config(layout="wide")
st.title("🔍 Job-to-Job Similarity Search(Based on Brief JD)")

job_df = load_job_data()
sectors = sorted(job_df['sector'].dropna().unique().tolist())
sub_sectors = sorted(job_df['sub_sector'].dropna().unique().tolist())
roles = sorted(job_df['occupation_role'].dropna().unique().tolist())

left_column, right_column = st.columns([0.5, 0.5])

with left_column:
    filtered_df = job_df.copy()
    filtered_df['title_display'] = filtered_df.apply(lambda x: f"{x['id']} - {x.get('job_title', 'N/A')} ({x.get('occupation_role', 'N/A')})", axis=1)

    st.markdown("### Select Input Job")
    selected_job_display = st.selectbox("Input Job:", options=filtered_df['title_display'].tolist())
    selected_job_id = int(selected_job_display.split(' - ')[0])
    top_k = st.slider("Number of matched jobs:", 1, 20, 5)
    match_triggered = st.button("🔍 Search")

    input_job = job_df[job_df['id'] == selected_job_id].iloc[0]
    st.markdown("## 📅 Input Job")
    st.markdown(f"**Title:** {input_job.get('job_title', 'N/A')}")
    st.markdown(f"**Occupation:** {input_job.get('occupation_role', 'N/A')}")
    st.markdown(f"**Sector:** {input_job.get('sector', 'N/A')} | Sub-sector: {input_job.get('sub_sector', 'N/A')}")
    st.markdown(f"**Description:** {input_job.get('job_description', 'N/A')}")

with right_column:
    st.markdown("### Include Filters")
    include_col1, include_col2, include_col3 = st.columns([1, 1, 1])
    with include_col1:
        selected_sector = st.selectbox("Sector", options=["All"] + sectors)
    with include_col2:
        selected_sub_sector = st.selectbox("Sub-sector", options=["All"] + sub_sectors)
    with include_col3:
        selected_role = st.selectbox("Occupation Role", options=["All"] + roles)

    st.markdown("### Exclude Filters")
    exclude_col1, exclude_col2, exclude_col3 = st.columns([1, 1, 1])
    with exclude_col1:
        excluded_sector = st.selectbox("Exclude Sector", options=["None"] + sectors)
    with exclude_col2:
        excluded_sub_sector = st.selectbox("Exclude Sub-sector", options=["None"] + sub_sectors)
    with exclude_col3:
        excluded_role = st.selectbox("Exclude Role", options=["None"] + roles)

    if 'match_triggered' not in st.session_state:
        st.session_state.match_triggered = False
    if match_triggered:
        st.session_state.match_triggered = True

    if st.session_state.match_triggered:
        st.markdown("### 🔍 Matched Jobs")
        with st.spinner("Searching for similar jobs from existing corpus..."):
            temp_results = search_similar_jobs_from_id(selected_job_id, job_df, top_k)
            results = []
            for r in temp_results:
                job = r.payload
                if selected_sector != "All" and job.get('sector') != selected_sector:
                    continue
                if selected_sub_sector != "All" and job.get('sub_sector') != selected_sub_sector:
                    continue
                if selected_role != "All" and job.get('occupation_role') != selected_role:
                    continue
                if excluded_sector != "None" and job.get('sector') == excluded_sector:
                    continue
                if excluded_sub_sector != "None" and job.get('sub_sector') == excluded_sub_sector:
                    continue
                if excluded_role != "None" and job.get('occupation_role') == excluded_role:
                    continue
                results.append(r)
            if not results:
                st.warning("No matches found.")
            else:
                matched_df = pd.DataFrame([r.payload | {'score': round(r.score * 100, 2)} for r in results])
                for i, row in matched_df.iterrows():
                    with st.expander(f"{i+1}. {row.get('job_title', 'N/A')} (Score: {row.get('score', 'N/A')})"):
                        st.markdown(f"<div style='font-size: 1.2rem'><b>Occupation:</b> {row.get('occupation_role', 'N/A')}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 1.2rem'><b>Sector:</b> {row.get('sector', 'N/A')} | Sub-sector: {row.get('sub_sector', 'N/A')}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 1.1rem'><b>Description:</b> {row.get('job_description', 'N/A')}</div>", unsafe_allow_html=True)

st.markdown("""
<style>
.footer {
    position: fixed;
    right: 10px;
    bottom: 10px;
    text-align: right;
    font-size: 13px;
    color: grey;
}
.stExpander > div:first-child {
    font-size: 1.3rem !important;
    font-weight: 600;
}
.stExpander p {
    font-size: 1.2rem !important;
}

.stExpander > div:first-child button {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
}
</style>
<div class="footer">Developed by Technology and AI team<br>CEEW</div>
""", unsafe_allow_html=True)
