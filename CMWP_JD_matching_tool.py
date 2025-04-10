import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import google.generativeai as genai
import pandas as pd
import time
import matplotlib.pyplot as plt

SECRET_QDRANT_URL = st.secrets["qdrant"]["url"]
SECRET_QDRANT_API_KEY = st.secrets["qdrant"]["api_key"]
GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
# --- Setup ---
QDRANT_URL = "SECRET_QDRANT_URL"
QDRANT_API_KEY = "SECRET_QDRANT_API_KEY"
COLLECTION_NAME = "JD_without_title"

# --- Configure APIs ---
genai.configure(api_key='GEMINI_API_KEY')
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- Gemini embedding ---
def gemini_embedding(text, max_retries=5, wait_seconds=5):
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="SEMANTIC_SIMILARITY"
            )
            return result['embedding']
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                st.error(f"Error generating embedding after {max_retries} attempts: {str(e)}")
                return None

# --- Search Function ---
def search_similar_jobs_with_filters(query_text, top_k=3, include_sector=None, exclude_sector=None,
                                     include_sub_sector=None, include_occupation_role=None):
    query_vector = gemini_embedding(query_text)
    if query_vector is None:
        return []

    must_filters = []
    must_not_filters = []

    if include_sector:
        must_filters.append(rest.FieldCondition(key="sector", match=rest.MatchValue(value=include_sector)))
    if include_sub_sector:
        must_filters.append(rest.FieldCondition(key="sub_sector", match=rest.MatchValue(value=include_sub_sector)))
    if include_occupation_role:
        must_filters.append(rest.FieldCondition(key="occupation_role", match=rest.MatchValue(value=include_occupation_role)))
    if exclude_sector:
        must_not_filters.append(rest.FieldCondition(key="sector", match=rest.MatchValue(value=exclude_sector)))

    metadata_filter = rest.Filter(must=must_filters, must_not=must_not_filters) if must_filters or must_not_filters else None

    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=metadata_filter,
        limit=top_k
    )
    return results

# --- Load dropdown options (assumes you have a JSON with full dataset) ---
@st.cache_data
def load_dropdown_data():
    df = pd.read_json("JD_without_title_embeddings.json")
    sectors = sorted(df['sector'].dropna().unique().tolist())
    sub_sectors = sorted(df['sub_sector'].dropna().unique().tolist())
    roles = sorted(df['occupation_role'].dropna().unique().tolist())
    return sectors, sub_sectors, roles

# --- Streamlit UI ---
st.title("🔍 Job Similarity Search")

query_text = st.text_area("Enter a job-related query:", "Manage a vehicle repair center and interact with customers")

# Load dropdown options
sectors, sub_sectors, roles = load_dropdown_data()

col1, col2 = st.columns(2)
with col1:
    include_sector = st.selectbox("Include Sector (optional):", options=["None"] + sectors)
    include_sub_sector = st.selectbox("Include Sub-sector (optional):", options=["None"] + sub_sectors)
with col2:
    exclude_sector = st.selectbox("Exclude Sector (optional):", options=["None"] + sectors)
    include_occupation_role = st.selectbox("Include Occupation Role (optional):", options=["None"] + roles)

top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

if st.button("🔍 Search"):
    with st.spinner("Searching Qdrant for similar jobs..."):
        try:
            results = search_similar_jobs_with_filters(
                query_text=query_text,
                top_k=top_k,
                include_sector=None if include_sector == "None" else include_sector,
                exclude_sector=None if exclude_sector == "None" else exclude_sector,
                include_sub_sector=None if include_sub_sector == "None" else include_sub_sector,
                include_occupation_role=None if include_occupation_role == "None" else include_occupation_role
            )
            if not results:
                st.warning("No matches found.")
            else:
                for i, hit in enumerate(results, start=1):
                    score = round(hit.score * 100, 2)
                    st.markdown(f"### {i}. {hit.payload.get('job_title', 'N/A')} (Score: {score})")
                    st.markdown(f"**Occupation:** {hit.payload.get('occupation_role', 'N/A')}")
                    st.markdown(f"**Sector:** {hit.payload.get('sector', 'N/A')} | **Sub-sector:** {hit.payload.get('sub_sector', 'N/A')}")
                    st.markdown(f"**Description:** {hit.payload.get('job_description', 'N/A')}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
