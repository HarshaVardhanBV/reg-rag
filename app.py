import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Regulatory RAG Q&A",
    page_icon="📋",
    layout="centered"
)

st.title("📋 Regulatory RAG Q&A")
st.caption("Ask questions about ICH/FDA guidelines. Powered by Claude + ChromaDB.")

question = st.text_input(
    "Your question",
    placeholder="e.g. What are the criteria for a serious adverse event?"
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()

                st.markdown("### Answer")
                st.write(data["answer"])

                st.markdown("### Source chunks")
                for i, source in enumerate(data["sources"]):
                    with st.expander(f"Source {i+1}"):
                        st.write(source)
            else:
                st.error(f"API error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Is FastAPI running? → `uvicorn main:app --reload`")