import streamlit as st
from Mcp_Action import *
 
st.set_page_config(page_title="🔥 Firecrawl Quiz Generator", layout="centered")
st.title("🌐 Web-Based Intelligent Quiz Generator")
 
# Initial state
if "step" not in st.session_state:
    st.session_state.step = "input"
 
# Step 1: URL input
if st.session_state.step == "input":
    st.markdown("Enter one or more comma-separated URLs to generate a quiz from the content.")
    urls_input = st.text_area("🔗 URLs (comma-separated):")
    num_q = st.number_input("🧠 Number of questions", min_value=1, max_value=20, value=5)
 
    if st.button("🚀 Generate Quiz"):
        urls = [u.strip() for u in urls_input.split(",") if u.strip()]
        if not urls:
            st.warning("Please enter at least one valid URL.")
        else:
            with st.spinner("🔍 Scraping websites..."):
                content = scrape_multiple(urls)
            try:
                with st.spinner("🤖 Generating quiz using LLM..."):
                    quiz = call_llm_generate(content, num_questions=num_q)
                    st.session_state.quiz = quiz
                    st.session_state.step = "quiz"
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to generate quiz: {e}")
 
# Step 2: Show Quiz
elif st.session_state.step == "quiz":
    st.subheader("📋 Quiz Questions")
    quiz = st.session_state.quiz
 
    for idx, q in enumerate(quiz, start=1):
        st.markdown(f"### Q{idx}: ({q['type']})\n**{q['question']}**")
 
        if q["type"] == "MCQ":
            st.radio("Choose:", q["options"], key=f"mcq_{idx}")
        elif q["type"] == "ShortAnswer":
            st.text_input("Answer:", key=f"short_{idx}")
        elif q["type"] == "Coding":
            st.text_area("Write Code:", key=f"code_{idx}")
            if "test_cases" in q:
                st.json(q["test_cases"])
 
        st.markdown("---")
 
    if st.button("🔁 Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()