import streamlit as st
import streamlit as st
from Mcp_Action import *
import streamlit as st
import time
import json
from Actions import *
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from collections import Counter
from streamlit_ace import st_ace
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
# Initialize session state
if "role" not in st.session_state:
    st.session_state.role = None

def load_css():
    """Load global stylesheet for the app."""
    try:
        with open("assets/styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fail silently if styles are not present
        pass

# === Role Selection Page ===
if st.session_state.role is None:
    st.set_page_config(page_title="Intelligent Evaluator", layout="centered")
    load_css()
    st.title("Welcome to the Intelligent Evaluation Platform")

    st.subheader("Please select your role:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("I'm a Student"):
            st.session_state.role = "student"
            st.rerun()
    with col2:
        if st.button("I'm an SME"):
            st.session_state.role = "sme"
            st.rerun()

if st.session_state.role == "student":
    #--------------------------------------------------------------------------------------------------------
    # === LLM Client Setup ===
    load_dotenv()
    hf_token = os.getenv("hf_token") or st.secrets["hf_token"]
    client = InferenceClient(provider="fireworks-ai", api_key=hf_token)

    # === UI Setup ===
    st.set_page_config(page_title="Intelligent Evaluator", layout="centered")
    load_css()
    st.title("Intelligent Evaluator (LLM-assisted Flow)")

    # === Session Initialization ===
    if "topic" not in st.session_state:
        st.session_state.topic = ""
    if "tags" not in st.session_state:
        st.session_state.tags = []
    if "beliefs" not in st.session_state:
        st.session_state.beliefs = {}
    if "question" not in st.session_state:
        st.session_state.question = {}
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "max_questions" not in st.session_state:
        st.session_state.max_questions = 10
    if "step" not in st.session_state:
        st.session_state.step = "start"
    if "question_counts" not in st.session_state:
        st.session_state.question_counts = {}

    def end_test():
        st.session_state.step = "summarize"

    # === LLM Helper ===
    def call_llm_for_next_question(tags, beliefs, asked_types):
        type_counts = Counter(asked_types)
        total_asked = len(asked_types)
        max_questions = st.session_state.max_questions
        mcq_count = type_counts.get("MCQ", 0)
        short_answer_count = type_counts.get("ShortAnswer", 0)
        coding_count = type_counts.get("Coding", 0)
        print(mcq_count,short_answer_count,coding_count)
        system_prompt = f"""
    You are an intelligent evaluator tasked with generating a high-quality question to assess a student's understanding of a technical topic (e.g., Python).
    
    Use the following constraints:
    - Use only the provided list of tags.
    - Choose one or more tags that have not yet been assessed.
    - Use the "difficulty" field to adapt based on belief strength: start easier for unknown topics, or increase difficulty if belief is high.
    - Create only one set of tags for a single question
    
    Important Distribution Rule:
    - This test will consist of a total of {max_questions} questions.
    - Questions should be distributed as:
        • 50% MCQ →  questions
        • 20% ShortAnswer →  questions
        • 20% Coding →  questions
    - Total questions asked so far: {total_asked}
    - Already asked: {mcq_count} MCQ, {short_answer_count} ShortAnswer, {coding_count} Coding
    
    You must return your next question in strict JSON format using the following structure:
    {{
    "tags": ["list", "of", "tags"],
    "type": "MCQ" | "ShortAnswer" | "Coding",
    "difficulty": "easy" | "medium" | "hard"
    }}
    
    Only return the JSON object. Do not include any commentary, explanation, or markdown formatting.
    """.strip()

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": json.dumps({
                "tags": tags,
                "beliefs": beliefs,
                "asked_types": asked_types
            })}
        ]
        
        try:
            response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=messages
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            st.error(f"Error in generating the question please restart the test.")
            return {}

    # === Step 1: Enter Topic ===
    if st.session_state.step == "start":
        topic = st.text_input("Enter topic to evaluate:", value="Python")
        if st.button("Start Test"):
            try:
                result = generate_tags(topic)
                st.session_state.topic = topic
                st.session_state.tags = result.get("tags", [])
                st.session_state.beliefs = result.get("beliefs", {})
                st.session_state.asked_types = []
                st.session_state.step = "next_question"
                st.rerun()
            except Exception as e:
                print(f"Error generating tags: {e}")
                st.error(f"Error in generating the question please restart the test.")

    # === Step 2: LLM picks tag/type → generate question ===
    elif st.session_state.step == "next_question":
        if st.session_state.question_count >= st.session_state.max_questions:
            st.session_state.step = "summarize"
            st.rerun()
        else:
            decision = call_llm_for_next_question(
                tags=st.session_state.tags,
                beliefs=st.session_state.beliefs,
                asked_types=st.session_state.get("asked_types", [])
            )
            if decision:
                try:
                    q = generate_question(
                        tag=decision["tags"],
                        type=decision["type"],
                        difficulty=decision["difficulty"]
                    )
                    print(q)
                    st.session_state.question = q
                    st.session_state.current_tag = decision["tags"]
                    st.session_state.asked_types.append(decision["type"])
                    st.session_state.step = "show_question"
                    st.rerun()
                except Exception as e:
                    print(f"Error generating question: {e}")
                    st.error(f"Error in generating the question please restart the test.")
            else:
                print("LLM failed to suggest a next question.")
                st.error(f"Error in generating the question please restart the test.")

# === Step 3: Show Question and Capture Answer ===
    # === Step 3: Show Question and Capture Answer ===
    elif st.session_state.step == "show_question":
        q = st.session_state.question
        st.subheader(f"Question {st.session_state.question_count + 1}")
        st.markdown(f"**{q['question']}**")

        # === Timer Setup ===
        if "question_start_time" not in st.session_state:
            st.session_state.question_start_time = time.time()

        question_duration = q["time_limit"]
        elapsed = int(time.time() - st.session_state.question_start_time)
        remaining = max(question_duration - elapsed, 0)

        # === JavaScript Countdown Timer (displays but logic is backend-controlled) ===
        import streamlit.components.v1 as components
        components.html(f"""
            <div id="timer" style="font-size:20px; color:#336699; margin-bottom: 10px;"></div>
            <script>
            let countdown = {remaining};
            let timerElement = document.getElementById("timer");

            function updateTimer() {{
                let minutes = Math.floor(countdown / 60);
                let seconds = countdown % 60;
                timerElement.innerHTML = "Time Remaining: " + 
                String(minutes).padStart(2, '0') + ":" + 
                String(seconds).padStart(2, '0');
                countdown--;
                if (countdown < 0) {{
                timerElement.innerHTML = "Time is up!";
                clearInterval(timer);
                }}
            }}
            updateTimer();
            let timer = setInterval(updateTimer, 1000);
            </script>
        """, height=50)

        # === Logic Control for Disabling Inputs ===
        time_up = remaining <= 0

        # === Input Setup ===
        user_answer = None

        # Check if the flag is set to clear the input fields
        if st.session_state.get("flag", True):
            # if q["type"] == "MCQ":
            #     st.session_state["mcq_answer"] = ""
            if q["type"] == "ShortAnswer":
                st.session_state["short_answer"] = ""
            elif q["type"] == "Coding":
                st.session_state["coding_answer"] = ""
            st.session_state.flag = False  # Reset the flag

            # Clear stale answer if it's incompatible
        user_answer = None
        if q["type"] == "MCQ":
            user_answer = st.selectbox("Choose your answer:", q["options"], key="mcq_answer")

        elif q["type"] == "ShortAnswer":
            user_answer = st.text_input("Enter your answer:", key="short_answer")

        elif q["type"] == "Coding":
            st.markdown("**Please write your code inside a function named `solution` and return the expected result.**")
            st.markdown("Example:")
            st.code("def solution(...):\n    # your logic here\n    return result", language="python")
            user_answer = st_ace(
            value=st.session_state["coding_answer"],
            language="python",  
            theme="monokai",
            key="coding_answer",
            height=200,
            auto_update=True,
        )
        
        st.session_state.flag = True

    # === Submission Buttons ===
        col1, col2, col3 = st.columns([1, 1, 1])
        submitted = col1.button("Submit Answer", disabled=time_up)
        skipped = col2.button("Skip Question")
        End_test = col3.button("End Test")
        if End_test:
            end_test()
        # if st.button("End Test", on_click=end_test, key="end_test"):
        #     st.rerun()


        if time_up:
            st.warning("Time is up! You can only skip this question.")

        if skipped:
            st.session_state.beliefs = update_beliefs(tags=st.session_state.current_tag, score=0.0)
            st.success("Question skipped. Moving to the next one.")
            st.session_state.question_count += 1
            st.session_state.flag = True
            st.session_state.step = "next_question"
            st.session_state.pop("question_start_time", None)
            st.rerun()

        if submitted and not time_up:
            try:
                score = 0
                if q["type"] == "MCQ":
                    score = evaluate_mcq([user_answer], q["correct_answer"])
                elif q["type"] == "ShortAnswer":
                    score = float(evaluate_short_answer(user_answer, q["correct_answer"]))
                elif q["type"] == "Coding":
                    result = run_code_in_sandbox(user_answer, q["test_cases"])
                    passed = result.get("passed", 0)
                    total = result.get("total", 1)
                    score = passed / total
                    st.write("Code Result:", result)

                for tag in st.session_state.current_tag:
                    st.session_state.question_counts[tag] += 1
                st.session_state.question_count += 1
                st.session_state.step = "next_question"
                updated = update_beliefs(tags=st.session_state.current_tag, score=score)
                st.session_state.beliefs = updated
                st.success("Submitted successfully")
                st.session_state.flag = True
                st.session_state.pop("question_start_time", None)
                st.rerun()
            except Exception as e:
                print(f"Error during evaluation: {e}")
                st.error(f"Error in generating the question please restart the test.")



    # === Step 4: Summary ===
    elif st.session_state.step == "summarize":
        try:
            summary = summarize_results(st.session_state.beliefs)
            st.subheader("Final Evaluation Summary")
            st.markdown(summary)
            st.write("Beliefs:", st.session_state.beliefs)

            if st.button("Restart"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()
        except Exception as e:
            print(f"Failed to summarize results: {e}")
            st.error(f"Error in generating the question please restart the test.")
if st.session_state.role == "sme":
    st.set_page_config(page_title="Firecrawl Quiz Generator", layout="centered")
    load_css()
    st.title("Web-Based Intelligent Quiz Generator")

    if "step" not in st.session_state:
        st.session_state.step = "input"

    if st.session_state.step == "input":
        st.markdown("Enter one or more comma-separated URLs to generate a quiz from the content.")
        urls_input = st.text_area("URLs (comma-separated):")
        num_q = st.number_input("Number of questions", min_value=1, max_value=20, value=5)

        if st.button("Generate Quiz"):
            urls = [u.strip() for u in urls_input.split(",") if u.strip()]
            if not urls:
                st.warning("Please enter at least one valid URL.")
            else:
                with st.spinner("Scraping websites..."):
                    content = scrape_multiple(urls)
                try:
                    with st.spinner("Generating quiz using LLM..."):
                        quiz = call_llm_generate(content, num_questions=num_q)
                        st.session_state.quiz = quiz
                        st.session_state.step = "quiz"
                        st.rerun()
                except Exception as e:
                    print(f"Failed to generate quiz: {e}")
                    st.error(f"Error in generating the question please restart the test.")

    elif st.session_state.step == "quiz":
        st.subheader("Quiz Questions")
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

        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
if st.session_state.role in ["student", "sme"]:
    if st.button("Go Back to Role Selection"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
