import json
from huggingface_hub import InferenceClient
import streamlit as st
import os
from dotenv import load_dotenv
import re
import docker
import uuid
import tempfile
from sentence_transformers import SentenceTransformer, util
import time
import torch


load_dotenv()
# === Setup Inference Client ===
client = InferenceClient(provider="fireworks-ai", api_key=os.getenv("hf_token"))
def query_llm(prompt):
    # prompt = json.dumps({"messages": messages, "actions": actions or []})
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": prompt}
            ],
        )
        return completion.choices[0].message["content"]
    except:
        return {'Error generating the question.'}


def extract_json(raw_response: str):

    raw = raw_response.strip()

    # Handle markdown code blocks like ```json ... ``` or ``` ...
    if "```" in raw:
        _, _, after = raw.partition("```")
        after = after.strip()

        if after.lower().startswith("json"):
            after = after[4:].strip()

        if after.endswith("```"):
            after = after[:-3].strip()

        raw = after
    raw = raw.replace(": None", ": null")
    # Attempt JSON parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON:\n{e}\n\nContent:\n{raw}")
        raise ValueError(f"Error in generating the question please restart the test.")


def generate_tags(topic: str):
    prompt = f"""
You are a helpful assistant designed to break down a learning topic into its core subtopics.
Given a topic, return a JSON object with two keys: "topic" and "subtopics".

Requirements:
- The "topic" key should have the name of the topic.
- The "subtopics" key should be a list of 5 to 10 relevant subtopics necessary to evaluate knowledge in that topic.
- Respond ONLY with valid JSON.

Example:
Input: Python  
Output:
{{
"topic": "Python",
"subtopics": ["Data Types", "Control Flow", "Functions", "OOP", "Modules", "File I/O", "Error Handling"]
}}

Now generate for topic: {topic}
"""

    try:
        raw_response = query_llm(prompt)
        parsed = json.loads(raw_response)
        subtopics = parsed.get("subtopics", [])

        # Initialize beliefs in session state
        if "beliefs" not in st.session_state:
            st.session_state.beliefs = {}

        if 'question_counts' not in st.session_state:
            st.session_state.question_counts = {}

        for tag in subtopics:
            st.session_state.beliefs[tag] = 0.5 
            st.session_state.question_counts[tag] = 1

        return {
            "topic": topic,
            "tags": subtopics,
            "beliefs": st.session_state.beliefs
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to generate tags or parse LLM response."
        }
 

def generate_question(tag: list,type: str, difficulty: str = "medium"):
    prompt = f"""
You are a helpful assistant designed to generate **one** Python assessment question based on the given topics and type and difficulty.
MCQ are option questions where one or more are correct 
ShortAnswer are question which are meant to test users subject knowledge not code
Coding are questions which are supposed to ask coding question, to evaluate users appilication of learned knowlege.
- The question should ideally combine multiple related tags in one prompt to evaluate multiple areas at once.
Inputs :
- topics: {tag}          
- type: {type}           
- difficulty: "{difficulty}"

Specifications:
- Question difficulty should match the given difficulty.
- The question should me covering those tags.
- Time limits:
    • MCQ or ShortAnswer → time_limit = 120
    • Coding → time_limit = 600

Output:
Respond only with a single JSON object no more statments just with the object, which contain only one question using exactly the template as below in Md format:
You are a JSON-compliant assistant. All outputs must be strictly valid JSON using double quotes for keys and string values.
If type == "MCQ":
{{
  "question": "<string>",
  "options": ["<opt1>", "<opt2>", "<opt3>", "<opt4>"],
  "type": "MCQ",
  "correct_answer": "[<correctopt1> , <correctopt2> , ..] ",
  "time_limit": 120
}}

If type == "ShortAnswer:
    {{
    "question": "<string>",
    "options": [],
    "type": "ShortAnswer",
    "correct_answer":"<model answer: 1-2 sentences>",
    "time_limit": 120
    }}
If type == "Coding":
    {{
    "question": "<string>",
    "options": [],
    "type": "Coding",
    "test_cases": [
        {{
            "input": <literal or list/tuple>,
            "expected_output": <literal or list/tuple>
        }},
        // 'include at least 5-10 test cases'
        // Testcases shouldn't contain none
        ]
    ,
    "time_limit": 600
    }}

"""
    
    # prompt = generate_questions_prompt(tag, difficulty)
    raw_response = ""
    try:
        raw_response = query_llm(prompt)
        time.sleep(1)
        print(raw_response)
        questions = extract_json(raw_response)
        return questions
    except Exception as e:
        print(f"Failed to parse LLM response: {e}\nRaw:\n{raw_response}")
        raise ValueError(f"Error in generating the question please restart the test.")



def evaluate_mcq(choosen_answer: list, correct_answer: list):
    # need to count the number of corrrect options choosen
    correct_answer = [x.strip().upper() for x in correct_answer]
    score = 0
    for i in choosen_answer:
        if i.strip().upper() in correct_answer:
            score += 1
    return score/len(correct_answer)


def evaluate_short_answer(user_answer: str, correct_answer: str) -> int:
    """
    Compares two text answers and returns:
    - 1 if semantic similarity >= 0.5
    - 0 if similarity < 0.5
    """
    # Compute embeddings
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = _model.encode([user_answer, correct_answer], convert_to_tensor=True, device='cpu')
   
    # Calculate cosine similarity
    sim_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
   
    # Debug: print score if needed
    # print(f"Similarity score: {sim_score:.3f}")
   
    return 1 if sim_score >= 0.5 else 0



def run_code_in_sandbox(code: str, testcases: list):
    temp_dir = tempfile.gettempdir()
    client = docker.from_env()

    passed = 0
    failed = 0
    errors = []

    for test in testcases:
        test_input = test["input"]
        expected_output = str(test["expected_output"]) 

        filename = os.path.join(temp_dir, f"{uuid.uuid4().hex}.py")
        with open(filename, 'w') as f:
            f.write(code.strip() + f"\nprint(solution({test_input}))")

        try:
            result = client.containers.run(
                image="python:3.10-slim",
                command=f"python /code/{os.path.basename(filename)}",
                volumes={temp_dir: {"bind": "/code", "mode": "ro"}},
                stderr=True,
                stdout=True,
                network_disabled=True,
                remove=True,
                mem_limit="128m",
                cpu_quota=50000,
            )
            result_str = result.decode().strip()
            if result_str == expected_output:
                passed += 1
            else:
                failed += 1
                errors.append({
                    "input": test_input,
                    "expected": expected_output,
                    "got": result_str
                })

        except Exception as e:
            failed += 1
            errors.append({
                "input": test_input,
                "error": str(e)
            })

        # Clean up file
        try:
            os.remove(filename)
        except:
            pass

    return {
        "passed": passed,
        "failed": failed,
        "total": len(testcases),
        "details": errors
    }
    return passed/len(testcases)

def update_beliefs(tags: list, score: float):
    for tag in tags:
        n = st.session_state.question_counts[tag]
        current_belief = st.session_state.beliefs[tag]

        # Running mean formula
        new_belief = (current_belief * n + score) / (n + 1)
        print(new_belief)
        # Store updated values
        st.session_state.beliefs[tag] = new_belief
        st.session_state.question_counts[tag] = n + 1

    return st.session_state.beliefs

def summarize_results(beliefs: dict):
    strong_knowledge = [tag for tag, belief in beliefs.items() if belief > 0.7]
    moderate_knowledge = [tag for tag, belief in beliefs.items() if 0.3 < belief <= 0.7]
    weak_knowledge = [tag for tag, belief in beliefs.items() if belief <= 0.3]

    return f"User has strong knowledge in {', '.join(strong_knowledge)} and User has moderate knowledge in {', '.join(moderate_knowledge)} and User has weak knowledge in {', '.join(weak_knowledge)}."

# def query_llm(prompt: str) -> str:
#     try:
#         response = client.text_generation(
#             model="accounts/fireworks/models/llama-v3-8b-instruct",
#             prompt=prompt.strip(),
#             max_new_tokens=200,
#             temperature=0.3
#         )
#         return response.strip()
#     except Exception as e:
#         raise RuntimeError(f"LLM query failed: {e}")


