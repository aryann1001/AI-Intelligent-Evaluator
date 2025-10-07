import streamlit as st
import json
import time
from huggingface_hub import InferenceClient
from streamlit_autorefresh import st_autorefresh
from Actions import *
from dotenv import load_dotenv
import os

# === Fireworks.ai client setup ===
load_dotenv()
hf_token = os.getenv("hf_token")
client = InferenceClient(provider="fireworks-ai", api_key=hf_token)

# === Action Map ===
action_map = {
    "generate_tags": generate_tags,
    "generate_question": generate_question,
    "evaluate_mcq": evaluate_mcq,
    "evaluate_short_answer": evaluate_short_answer,
    "run_code_in_sandbox": run_code_in_sandbox,
    "update_beliefs": update_beliefs,
    "summarize_results": summarize_results
}

# # === LLM Response Stub ===
# class FakeLLMResponse:
#     def __init__(self, content):
#         self.content = content

# def call_llm_agent(messages, actions=None):
#     return FakeLLMResponse("""**Question 5:** What is the difference between a list and a tuple in Python?

# A) A list is a collection of items that can be changed, while a tuple is a collection of items that cannot be changed  
# B) A list is a collection of items that cannot be changed, while a tuple is a collection of items that can be changed  
# C) A list and a tuple are equivalent and can be used interchangeably  
# D) A list and a tuple are not supported in Python  

# Please respond with the letter of your chosen answer.""")

def call_llm_agent(messages, actions=None):
    # prompt = json.dumps({"messages": messages, "actions": actions or []})
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages
    )
    return completion.choices[0].message

# === Utility ===
def clear_user_input():
    st.session_state["user_answer_input"] = ""

# === UI Setup ===
st.set_page_config(page_title="Intelligent Evaluator Agent", layout="centered")
st.title("🧠 Intelligent Evaluator")

# === Session Initialization ===
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "system", "content": f"You are an intelligent evaluator that tests the user's knowledge on selected topic. Choose actions, ask questions, evaluate responses."}
#     ]

action_instruction_prompt = """
You are an intelligent evaluator designed to assess a user's knowledge on a given topic using different question types. 
You have the ability to call the following actions to assist in evaluation:

1. generate_tags(topic: str) → returns {"topic": str, "tags": List[str], "beliefs": dict}
   - Generates subtopics/tags for the evaluation based on a main topic.

2. generate_question(tag: list, type: str, difficulty: str = "medium") → returns question object in JSON
   - Generates a single question of a specified type ("MCQ", "ShortAnswer", "Coding") and difficulty for given tags.

3. evaluate_mcq(choosen_answer: list, correct_answer: list) → returns float
   - Compares chosen answers against correct ones, returns score.

4. evaluate_short_answer(user_answer: str, correct_answer: str) → returns similarity score string (0.000 to 1.000)

5. run_code_in_sandbox(code: str, testcases: list) → returns test result summary with pass/fail counts and errors

6. update_beliefs(tags: list, score: float) → returns updated belief scores dict

7. summarize_results(beliefs: dict) → returns a string summary of user's strengths and weaknesses.

When you want to call an action, respond exactly in this format:

CALL: action_name {"param1": value1, "param2": value2, ...}

Do NOT include any markdown or commentary around the action call. Only use CALL when you need the result to continue.
"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": action_instruction_prompt.strip()
        }
    ]


if "action_results" not in st.session_state:
    st.session_state.action_results = []

# === Start Assessment ===
if "started" not in st.session_state:
    topic = st.text_input("Enter the topic to evaluate:", "Python")
    if st.button("Start Assessment"):
        st.session_state.started = True
        st.session_state.messages.append({"role": "user", "content": f"Start evaluating the topic: {topic}"})
        st.rerun()

# === Assessment Flow ===
if st.session_state.get("started", False):
    # Refresh every second to update countdown
    st_autorefresh(interval=1000, limit=None, key="timer_refresh")

    llm_response = call_llm_agent(st.session_state.messages)
    content = llm_response.content or ""
    st.chat_message("ai").write(content)

    try:
        # Check for action call
        if "CALL:" in content:
            print("here")
            try:
                _, rest = content.split("CALL:")
                action_name, args_json = rest.strip().split(" ", 1)
                action_name = action_name.strip()
                args = json.loads(args_json)
                if action_name in action_map:
                    try:
                        # Call the action function with the parsed args
                        result = action_map[action_name](**args) if isinstance(args, dict) else action_map[action_name]()

                        # Store the result
                        st.session_state.action_results.append({action_name: result})

                        # Add the assistant message and action response message
                        st.session_state.messages.append({"role": "assistant", "content": content})
                        st.session_state.messages.append({
                            "role": "action",
                            "name": action_name,
                            "content": json.dumps(result)
                        })

                    except Exception as e:
                        st.error(f"Error executing action '{action_name}': {e}")
                else:
                    st.error(f"Action '{action_name}' not recognized.")
            except Exception as e:
                st.error(f"Failed to parse action call: {e}")
        else:
            # Display question and handle answer
            st.session_state.messages.append({"role": "assistant", "content": content})

            if "user_answer" not in st.session_state:
                st.session_state.user_answer = ""

            if st.session_state.get("clear_input_next", False):
                clear_user_input()
                st.session_state.clear_input_next = False

            # Start timer for question
            if "question_start_time" not in st.session_state:
                st.session_state.question_start_time = time.time()

            # Show countdown
            time_limit = 60
            elapsed = time.time() - st.session_state.question_start_time
            remaining_time = max(0, int(time_limit - elapsed))
            st.info(f"⏳ Time remaining: {remaining_time} seconds")

            if remaining_time == 0:
                st.error("⏰ Time's up! This answer will not be evaluated.")

            # Input and submission
            user_answer = st.text_input("Your Answer:", value="", key="user_answer_input")
            if st.button("Submit Answer"):
                if user_answer.strip():
                    time_taken = time.time() - st.session_state.question_start_time
                    if time_taken > time_limit:
                        st.warning("⏰ Time limit exceeded! This answer will not be evaluated.")
                    else:
                        st.session_state.messages.append({"role": "user", "content": user_answer.strip()})
                    st.session_state.clear_input_next = True
                    st.session_state.pop("question_start_time", None)
                    st.rerun()

    except Exception as e:
        st.error(f"Error parsing action call or response: {e}")



# import streamlit as st
# import json
# from huggingface_hub import InferenceClient
# from Actions import *
# from dotenv import load_dotenv
# import os
# # === Fireworks.ai client setup ===
# load_dotenv()
# hf_token = os.getenv("hf_token")
# client = InferenceClient(provider="fireworks-ai", api_key=hf_token)

# # === action STUBS (replace these with your real action implementations) ===


# # === action MAP ===
# action_map = {
#     "generate_tags": generate_tags,
#     "generate_question": generate_question,
#     "evaluate_mcq": evaluate_mcq,
#     "evaluate_short_answer": evaluate_short_answer,
#     "run_code_in_sandbox": run_code_in_sandbox,
#     "update_beliefs": update_beliefs,
#     "summarize_results": summarize_results
# }

# # === LLM CALL WRAPPER ===

# class FakeLLMResponse:
#     def __init__(self, content):
#         self.content = content

# def call_llm_agent(messages, actions=None):
#     return FakeLLMResponse("""**Question 5:** What is the difference between a list and a tuple in Python?

# A) A list is a collection of items that can be changed, while a tuple is a collection of items that cannot be changed  
# B) A list is a collection of items that cannot be changed, while a tuple is a collection of items that can be changed  
# C) A list and a tuple are equivalent and can be used interchangeably  
# D) A list and a tuple are not supported in Python

# Please respond with the letter of your chosen answer.""")



# # === Streamlit UI ===

# def clear_user_input():
#     st.session_state["user_answer_input"] = ""

# st.set_page_config(page_title="Intelligent Evaluator Agent", layout="centered")
# st.title("🧠 Intelligent Evaluator")

# # Session state 
# # initializations runs only once {
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "system", "content": "You are an intelligent evaluator that tests the user's knowledge on Python topics. Choose actions, ask questions, evaluate responses."}
#     ]
# if "action_results" not in st.session_state:
#     st.session_state.action_results = []
# # }

# # === User input to start assessment ===
# if "started" not in st.session_state:
#     topic = st.text_input("Enter the topic to evaluate:", "Python")
#     if st.button("Start Assessment"):
#         st.session_state.started = True
#         st.session_state.messages.append({"role": "user", "content": f"Start evaluating the topic: {topic}"})
#         st.rerun()

# # === Assessment Flow ===
# if st.session_state.get("started", False):

#     # Call LLM agent
#     llm_response = call_llm_agent(st.session_state.messages)

#     # Check for action call in LLM response (simulated parsing)
#     content = llm_response.content or ""
#     st.chat_message("ai").write(content)

#     try:
#         # Check if action call embedded in message (format: CALL: action_name args_json)
#         if "CALL:" in content:
#             _, rest = content.split("CALL:")
#             action_name, args_json = rest.strip().split(" ", 1)
#             action_name = action_name.strip()
#             args = json.loads(args_json)

#             if action_name in action_map:
#                 result = action_map[action_name](**args)
#                 st.session_state.action_results.append({action_name: result})

#                 # Feed action result back to agent
#                 action_response_msg = {   
#                     "role": "action",
#                     "name": action_name,
#                     "content": json.dumps(result)
#                 }
#                 st.session_state.messages.append({"role": "assistant", "content": content})
#                 st.session_state.messages.append(action_response_msg)
#                 # st.rerun()
#             else:
#                 st.error(f"action '{action_name}' not recognized.")
#         else:
#             # If it's a question, display and wait for user input
#             st.session_state.messages.append({"role": "assistant", "content": content})

#             # Ensure we reset user_answer only when new question is added
#             if "user_answer" not in st.session_state:
#                 st.session_state.user_answer = ""

#             if st.session_state.get("clear_input_next", False):
#                 clear_user_input()
#                 st.session_state.clear_input_next = False
#             # Text input for answer
#             user_answer = st.text_input("Your Answer:", value="", key="user_answer_input")
#             if st.button("Submit Answer"):
#                 if user_answer.strip():
#                     print(user_answer)
#                     st.session_state.messages.append({"role": "user", "content": user_answer.strip()})
#                     # Step 3: Set a flag to clear input on next rerun
#                     st.session_state.clear_input_next = True
#                     st.rerun()



#     except Exception as e:
#         st.error(f"Error parsing action call or response: {e}")
