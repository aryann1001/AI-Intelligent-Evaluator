# Intelligent Evaluator

An interactive Streamlit application to automatically generate and evaluate quizzes on technical topics. It supports two roles:

- **Student**: Adaptive assessment driven by an LLM. Questions are generated across MCQ, Short Answer, and Coding, with timers and live belief updates per topic.
- **SME (Subject Matter Expert)**: Generate quizzes from one or more URLs using Firecrawl + LLM.

The UI is theme-aware and works in both Streamlit light and dark themes. Global styles are managed via `assets/styles.css`.

---

## Features

- **Adaptive Student Flow** in `App.py`
  - Tag generation from topic
  - LLM-driven selection of next question type/tags/difficulty
  - Question rendering (MCQ, Short Answer, Coding with ACE editor)
  - Per-question timer and stateful navigation (submit/skip/end)
  - Belief updates per tag and final summary
- **SME Flow** in `App.py`
  - Firecrawl-based web scraping
  - LLM-based quiz generation from scraped content
- **Standalone SME App** in `Mcp_Generator.py` (optional alt entry)
- **Agent-style Prototype** in `Intelligent_Evaluator_Agent.py` (LLM that can "call" actions)
- **Theming**
  - `.streamlit/config.toml` keeps theme minimal
  - `assets/styles.css` switches variables using `[data-theme="light"|"dark"]`

---

## Directory Structure

```
.
├─ App.py                        # Main app with role selection: Student and SME flows
├─ Actions.py                    # LLM prompts, evaluation logic, coding sandbox
├─ Intelligent_Evaluator_Agent.py# Agent-style prototype using action calls
├─ Mcp_Action.py                 # Firecrawl scraping + quiz generation helpers
├─ Mcp_Generator.py              # Simple generator app for SME use-case
├─ Test.py                       # Local test for run_code_in_sandbox()
├─ requirements.txt              # Python dependencies (see notes below)
├─ .streamlit/
│  └─ config.toml               # Streamlit theme config
└─ assets/
   └─ styles.css                # Global CSS (light/dark aware)
```

---

## Prerequisites

- Python 3.10+
- pip
- For coding sandbox evaluation: Docker Desktop running locally
  - Windows users should enable WSL2 backend
- Accounts/API keys
  - Hugging Face token (Fireworks provider): `hf_token`
  - Firecrawl API key: `firecrawl_api_key`

---

## Environment Variables

Create a `.env` file in the project root:

```
hf_token=YOUR_HUGGINGFACE_TOKEN
firecrawl_api_key=YOUR_FIRECRAWL_KEY
```

Do not commit your `.env` file.

---

## Installation

It’s recommended to use a virtual environment.

```
python -m venv venv
# Windows PowerShell
./venv/Scripts/Activate.ps1

pip install -r requirements.txt

# Additional packages used by the code (if not in requirements)
pip install docker sentence-transformers torch
```

Notes:
- `Actions.py` imports `docker`, `sentence_transformers`, and `torch` for the sandbox and text similarity. If these are missing from `requirements.txt`, install them as shown above.
- Torch install can be platform specific. See https://pytorch.org/get-started/locally/ for optimized wheels if needed.

---

## Running the App

Run the main app:

```
streamlit run App.py
```

Alternative SME-focused generator:

```
streamlit run Mcp_Generator.py
```

The terminal will show a local URL (default http://localhost:8501).

---

## Theming and Styling

- Base theme is configured in `.streamlit/config.toml` with `base = "light"` and a `primaryColor`.
- Global styling is injected via `load_css()` in `App.py`, loading `assets/styles.css`.
- The stylesheet defines variables for both themes:
  - `[data-theme="light"]` and `[data-theme="dark"]` set colors for `--bg`, `--surface`, `--text`, `--border`, `--muted`.
- To switch themes at runtime: Streamlit menu → Settings → Theme.

---

## Workflows

### Student Flow (in `App.py`)

1. Role selection → choose Student.
2. Enter a topic (e.g., "Python").
3. `generate_tags()` creates tag list and initializes beliefs.
4. For each question:
   - `call_llm_for_next_question()` decides next tags/type/difficulty.
   - `generate_question()` produces a strictly JSON question object.
   - UI renders based on type:
     - MCQ: `st.selectbox()`
     - Short Answer: `st.text_input()` and semantic scoring via Sentence Transformers
     - Coding: ACE editor captures code, run against `test_cases` in Docker
   - Submit/Skip/End controls update `question_count` and `beliefs` via `update_beliefs()`.
5. After `max_questions`, `summarize_results()` outputs strengths/weaknesses.

Timing:
- A per-question timer is displayed via a small HTML/JS snippet; logic for timeouts is managed in Python.

### SME Flow (in `App.py`)

1. Role selection → choose SME.
2. Enter one or more URLs and desired number of questions.
3. `scrape_multiple()` fetches visible text via Firecrawl API.
4. `call_llm_generate()` prompts the LLM to produce a JSON array of mixed-type questions.
5. Quiz is displayed (MCQ/ShortAnswer/Coding); coding questions also show `test_cases` when present.

### Agent Prototype (in `Intelligent_Evaluator_Agent.py`)

- Demonstrates an agent pattern where the LLM can request to call functions by emitting a `CALL: action_name {json}` string.
- The app parses the call, executes the mapped function from `action_map`, and feeds the result back into the conversation.

---

## Key Modules

- `Actions.py`
  - `query_llm(prompt)` – wraps LLM chat completions
  - `extract_json(raw_response)` – robust extraction from fenced blocks
  - `generate_tags(topic)` – topic → subtopics and initializes beliefs
  - `generate_question(tag, type, difficulty)` – emits one JSON question
  - `evaluate_mcq(choosen_answer, correct_answer)` – partial credit scoring
  - `evaluate_short_answer(user_answer, correct_answer)` – semantic similarity using Sentence Transformers
  - `run_code_in_sandbox(code, testcases)` – executes user code in a Python Docker container with memory/CPU/network limits
  - `update_beliefs(tags, score)` – running mean per tag
  - `summarize_results(beliefs)` – strengths/weaknesses string

- `Mcp_Action.py`
  - `scrape_with_firecrawl(url)` – calls Firecrawl API
  - `scrape_multiple(urls)` – loops and aggregates
  - `call_llm_generate(content, num_questions)` – JSON quiz array from content

- `App.py`
  - `load_css()` – injects global CSS
  - Contains both Student and SME flows with session-state driven navigation

---

## Security & Privacy

- Do not log or store sensitive API keys. Use `.env` locally and secure secrets in deployments.
- Code execution for coding questions is sandboxed in Docker with limited resources and no network, but always treat untrusted code carefully.

---

## Troubleshooting

- "LLM parsing failed" or JSON errors:
  - LLMs occasionally return formatting noise; `extract_json()` and callers already attempt to sanitize. Retry if it persists.
- Docker errors during coding evaluation:
  - Ensure Docker Desktop is running and accessible to your user.
  - On Windows, enable WSL2 backend.
- Missing packages (e.g., `docker`, `sentence_transformers`, `torch`):
  - Install them as shown in Installation or add them to `requirements.txt`.
- Theme visibility (light vs dark):
  - The stylesheet uses theme-scoped variables. Use Streamlit menu → Settings → Theme to confirm.

---

## Development Notes

- `st.set_page_config()` should be called before rendering; we also load CSS early for all role states.
- Keep outputs strictly JSON in LLM prompts where parsing is expected.
- Consider caching embeddings/models if performance becomes a concern.
- You can factor prompts into dedicated helpers if you plan to support more subjects.

---

## Roadmap Ideas

- Add persistence of results per user/session
- Richer analytics on belief trajectories
- Human-in-the-loop SME edits to generated questions
- More robust JSON schema validation and retry logic
- Model/config abstraction to swap LLMs/providers
- Optional in-app theme toggle

---

## License

Proprietary. All rights reserved. Update this section if you adopt an open-source license.
