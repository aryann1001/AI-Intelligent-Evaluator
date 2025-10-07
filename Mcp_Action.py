import os
import json
import re
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
 
load_dotenv()
 
FIRECRAWL_API_KEY = os.getenv("firecrawl_api_key")
HF_TOKEN = os.getenv("hf_token")
 
client = InferenceClient(provider="fireworks-ai", api_key=HF_TOKEN)
 
def scrape_with_firecrawl(url: str) -> str:
    """Scrape visible text content from a single webpage using Firecrawl."""
    response = requests.post(
        "https://api.firecrawl.dev/v1/scrape",
        headers={
            "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"url": url}
    )
    if response.status_code != 200:
        raise Exception(f"Failed to scrape {url}: {response.text}")
   
    data = response.json()
    return data.get("content", {}).get("text", "")
 
def scrape_multiple(urls: list) -> str:
    texts = []
    for url in urls:
        try:
            content = scrape_with_firecrawl(url)
            texts.append(f"Content from {url}:\n{content}")
        except Exception as e:
            texts.append(f"[ERROR scraping {url}]: {e}")
    return "\n\n".join(texts)
 
def call_llm_generate(content: str, num_questions=5):
    """Generate a list of quiz questions from scraped content."""
    prompt = f"""
You are a helpful quiz generator assistant.
 
From the text content below, generate {num_questions} quiz questions.
Use different types: MCQ, ShortAnswer, and Coding.
 
Return a JSON array like this:
 
[
 {{
   "type": "MCQ",
   "question": "...",
   "options": ["A", "B", "C", "D"],
   "correct_answer": ["A"]
 }},
 {{
   "type": "ShortAnswer",
   "question": "...",
   "options": [],
   "correct_answer": "..."
 }},
 {{
   "type": "Coding",
   "question": "...",
   "options": [],
   "correct_answer": "Expected behavior or output",
   "test_cases": [
     {{
       "input": [1, 2],
       "expected_output": 3
     }}
   ]
 }}
]
 
Only return valid JSON. No explanation.
 
Content:
\"\"\"{content}\"\"\"
"""
 
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "system", "content": prompt.strip()}]
    )
 
    raw = response.choices[0].message.content.strip()
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw)
 
    if not cleaned:
        raise ValueError("LLM returned empty or null response.")
 
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response:\n{raw}\n\nError: {e}")