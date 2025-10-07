import json
from Actions import run_code_in_sandbox  # replace with actual filename or remove if in same file

# Sample user-submitted code (must define a function named `solution`)
user_code = """
def solution(x, y):
    return x + y
"""

# Sample test cases
test_cases = [
    {"input": "2, 3", "expected_output": "5"},
    {"input": "0, 0", "expected_output": "0"},
    {"input": "-5, 5", "expected_output": "0"},
    {"input": "100, 200", "expected_output": "300"},
]

# Run the sandbox test
result = run_code_in_sandbox(user_code, test_cases)

# Pretty print results
print("=== Test Results ===")
print(json.dumps(result, indent=2))
