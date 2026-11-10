"""
03_with_structured_output_json_schema.py
----------------------------------------
Concept: Structured output using external JSON Schema
LangChain Feature: with_structured_output()

This script demonstrates:
- How to load a JSON Schema from a file.
- How to enforce model responses to match that schema.
- How to output structured, validated JSON results.

Example Use Case:
Extract key book information based on a defined schema stored externally.
"""

import os
import json
from dotenv import load_dotenv

# Must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# Setup
# ==========================================
load_dotenv()

with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
model_cfg = config[provider]

# Initialize base LLM
if provider == "openai":
    base_llm = ChatOpenAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.7),
        max_tokens=model_cfg.get("max_tokens", 400),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif provider == "gemini":
    base_llm = ChatGoogleGenerativeAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.7),
        max_output_tokens=model_cfg.get("max_output_tokens", 400),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
else:
    raise ValueError("Invalid provider in config.json — must be 'openai' or 'gemini'.")


# ==========================================
# Step 1: Load JSON Schema from file
# ==========================================
schema_path = "book_schema.json"

with open(schema_path, "r") as f:
    raw_schema = json.load(f)

# Ensure OpenAI-compatible schema structure
schema = {
    "name": raw_schema.get("name", "BookInfo"),
    "description": raw_schema.get("title", "Book Information Schema"),
    "parameters": raw_schema  # 👈 wrap actual schema under 'parameters'
}
'''
This reformats your schema to match what OpenAI expects.

* Why this wrapping is needed:
OpenAI's structured output feature requires a specific format:
- name = Name of the function/schema (top-level identifier)
- description = What this schema is for (helps the AI understand)
- parameters = The actual JSON schema rules (nested inside)

Think of it like mailing a package:
- name = "Fragile Electronics" (label on the box)
- description = "Contains a laptop and charger" (description on customs form)
- parameters = The actual items inside the box (the real content)
'''

# ==========================================
# Step 2: Wrap model with structured output enforcement
# ==========================================
llm = base_llm.with_structured_output(schema=schema)
''' This connects your formatted schema to the LLM. '''

# ==========================================
# Step 3: Define prompt and run
# ==========================================
prompt = (
    "Extract structured book information from this description:\n\n"
    "'In 1984, George Orwell depicts a dystopian future where government surveillance "
    "and censorship dominate society. The novel explores themes of freedom, control, "
    "and resistance against totalitarianism.'"
)

print("📘 Structured Output with External JSON Schema")
print("---------------------------------------------")
print(f"🔹 Loaded Schema:  {schema_path}")
print("\n🔹 Prompt:\n")
print(prompt)
print("\n🔹 Model Structured Response:\n")

response = llm.invoke(prompt)
print(json.dumps(response, indent=2))