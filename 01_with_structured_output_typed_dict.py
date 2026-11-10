"""
01_with_structured_output_typed_dict.py
---------------------------------------
Concept: Native structured output using TypedDict
LangChain Feature: with_structured_output()

This script demonstrates:
- How to define a Python TypedDict schema for structured responses.
- How to ask the LLM to return data following that schema directly.
- How to access the structured (parsed) data in Python.

Example Use Case:
Extract key details from a simple product description into a structured format.
"""

import os
import json
from typing import TypedDict
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

# Load model config
with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
model_cfg = config[provider]

# Initialize LLM dynamically
if provider == "openai":
    base_llm = ChatOpenAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.7),
        max_tokens=model_cfg.get("max_tokens", 300),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif provider == "gemini":
    base_llm = ChatGoogleGenerativeAI(
        model=model_cfg.get("model"),
        temperature=model_cfg.get("temperature", 0.7),
        max_output_tokens=model_cfg.get("max_output_tokens", 300),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
else:
    raise ValueError("Invalid provider in config.json — must be 'openai' or 'gemini'.")


# ==========================================
# Step 1: Define structured schema
# ==========================================
class ProductInfo(TypedDict):
    """Defines the structured output schema for product extraction."""
    product_name: str
    category: str
    key_features: list[str]
'''
This creates a Python dictionary blueprint with specific requirements.

TypedDict = A way to say "this dictionary must have exactly these keys with these data types"
product_name: str = Key must exist, value must be text
category: str = Another text field
key_features: list[str] = Must be a list of strings (like ["waterproof", "lightweight", "durable"])
'''

# Wrap the base LLM to enforce structured output
llm = base_llm.with_structured_output(ProductInfo)
'''
> This upgrades a regular LLM into a structured-data machine.

> Without with_structured_output:
User: "Tell me about this product"
AI: "The Galaxy X is a great smartphone with amazing camera..." 
(free-form text - unpredictable format)

> With with_structured_output:
User: "Tell me about this product"
AI: {
    "product_name": "Galaxy X",
    "category": "Smartphone",
    "key_features": ["Great camera", "Long battery", "5G capable"]
}
(always follows your exact blueprint)

> How it works internally:

Takes your ProductInfo schema
Converts it to instructions for the LLM (like "return JSON with these fields")
Forces the LLM's response into that exact shape
If the LLM tries to deviate, it gets corrected or re-prompted

> Why This is Powerful:

Predictability - Every response has the same structure
Easy processing - Can directly convert to JSON, database rows, etc.
Error prevention - No missing or extra fields
Type safety - Guarantees you get lists where you expect lists
Better for applications - Other code can rely on the format
'''

# ==========================================
# Step 2: Prepare prompt & run query
# ==========================================
prompt = (
    "Extract the product name, category, and 3 key features "
    "from the following description:\n\n"
    "The new SoundMax Pro X headphones deliver superior bass, "
    "active noise cancellation, and 30 hours of wireless playback. "
    "They are perfect for travel and studio use."
)

print("🎯 Structured Output with TypedDict Demonstration")
print("-----------------------------------------------")
print("🔹 Prompt:\n")
print(prompt)

print("\n🔹 Model Structured Response:\n")
# Call the LLM
structured_response = llm.invoke(prompt)
# Print structured Python dict
print(structured_response)

print("\n🔹 Accessing Individual Fields:\n")
print(f"Product Name: {structured_response['product_name']}")
print(f"Category: {structured_response['category']}")
print(f"Key Features: {', '.join(structured_response['key_features'])}")