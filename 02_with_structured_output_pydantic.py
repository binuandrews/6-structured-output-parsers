"""
02_with_structured_output_pydantic.py
-------------------------------------
Concept: Native structured output using Pydantic models
LangChain Feature: with_structured_output()

This script demonstrates:
- How to define a structured output schema using a Pydantic model.
- How to ask an LLM to return responses validated by that model.
- How to access typed fields and leverage validation automatically.

Example Use Case:
Extract structured metadata (title, author, genre) from a book description.
"""

import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Must be set BEFORE importing google.generativeai
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# Setup
# ==========================================
load_dotenv()

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

provider = config["provider"]
model_cfg = config[provider]

# Initialize model
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
# Step 1: Define structured schema with Pydantic
# ==========================================
class BookInfo(BaseModel):
    """Structured schema for extracting book metadata."""
    title: str = Field(..., description="The title of the book.")
    author: str = Field(..., description="The author of the book.")
    genre: str = Field(..., description="The main literary genre or category of the book.")
    key_themes: list[str] = Field(..., description="List of major themes covered in the book.")
'''
This creates a validation-enforced blueprint for book data extraction.

Key Components:

* BaseModel (vs TypedDict from before)
- Pydantic BaseModel = Validation on steroids! Not just a blueprint, but an active enforcer
- Difference from TypedDict:
    - TypedDict = "Please follow this shape" (gentle suggestion)
    - BaseModel = "You MUST follow this shape AND these rules" (strict enforcement)

* Field(...) - The Three Dots Mystery
- ... (ellipsis) means "REQUIRED - this field MUST be present"
- Alternative could be Field("default value") for optional fields
- It's Python's way of saying "this field has no default, so it must be provided"

* description= - Instructions for the AI
- Each field gets a plain-English explanation of what should go there
- Helps the AI understand what you want in each slot
'''

# Wrap model with structured output enforcement
llm = base_llm.with_structured_output(BookInfo)
'''
This transforms a free-text AI into a structured-data robot that MUST follow your Pydantic rules.

** What happens in the background:
1. Schema Conversion: LangChain takes your BookInfo model and converts it to:
- JSON Schema for the AI to understand
- Instructions like "Return a JSON object with these exact fields"

2. Response Parsing: When the AI responds, it automatically:
- Validates the response against your schema
- Converts JSON to a proper BookInfo object
- Raises errors if the AI didn't follow the rules
'''

# ==========================================
# Step 2: Create prompt and run
# ==========================================
prompt = (
    "Extract the following structured details from this book description:\n\n"
    "'In The Silent Forest, renowned author Anna Grey crafts a moving tale of survival "
    "and human resilience, set against the backdrop of a mysterious post-apocalyptic world. "
    "Themes of hope, fear, and companionship drive the narrative.'"
)

print("📚 Structured Output with Pydantic Demonstration")
print("-----------------------------------------------")
print("🔹 Prompt:\n")
print(prompt)
print("\n🔹 Model Structured Response:\n")

response = llm.invoke(prompt)

# Print structured validated object
print(response)

print("\n🔹 Accessing Individual Fields:\n")
print(f"Title: {response.title}")
print(f"Author: {response.author}")
print(f"Genre: {response.genre}")
print(f"Key Themes: {', '.join(response.key_themes)}")