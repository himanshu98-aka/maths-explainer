import os
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import streamlit as st

def load_api_keys():
    """
    Loads Gemini API keys from Streamlit secrets and/or a .env file.
    Streamlit secrets are prioritized for deployment environments.
    """
    # Attempt to load keys from .env file for local development
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # python-dotenv is not installed, which is fine for deployment
        pass

    # Use a set to avoid duplicate keys
    keys = set()
    # Check Streamlit secrets first
    if "GEMINI_API_KEY_1" in st.secrets: keys.add(st.secrets["GEMINI_API_KEY_1"])
    if "GEMINI_API_KEY_2" in st.secrets: keys.add(st.secrets["GEMINI_API_KEY_2"])
    # Check environment variables (which .env populates)
    if os.getenv("GEMINI_API_KEY_1"): keys.add(os.getenv("GEMINI_API_KEY_1"))
    if os.getenv("GEMINI_API_KEY_2"): keys.add(os.getenv("GEMINI_API_KEY_2"))
    
    return [k for k in keys if k]

def generate_with_failover(prompt: str, model_name: str = "gemini-1.5-flash-latest", file_name: str | None = None) -> str:
    """
    Generate content using a pool of API keys, with automatic failover.
    Optionally accepts a file_name for RAG.
    """
    if not API_KEYS:
        return "ERROR: No API keys configured."

    # Use the correct model name
    for key in API_KEYS:
        try:
            # Configure the API key for this attempt
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name)

            # Prepare content for the API call
            content = [prompt]
            if file_name:
                content.insert(0, genai.get_file(file_name))

            resp = model.generate_content(content)
            return (resp.text or "").strip() or "ERROR: Empty response from model."
        except google_exceptions.ResourceExhausted:
            continue
        except Exception as e:
            return f"ERROR: An unexpected error occurred: {e}"

    return "ERROR: All API keys are currently rate-limited. Please wait and try again."

# Load keys when the module is imported
API_KEYS = load_api_keys()