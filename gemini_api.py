import os
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import streamlit as st

# Use st.cache_data to load keys once and cache the result.
@st.cache_data(show_spinner=False)
def get_api_keys():
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
    # Safely check for Streamlit secrets. This will fail if no secrets file is found.
    try:
        if "GEMINI_API_KEY_1" in st.secrets: keys.add(st.secrets["GEMINI_API_KEY_1"])
        if "GEMINI_API_KEY_2" in st.secrets: keys.add(st.secrets["GEMINI_API_KEY_2"])
    except st.errors.StreamlitAPIException:
        # This is expected when running locally without a secrets.toml file.
        pass
    # Check environment variables (which .env populates)
    if os.getenv("GEMINI_API_KEY_1"): keys.add(os.getenv("GEMINI_API_KEY_1"))
    if os.getenv("GEMINI_API_KEY_2"): keys.add(os.getenv("GEMINI_API_KEY_2"))
    
    return [k for k in keys if k]

def generate_with_failover(prompt: str, model_name: str = "gemini-2.0-flash-lite", file_name: str | None = None, system_instruction: str | None = None) -> str:
    """
    Generate content using a pool of API keys, with automatic failover.
    Optionally accepts a file_name for RAG.
    """
    # Load keys on first use
    api_keys = get_api_keys()
    if not api_keys:
        return "ERROR: No API keys configured."

    # Use the correct model name
    for key in api_keys:
        try:
            # Configure the API key for this attempt
            genai.configure(api_key=key)
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction
            )

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