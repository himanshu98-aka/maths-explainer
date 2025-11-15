import os
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY_1 = os.getenv("GEMINI_API_KEY_1")
API_KEY_2 = os.getenv("GEMINI_API_KEY_2")
API_KEYS = [k for k in [API_KEY_1, API_KEY_2] if k]

def generate_with_failover(prompt: str, model_name: str = "gemini-1.5-flash-latest", file_name: str | None = None) -> str:
    """
    Generate content using a pool of API keys, with automatic failover.
    Optionally accepts a file_name for RAG.
    """
    if not API_KEYS:
        return "ERROR: No API keys configured."

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