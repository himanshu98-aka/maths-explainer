import streamlit as st
import os
import time
import google.generativeai as genai
import tempfile
import uuid
from dotenv import load_dotenv


# --- Configuration and Initialization ---


# Use st.set_page_config for better display
st.set_page_config(
    page_title="Personalized Math Explainer (Gemini RAG)",
    layout="centered"
)


# Use st.cache_resource for objects that should be created only once (like the API client)
@st.cache_resource
def configure_gemini():
    """Configures the Gemini client."""
    # Load environment variables from .env file (local)
    load_dotenv()

    # Try to get API key from environment or Streamlit secrets
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]

    if not api_key:
        st.error("Error: GEMINI_API_KEY not found. Please configure it in secrets.")
        return False

    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return False


is_client_configured = configure_gemini()


# Initialize session state variables for chat history and the file search store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_search_store_name" not in st.session_state:
    st.session_state.file_search_store_name = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "selected_instructions" not in st.session_state:
    st.session_state.selected_instructions = []


# --- Helper Functions ---


def clean_up_store():
    """Deletes the File Search Store if it exists in session state."""
    if st.session_state.file_search_store_name:
        try:
            with st.spinner(f"Deleting previous RAG store: {st.session_state.file_search_store_name}..."):
                genai.delete_file(name=st.session_state.file_search_store_name)
            st.session_state.file_search_store_name = None
            st.session_state.file_name = None
            st.session_state.chat_history = []
            st.success("RAG store cleaned up successfully.")
        except Exception as e:
            st.warning(f"Could not delete store: {e}. You may need to manually delete it later.")


def upload_syllabus_to_rag(uploaded_file):
    """Uploads the file and creates the File Search Store."""
    clean_up_store() # Cleanup any old store first
    
    # 1. Write the UploadedFile to a temporary local file (required for the API)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        local_file_path = tmp_file.name

    try:
        # 2. Upload the file to Gemini
        store_display_name = f"Math_Syllabus_Store_{uuid.uuid4().hex[:6]}"
        with st.spinner("Step 1/2: Uploading syllabus file..."):
            uploaded_gemini_file = genai.upload_file(path=local_file_path, display_name=uploaded_file.name)
            st.session_state.file_search_store_name = uploaded_gemini_file.name
            st.session_state.file_name = uploaded_file.name

        # 3. Wait for file processing
        with st.spinner("Step 2/2: Processing file..."):
            while uploaded_gemini_file.state.name == "PROCESSING":
                time.sleep(5)
                uploaded_gemini_file = genai.get_file(uploaded_gemini_file.name)
            
            if uploaded_gemini_file.state.name == "ACTIVE":
                st.success("🎉 Syllabus uploaded successfully! You can now ask questions about your course.")
                st.session_state.chat_history = [] # Reset chat history
                st.session_state.chat_history.append(("assistant", f"I have successfully indexed your syllabus file: **{uploaded_file.name}**. What math concept from this document can I explain to you?"))
                st.rerun() # Rerun to update the main chat
            else:
                st.error(f"File processing failed with state: {uploaded_gemini_file.state.name}")
                clean_up_store()

    except Exception as e:
        st.error(f"An error occurred during upload: {e}")
        clean_up_store()
    finally:
        # Clean up the local temporary file
        if os.path.exists(local_file_path):
            os.remove(local_file_path)


def generate_rag_response(prompt: str, file_name: str, selected_instructions: list):
    """Generates a response using Gemini with file context."""
    base_system_instruction = (
        "You are an expert math tutor named Himanshu's Gemini Math Explainer. "
        "Your student is a first-year BCA student interested in Data Science. "
        "Your goal is to provide a detailed, academic explanation of the math concept requested. "
        "The response MUST be structured using Markdown headings (like '## Explanation', '## Formulas/Calculations', etc.). "
        "Ensure the explanation is grounded in the provided syllabus context and tailored to a college student's level. "
        "If you cannot find specific information (like merits/demerits) in the syllabus, base it on general mathematical knowledge for that topic, but prioritize the syllabus content. "
        "The response MUST include the following sections, where applicable to the topic: "
        "1. **Explanation:** A detailed, conceptual overview. "
        "2. **Formulas/Calculations:** The core mathematical equations and an example where possible. **Use LaTeX syntax for all math equations (inline: $...$, display: $$...$$).** "
        "3. **Applications (Data Science Focus):** How this concept is used in programming or data science. "
        "4. **Merits and Demerits/Caveats:** The advantages and limitations of the concept. "
        "If the context does not contain the answer, state that you cannot find the information in the syllabus."
    )
    
    # Append selected custom instructions
    if selected_instructions:
        instructions_text = "\n".join([f"- {instruction}" for instruction in selected_instructions])
        system_instruction = base_system_instruction + f"\n\nADDITIONAL USER PREFERENCES:\n{instructions_text}"
    else:
        system_instruction = base_system_instruction

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        system_instruction=system_instruction
    )

    # Simple exponential backoff loop for API call
    for i in range(3):
        try:
            # Get the file object
            file = genai.get_file(file_name)
            
            response = model.generate_content(
                [file, prompt]
            )
            return response
        except Exception as e:
            if i < 2:
                time.sleep(2 ** i) # Wait 1s, then 2s
            else:
                st.error(f"API call failed after multiple retries: {e}")
                return None
    return None


# --- Streamlit UI Layout ---


st.title("🔢 Personalized Math Explainer")
st.markdown("---")

st.sidebar.header("Try it:")

# Sidebar for API Key (using a simple placeholder/reminder)
st.sidebar.markdown(
    "1. **Use only 5 times:** It's just a prototype on free tier"
)
st.sidebar.markdown(
    "2. **Upload Syllabus:** Upload your PDF/DOCX/TXT math syllabus below"
)

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Syllabus Document (PDF, DOCX, TXT)",
    type=['pdf', 'docx', 'txt'],
    key="file_uploader"
)

# Check for file and process if a new file is uploaded
if uploaded_file and uploaded_file.name != st.session_state.get('file_name'):
    if is_client_configured:
        upload_syllabus_to_rag(uploaded_file)
    else:
        st.error("Cannot proceed. Please ensure your GEMINI_API_KEY is configured correctly.")

# Display Current Status
if st.session_state.file_search_store_name:
    st.sidebar.success(f"Syllabus: **{st.session_state.file_name}** is READY.")
else:
    st.sidebar.warning("Please upload your syllabus to begin.")

st.sidebar.markdown("---")

# Custom Instructions Section with limit
st.sidebar.subheader("🎯 Custom Instructions (Max 3)")
st.sidebar.markdown("Select up to 3 preferences:")

# Define available instruction options
instruction_options = {
    "Use simple language": "Always use simple, easy-to-understand language without complex jargon.",
    "Include real-world examples": "Always provide real-world examples and practical scenarios.",
    "Focus on coding applications": "Emphasize how to implement concepts in Python or programming.",
    "Step-by-step breakdown": "Break down explanations into very detailed step-by-step instructions.",
    "Visual explanations": "Describe concepts in visual terms (diagrams, graphs) when possible.",
    "Connect to Data Science": "Always relate concepts to data science and machine learning applications.",
}

# Use multiselect with max_selections parameter
selected_instructions = st.sidebar.multiselect(
    "Choose your preferences:",
    options=list(instruction_options.keys()),
    default=st.session_state.selected_instructions,
    max_selections=3,
    help="You can select up to 3 custom instructions to personalize the tutor's responses.",
    label_visibility="collapsed"
)

# Update session state
st.session_state.selected_instructions = selected_instructions

# Show active instructions
if selected_instructions:
    st.sidebar.success(f"✅ {len(selected_instructions)}/3 instructions active")
    with st.sidebar.expander("View active instructions"):
        for instruction in selected_instructions:
            st.markdown(f"- {instruction}")

st.sidebar.markdown("---")

if st.sidebar.button("🗑️ Clear Indexed Syllabus & Chat"):
    clean_up_store()
    st.rerun()

# --- Main Chat Interface ---

# Display chat messages from history on app rerun
for role, text in st.session_state.chat_history:
    st.chat_message(role).markdown(text)

# Accept user input
user_message_count = sum(1 for role, _ in st.session_state.chat_history if role == "user")

if user_message_count >= 5:
    st.chat_input("You have reached the 5-message limit. Please clear the chat to start over.", disabled=True)
    if "limit_reached_message" not in st.session_state:
        st.warning("You have reached the 5-message limit for this prototype. Please use the 'Clear Indexed Syllabus & Chat' button in the sidebar to start a new conversation.")
        st.session_state.limit_reached_message = True
else:
    if prompt := st.chat_input("Ask a question about your syllabus (e.g., 'Explain the concept of Eigenvalues')"):
        # Add user message to chat history and display
        st.session_state.chat_history.append(("user", prompt))
        st.chat_message("user").markdown(prompt)

        if not st.session_state.file_search_store_name:
            st.chat_message("assistant").error("Please upload and index your syllabus first.")
        else:
            # Generate the response
            with st.chat_message("assistant"):
                with st.spinner("Thinking... Retrieving context from syllabus..."):
                    file_name = st.session_state.file_search_store_name
                    # Pass the full instruction text to the function
                    instruction_texts = [instruction_options[key] for key in selected_instructions]
                    response = generate_rag_response(prompt, file_name, instruction_texts)
                    
                    if response and response.text:
                        # Display the response text
                        st.markdown(response.text)
                        st.session_state.chat_history.append(("assistant", response.text))
                    else:
                        st.error("Sorry, I couldn't generate a response based on your syllabus.")
