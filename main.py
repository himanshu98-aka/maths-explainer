import streamlit as st
import os
import time
import google.generativeai as genai
import tempfile
import uuid
import json
from gemini_api import generate_with_failover, get_api_keys

# --- Configuration and Initialization ---


# Use st.set_page_config for better display
st.set_page_config(
    page_title="Personalized Math Explainer ",
    page_icon="🎧",
    layout="centered"
)

def load_css(file_name):
    """Loads a CSS file and injects it into the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Use st.cache_resource for objects that should be created only once (like the API client)
@st.cache_resource
def configure_gemini():
    """Configures the Gemini client for file operations using the first available key."""
    # API keys are loaded on-demand from gemini_api.py
    api_keys = get_api_keys()
    if not api_keys:
        st.error("Error: GEMINI_API_KEYs not found. Please configure them in your .env file or Streamlit secrets.")
        return False

    try:
        # Configure genai for file operations with the first key.
        # Content generation will use the failover logic in gemini_api.py
        genai.configure(api_key=api_keys[0])
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
if "extracted_topics" not in st.session_state:
    st.session_state.extracted_topics = None
if "limit_unlocked" not in st.session_state:
    st.session_state.limit_unlocked = False


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
            st.session_state.limit_unlocked = False # Reset cheat code
            st.session_state.extracted_topics = None
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
                # Extract topics from the newly uploaded syllabus
                extract_topics_from_syllabus(uploaded_gemini_file.name)

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


def extract_topics_from_syllabus(file_name: str):
    """
    Uses Gemini to extract topics and subtopics from the syllabus file and stores them in session state.
    Enhanced for handwritten notes recognition.
    """
    with st.spinner("Analyzing syllabus to extract topics..."):
        extraction_prompt = """
You are an expert syllabus analyzer specializing in educational document structure. Your task is to carefully read and understand the provided syllabus document (which may be handwritten or typed) and extract its hierarchical structure.

IMPORTANT INSTRUCTIONS:

1. **Identify Main Topics/Units**: Look for:
   - Chapter headings (e.g., "Chapter 1", "Unit 1")
   - Bold or underlined section titles
   - Major headings with numbers (e.g., "1.", "2.", "I.", "II.")
   - Broad subject areas (e.g., "Linear Algebra", "Calculus", "Statistics")
   
2. **Identify Subtopics**: Look for:
   - Content listed under main topics
   - Bullet points or numbered lists
   - Sub-headings with decimal notation (e.g., "1.1", "1.2")
   - Specific concepts, theorems, or methods
   - Topics with smaller text or indentation

3. **What to IGNORE**: 
   - Page numbers
   - Random numbers or equations (like "x=23")
   - Date stamps
   - Handwritten notes in margins
   - Instructor names or contact information
   - Course codes or administrative details

4. **Handling Handwritten Text**:
   - If text is unclear, make your best interpretation
   - Focus on recognizable mathematical terms and concepts
   - Look for structural patterns (indentation, numbering, underlining)
   - Ignore scribbles or unclear annotations

5. **Output Format**: 
   Return ONLY a valid JSON object with this exact structure:
   {
     "topics": [
       {
         "topic": "Name of Main Topic/Unit",
         "subtopics": [
           "Subtopic 1",
           "Subtopic 2",
           "Subtopic 3"
         ]
       }
     ]
   }

EXAMPLES OF GOOD EXTRACTION:

From text like:
"Unit 1: Matrices
- Types of Matrices
- Matrix Operations
- Determinants
x = 23 (example problem)
Page 5"

Should extract:
{
  "topics": [
    {
      "topic": "Matrices",
      "subtopics": [
        "Types of Matrices",
        "Matrix Operations",
        "Determinants"
      ]
    }
  ]
}

From text like:
"Chapter 3 - Probability
3.1 Basic Probability
3.2 Conditional Probability  
3.3 Bayes Theorem"

Should extract:
{
  "topics": [
    {
      "topic": "Probability",
      "subtopics": [
        "Basic Probability",
        "Conditional Probability",
        "Bayes Theorem"
      ]
    }
  ]
}

CRITICAL: 
- Return ONLY the JSON object
- No explanatory text before or after
- Ensure valid JSON formatting
- Each topic must have at least an empty subtopics array []
- Focus on educational content, ignore administrative text
        """
        
       
        response = generate_with_failover(
            prompt=extraction_prompt,
            model_name="gemini-2.5-flash-lite",  # Better for OCR and handwriting
            file_name=file_name
        )

        if response and not response.startswith("ERROR:"):
            try:
                # Clean the response to ensure it's valid JSON
                json_str = response.strip()
                
                # Remove markdown code blocks if present
                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                if json_str.startswith("```"):
                    json_str = json_str[3:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                
                json_str = json_str.strip()
                
                # Parse JSON
                data = json.loads(json_str)
                
                if "topics" in data and isinstance(data["topics"], list):
                    # Validate and clean the extracted data
                    cleaned_topics = []
                    for topic_item in data["topics"]:
                        if isinstance(topic_item, dict) and "topic" in topic_item:
                            # Clean topic name - remove page numbers, course codes, etc.
                            topic_name = topic_item["topic"].strip()
                            
                            # Skip if it looks like a page number, equation, or administrative text
                            if (topic_name.lower().startswith("page") or 
                                topic_name.isdigit() or 
                                "=" in topic_name or
                                len(topic_name) < 3):
                                continue
                            
                            # Clean subtopics
                            subtopics = []
                            if "subtopics" in topic_item and isinstance(topic_item["subtopics"], list):
                                for subtopic in topic_item["subtopics"]:
                                    subtopic_clean = str(subtopic).strip()
                                    # Filter out invalid subtopics
                                    if (not subtopic_clean.isdigit() and 
                                        "=" not in subtopic_clean and
                                        len(subtopic_clean) >= 3 and
                                        not subtopic_clean.lower().startswith("page")):
                                        subtopics.append(subtopic_clean)
                            
                            cleaned_topics.append({
                                "topic": topic_name,
                                "subtopics": subtopics
                            })
                    
                    if cleaned_topics:
                        st.session_state.extracted_topics = cleaned_topics
                        st.sidebar.success(f"✅ Extracted {len(cleaned_topics)} topics!")
                    else:
                        st.sidebar.warning("No valid topics found in the document.")
                        st.session_state.extracted_topics = None
                else:
                    st.sidebar.warning("Could not find topics in the expected format.")
                    st.session_state.extracted_topics = None
                    
            except json.JSONDecodeError as e:
                st.sidebar.warning(f"Failed to parse topics from syllabus. Error: {str(e)}")
                st.session_state.extracted_topics = None
        else:
            st.sidebar.warning("Could not extract topics from syllabus. Please try uploading again.")
            st.session_state.extracted_topics = None


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
        "3. **Merits and Demerits/Caveats:** The advantages and limitations of the concept. "
        "If the context does not contain the answer, state that you cannot find the information in the syllabus."
    )
    
    # Append selected custom instructions
    if selected_instructions:
        instructions_text = "\n".join([f"- {instruction}" for instruction in selected_instructions])
        system_instruction = base_system_instruction + f"\n\nADDITIONAL USER PREFERENCES:\n{instructions_text}"
    else:
        system_instruction = base_system_instruction

    # Use the failover function from gemini_api.py
    response_text = generate_with_failover(
        prompt=prompt,
        model_name="gemini-2.5-flash-lite", # Use a consistent, modern model
        file_name=file_name,
        system_instruction=system_instruction
    )
    return response_text


# --- Streamlit UI Layout ---


st.title(" Personalized Math Explainer")
st.markdown("---")

# Calculate user message count early to use in the sidebar
user_message_count = sum(1 for role, _ in st.session_state.chat_history if role == "user")

st.sidebar.header("Try it:")

# Display chat limit and progress (only if not in unlimited mode)
if not st.session_state.limit_unlocked:
    st.sidebar.markdown(
        "**Chat Limit:** This prototype is limited to 3 questions per session."
    )
    st.sidebar.progress(user_message_count / 3, text=f"{user_message_count}/3 Questions Asked")
else:
    st.sidebar.success("🔓 **Unlimited Mode Active**")
    st.sidebar.markdown(f"Questions Asked: {user_message_count}")
st.sidebar.markdown(
    "**Get Started:** Upload your PDF/DOCX/TXT math syllabus below."
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

# Display extracted topics if they exist
if st.session_state.extracted_topics:
    with st.sidebar.expander("Syllabus Overview", expanded=True):
        for item in st.session_state.extracted_topics:
            if "topic" in item and item["topic"]:
                st.markdown(f"**{item['topic']}**")
                if "subtopics" in item and item["subtopics"]:
                    for subtopic in item["subtopics"]:
                        st.markdown(f"- {subtopic}")


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

# Check if limit is reached (only if not in unlimited mode)
limit_reached = user_message_count >= 3 and not st.session_state.limit_unlocked

if limit_reached:
    st.chat_input("You have reached the 3-message limit. Please clear the chat to start over.", disabled=True)
    if "limit_reached_message" not in st.session_state:
        st.warning("You have reached the 3-message limit for this prototype. Please use the 'Clear Indexed Syllabus & Chat' button in the sidebar to start a new conversation.")
        st.session_state.limit_reached_message = True
else:
    if prompt := st.chat_input("Ask a question about your syllabus (e.g., 'Explain the concept of Eigenvalues')"):
        # Handle the easter egg as a special case first
        if prompt.strip().lower() == "himanshu" and not st.session_state.file_search_store_name:
            st.session_state.limit_unlocked = True
            st.session_state.chat_history.append(("user", prompt))
            st.session_state.chat_history.append(("assistant", "🔓 **Unlimited mode activated!** You can now ask unlimited questions. Please upload your syllabus to get started."))
            st.rerun()
        # Then, handle the case where no file is uploaded for a regular prompt
        elif not st.session_state.file_search_store_name:
            st.chat_message("user").markdown(prompt)
            st.chat_message("assistant").error("Please upload and index your syllabus first.")
        # Finally, handle a regular prompt when a file is present
        else:
            # Add user message to chat history and display
            st.session_state.chat_history.append(("user", prompt))
            st.chat_message("user").markdown(prompt)

            # Generate the response
            with st.chat_message("assistant"):
                with st.spinner("Thinking... Retrieving context from syllabus..."):
                    file_name = st.session_state.file_search_store_name
                    instruction_texts = [instruction_options[key] for key in selected_instructions]
                    response = generate_rag_response(prompt, file_name, instruction_texts)

                    if response and not response.startswith("ERROR:"):
                        st.markdown(response)
                        st.session_state.chat_history.append(("assistant", response))
                    else:
                        st.error(f"Sorry, I couldn't generate a response. The API returned an error: {response}")
