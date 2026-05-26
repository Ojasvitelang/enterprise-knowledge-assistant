"""
Enterprise Knowledge Assistant - Streamlit Frontend

This is a chat-style web interface for the RAG-based knowledge assistant.
It connects to the FastAPI backend to query the knowledge base.

Features:
- Clean chat interface for asking questions
- Displays AI-generated answers
- Shows source documents used for the answer
- Maintains chat history during the session
- Handles errors gracefully

To run:
    cd frontend
    streamlit run streamlit_app.py

Make sure the FastAPI backend is running on http://localhost:8000
"""
import streamlit as st
import requests
from typing import Optional, Dict, Any


# =============================================================================
# CONFIGURATION
# =============================================================================

# Backend API URL
API_BASE_URL = "http://localhost:8000"
QUERY_ENDPOINT = f"{API_BASE_URL}/chat/query"
STATUS_ENDPOINT = f"{API_BASE_URL}/chat/status"

# Page configuration
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)


# =============================================================================
# API FUNCTIONS
# =============================================================================

def check_backend_status() -> Dict[str, Any]:
    """
    Check if the backend API is running and get its status.

    Returns:
        Dictionary with status information or error details
    """
    try:
        response = requests.get(STATUS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return {"connected": True, **response.json()}
        else:
            return {"connected": False, "error": f"Status code: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"connected": False, "error": "Cannot connect to backend. Is the server running?"}
    except requests.exceptions.Timeout:
        return {"connected": False, "error": "Backend request timed out"}
    except Exception as e:
        return {"connected": False, "error": str(e)}


def query_knowledge_base(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Send a question to the backend API and get the answer.

    This function:
    1. Sends POST request to /chat/query endpoint
    2. Includes the question in the request body
    3. Returns the answer and sources from the response

    Args:
        question: The user's question
        top_k: Number of context chunks to retrieve

    Returns:
        Dictionary with 'success', 'answer', 'sources', or 'error'
    """
    try:
        # ---------------------------------------------------------------------
        # Send POST request to the backend API
        # ---------------------------------------------------------------------
        response = requests.post(
            QUERY_ENDPOINT,
            json={"question": question, "top_k": top_k},
            headers={"Content-Type": "application/json"},
            timeout=120  # Allow time for LLM generation
        )

        # ---------------------------------------------------------------------
        # Handle successful response
        # ---------------------------------------------------------------------
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "answer": data.get("answer", "No answer received"),
                "sources": data.get("sources", [])
            }

        # ---------------------------------------------------------------------
        # Handle error responses
        # ---------------------------------------------------------------------
        elif response.status_code == 400:
            return {"success": False, "error": "Invalid question. Please try again."}
        elif response.status_code == 503:
            error_detail = response.json().get("detail", "Service unavailable")
            return {"success": False, "error": f"Service unavailable: {error_detail}"}
        else:
            return {"success": False, "error": f"Server error (Status {response.status_code})"}

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to backend. Please make sure the FastAPI server is running on http://localhost:8000"
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out. The server might be overloaded or Ollama might be slow."
        }
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
# Streamlit reruns the script on every interaction
# Session state persists data across reruns

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "backend_status" not in st.session_state:
    st.session_state.backend_status = None


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("🎓 About")
    st.markdown("""
    **Enterprise Knowledge Assistant**

    An AI-powered assistant that answers questions
    based on your organization's documents.

    ---

    **How it works:**
    1. Your question is analyzed
    2. Relevant documents are retrieved
    3. AI generates a grounded answer

    ---
    """)

    # Backend status check
    st.subheader("🔌 Backend Status")

    if st.button("Check Connection"):
        st.session_state.backend_status = check_backend_status()

    status = st.session_state.backend_status
    if status:
        if status.get("connected"):
            st.success("✅ Connected")
            st.caption(f"Documents indexed: {status.get('documents_indexed', 'N/A')}")
            st.caption(f"LLM: {status.get('llm_model', 'N/A')}")
        else:
            st.error(f"❌ {status.get('error', 'Not connected')}")
    else:
        st.info("Click 'Check Connection' to verify backend")

    st.markdown("---")

    # Settings
    st.subheader("⚙️ Settings")
    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=5,
        value=3,
        help="More sources = more context but slower response"
    )

    # Clear chat button
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header
st.title("🎓 Enterprise Knowledge Assistant")
st.markdown("*Ask questions about your organization's documents*")
st.markdown("---")


# -----------------------------------------------------------------------------
# Chat History Display
# -----------------------------------------------------------------------------
# Display previous Q&A pairs from session state

if st.session_state.chat_history:
    for i, chat in enumerate(st.session_state.chat_history):
        # User question
        with st.chat_message("user"):
            st.write(chat["question"])

        # Assistant response
        with st.chat_message("assistant"):
            st.write(chat["answer"])

            # Show sources in an expander
            if chat.get("sources"):
                with st.expander("📄 Sources"):
                    for source in chat["sources"]:
                        st.caption(f"• {source}")


# -----------------------------------------------------------------------------
# Question Input
# -----------------------------------------------------------------------------

# Create a form for the question input
# Using a form prevents the page from rerunning on every keystroke
with st.form(key="question_form", clear_on_submit=True):
    # Question input
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What is the attendance policy?",
        label_visibility="collapsed"
    )

    # Submit button
    submit_button = st.form_submit_button(
        label="Ask Question",
        use_container_width=True
    )


# -----------------------------------------------------------------------------
# Process Question
# -----------------------------------------------------------------------------

if submit_button and question:
    # Validate input
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        # Show loading spinner while processing
        with st.spinner("🔍 Searching knowledge base and generating answer..."):
            # Send question to backend API
            result = query_knowledge_base(question.strip(), top_k=top_k)

        # Handle the response
        if result["success"]:
            # -------------------------------------------------------------
            # SUCCESS: Display the answer
            # -------------------------------------------------------------

            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"]
            })

            # Rerun to show the new message in chat history
            st.rerun()

        else:
            # -------------------------------------------------------------
            # ERROR: Display error message
            # -------------------------------------------------------------
            st.error(f"❌ {result['error']}")

            # Provide helpful suggestions based on error type
            if "Cannot connect" in result["error"]:
                st.info("""
                **To start the backend server:**
                ```bash
                cd enterprise-knowledge-assistant/backend
                uvicorn app.main:app --reload
                ```
                """)
            elif "Ollama" in result["error"].lower():
                st.info("""
                **Make sure Ollama is running:**
                ```bash
                ollama serve
                ```
                """)


# -----------------------------------------------------------------------------
# Empty State
# -----------------------------------------------------------------------------

if not st.session_state.chat_history and not submit_button:
    st.markdown(
        """
        <div style="text-align: center; padding: 50px; color: #666;">
            <h3>👋 Welcome!</h3>
            <p>Ask a question about your organization's documents.</p>
            <p style="font-size: 14px;">
                Example: "What is the attendance policy?" or "How do I learn Python?"
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Powered by RAG (Retrieval-Augmented Generation) | "
    "Ollama (phi3:mini) | ChromaDB | FastAPI"
)
