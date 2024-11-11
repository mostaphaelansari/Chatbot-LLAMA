"""
This module is a Streamlit app for a chatbot powered by LLAMA 3.2.
The app features a wide layout and displays a user interface for interacting with the chatbot.
"""

import time
import tempfile
from pathlib import Path
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(page_title='Chatbot LLAMA 3.2', layout='wide')



st.title('💬 Chatbot with LLAMA 3.2')

if "messages" not in st.session_state:
    st.session_state.messages = []


def load_llm():
    """Initialize the LLAMA model"""
    return OllamaLLM(model="llama3.2")

def process_file(uploaded_file):
    """Process uploaded file and return its content"""
    if uploaded_file is not None:
        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    return None

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Settings")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf', 'docx'])
    
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            file_path = process_file(uploaded_file)
            st.success(f"File uploaded: {uploaded_file.name}")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.success("Conversation cleared!")
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This chatbot uses LLAMA 3.2 model to generate responses.
    - Upload documents to discuss their content
    - Have natural conversations
    - Get AI-powered assistance
    """)
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to discuss?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Initialize LLM
            llm = load_llm()
            
            with st.spinner("Thinking..."):
                # Generate response
                response = llm.invoke(prompt)
                
                # Display response with typing effect
                full_response = ""
                for chunk in response:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)
                
                # Final response without cursor
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
# Improved and Professional CSS Styling for Streamlit App

st.markdown("""
<style>
/* Universal Styles */
body {
    font-family: 'Roboto', sans-serif;
    background-color: #f5f5f5;
    color: #333;
    margin: 0;
    padding: 0;
}

/* Sidebar Styling */
.stSidebar {
    background-color: #4a148c;
    color: #fff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.stSidebar .stButton {
    background-color: #7b1fa2;
    color: #fff;
    border: none;
    border-radius: 5px;
    padding: 10px 16px;
    font-size: 1rem;
    transition: background-color 0.3s ease;
}
.stSidebar .stButton:hover {
    background-color: #6a1b9a;
}

/* Header Styling */
h1 {
    font-size: 3rem;
    color: #4a148c;
    margin-bottom: 20px;
}

/* Chat Message Bubbles */
.stChatMessage {
    border-radius: 20px;
    padding: 12px 18px;
    margin: 12px 0;
    max-width: 80%;
    line-height: 1.6;
    font-size: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.stChatMessage.user {
    background-color: #4a148c;
    color: #fff;
    margin-left: auto;
}
.stChatMessage.assistant {
    background-color: #e1bee7;
    color: #4a148c;
    margin-right: auto;
}

/* Typing Indicator Animation */
.stChatMessage .typing {
    display: inline-block;
    background-color: #ccc;
    border-radius: 50%;
    width: 14px;
    height: 14px;
    margin-left: 6px;
    animation: typing 1.5s infinite ease-in-out;
}
@keyframes typing {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.5); }
}

/* Chat Input Styling */
.stChatInput {
    border-radius: 30px;
    padding: 12px 16px;
    width: 100%;
    font-size: 1rem;
    margin-top: 20px;
    border: 1px solid #ccc;
    transition: border-color 0.3s ease;
    box-sizing: border-box;
}
.stChatInput:focus {
    outline: none;
    border-color: #4a148c;
}

/* Chat Messages Container */
.stChatMessages {
    max-height: 450px;
    overflow-y: auto;
    padding: 0 10px;
}

/* General Button Styling */
.stButton {
    background-color: #4a148c;
    color: #fff;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 1rem;
    transition: background-color 0.3s ease;
}
.stButton:hover {
    background-color: #6a1b9a;
}
</style>
""", unsafe_allow_html=True)
