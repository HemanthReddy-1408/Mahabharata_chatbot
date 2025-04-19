import streamlit as st
import json

# Function to save chat history to a file
def save_chat_history(chat_history, filepath="chat_history_mahabharata.json"):
    """
    Saves the chat history to a JSON file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)
    st.success("Chat history saved successfully!")

# Function to display the chat history
def display_chat_history(chat_history):
    """
    Displays the chat history in the Streamlit app.
    """
    if chat_history:
        st.subheader("🗨️ Chat History")
        for entry in reversed(chat_history):
            st.markdown(f"**🧘 You:** {entry['question']}")
            st.markdown(f"**🤖 Oracle:** {entry['answer']}")
            st.markdown("---")

# Function to initialize chat history in session state
def initialize_chat_history():
    """
    Initializes chat history in the Streamlit session state.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
