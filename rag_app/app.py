import streamlit as st
from chat_model import ChatBot

# Initialize chatbot instance
API_KEY = st.secrets["OPENAI_API_KEY"]
chat_bot = ChatBot(str(API_KEY))

# Page header with improved styling
st.markdown(
    f"""
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
        <h1 style="color: #4CAF50; font-size: 36px;">Recipe Generator 🍔</h1>
    </div>
    <p style="text-align: center; font-size: 18px;">I am Remy, AI-powered recipe assistant that can help you create delicious recipes based on the ingredients you have. 
    Simply enter your ingredients and preferences, and get personalized cooking suggestions!</p>
    <h3 style="text-align: center; color: #6C757D;">Try one of these sample prompts:</h3>
    <ul style="text-align: center; list-style-type: none; padding-left: 0;">
        <li>👉 "What can I cook with chicken, tomatoes, and garlic?"</li>
        <li>👉 "Give me a vegetarian pasta recipe."</li>
        <li>👉 "How do I bake a chocolate cake?"</li>
    </ul>
    """,
    unsafe_allow_html=True,
)

# Initialize chat history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Function to handle message sending
def send_message():
    user_message = st.session_state.get("user_input", "").strip()
    if user_message:
        # Get the response from the chatbot
        response = chat_bot.get_model_answer(user_message, session_id="default_session")
        # Append user and chatbot messages to the history
        st.session_state["history"].append(("You", user_message))
        st.session_state["history"].append(("Bot", response))
        # Clear the input box
        st.session_state["user_input"] = ""

# Input section
st.text_input(
    "Ask your recipe question:", key="user_input", placeholder="e.g., What can I make with chicken and rice?",
    on_change=send_message, label_visibility="collapsed"
)

# Display chat history
for idx, (user, message) in enumerate(reversed(st.session_state["history"])):
    if user == "You":
        st.markdown(f"<div style='text-align: right;'><b>You:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left;'><b>Remy:</b> {message}</div>", unsafe_allow_html=True)

st.markdown("---")
