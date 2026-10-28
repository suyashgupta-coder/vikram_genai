
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

load_dotenv()

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Personality Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Personality Chatbot")
st.caption("Chat with Gemini using your favorite personality!")

# -----------------------------
# Personalities
# -----------------------------
personalities = {
    "😂 Funny": "You are a funny chatbot who responds with humor.",
    "😠 Angry": "You are an angry chatbot who responds in an irritated tone, but do not insult or threaten the user.",
    "😢 Sad": "You are a sad chatbot who responds in a melancholic tone.",
    "❤️ Romantic": "You are a romantic chatbot who responds in a warm, affectionate, romantic manner."
}

# -----------------------------
# Initialize Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "personality" not in st.session_state:
    st.session_state.personality = "😂 Funny"

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Chat Settings")

    selected_personality = st.selectbox(
        "Choose your personality",
        list(personalities.keys()),
        index=list(personalities.keys()).index(
            st.session_state.personality
        )
    )

    if selected_personality != st.session_state.personality:
        st.session_state.personality = selected_personality
        st.session_state.messages = []
        st.rerun()

    st.info(
        f"Current personality: {selected_personality}"
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Google Gemini + LangChain")

# -----------------------------
# Initialize Gemini
# -----------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.8
)

# -----------------------------
# Display Chat History
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat Input
# -----------------------------
prompt = st.chat_input("Type your message here...")

if prompt:
    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare conversation messages for LangChain
    conversation = [
        SystemMessage(
            content=personalities[st.session_state.personality]
        )
    ]

    for message in st.session_state.messages:
        if message["role"] == "user":
            conversation.append(
                HumanMessage(content=message["content"])
            )
        elif message["role"] == "assistant":
            conversation.append(
                AIMessage(content=message["content"])
            )

    # Get Gemini response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = model.invoke(conversation)

                # Handle Gemini's response content
                if isinstance(response.content, str):
                    answer = response.content
                else:
                    answer = "\n".join(
                        block["text"]
                        for block in response.content
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                        and "text" in block
                    )

                if not answer:
                    answer = "I couldn't generate a text response. Please try again."

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"Something went wrong: {e}")
