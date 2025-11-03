import os
from openai import OpenAI
import streamlit as st

st.set_page_config(
    page_title="Chatbot",
    page_icon="🎯",
    layout="wide"
)
st.title("🎯 Log Analysis Chatbot")

API_KEY = os.getenv("TOKEN", "EMPTY")
BASE_URL = os.getenv("ENDPOINT_URL", "").split("/chat", 1)[0]
MODEL = os.getenv("MODEL", "")

try:
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
except Exception as e:
    st.error(f"Error initializing OpenAI client: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "log_content" not in st.session_state:
    st.session_state.log_content = ""

if "model_name" not in st.session_state:
    if len(MODEL) != 0:
        st.session_state.model_name = MODEL
    else:
        try:
            models = client.models.list()
            st.session_state.model_name = models.data[0].id
        except Exception as e:
            st.error(f"Unable to connect to model. Please check API server. Error: {e}")
            st.stop()

with st.sidebar:
    st.header("Analysis Tools")

    if st.button("New Chat"):
        st.session_state.messages = []
        st.session_state.log_content = ""
        st.rerun()

    st.text_area(
        "Paste the log content here:",
        height=300,
        key="log_content"
    )

    st.markdown("---")
    st.subheader("Tasks")
    summary_button = st.button("✂️ Summarize Log")
    root_cause_button = st.button("🔍 Find Root Cause")
    pattern_button = st.button("🧩 Find Patterns")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_prompt = None
if summary_button:
    _ = st.chat_input("What do you need to analyze in the log?")
    user_prompt = "Please provide a comprehensive summary of this log. Please answer in detail."

elif root_cause_button:
    _ = st.chat_input("What do you need to analyze in the log?")
    user_prompt = "Can you analyze the root cause of the main issues in this log? Please answer in detail."

elif pattern_button:
    _ = st.chat_input("What do you need to analyze in the log?")
    user_prompt = "What patterns can you identify in this log? Please answer in detail."

elif regular_prompt := st.chat_input("What do you need to analyze in the log?"):
    user_prompt = regular_prompt

if user_prompt:
    if not st.session_state.log_content.strip():
        st.warning("Please paste the log content into the sidebar before starting the analysis.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            messages_for_api = [msg.copy() for msg in st.session_state.messages]

            for message in messages_for_api:
                if message["role"] == "user":
                    message["content"] = f"{message['content']}\n\nLog content:\n\n{st.session_state.log_content}"
                    break

            try:
                stream = client.chat.completions.create(
                    model=st.session_state["model_name"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in messages_for_api
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"An error occurred while calling the API: {e}")
                st.session_state.messages.pop()