import os
from openai import OpenAI
import streamlit as st

st.set_page_config(
    page_title="Chatbot",
    page_icon="💬",
    layout="wide"
)
st.title("💊 Healthcare & 🥗 Food Chatbot")

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
    if st.button("New Chat"):
        st.session_state.messages = []
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_prompt = None
if regular_prompt := st.chat_input("Bạn cần tư vấn gì?"):
    user_prompt = regular_prompt

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.chat_message("assistant"):
        messages_for_api = [msg.copy() for msg in st.session_state.messages]

        for message in messages_for_api:
            if message["role"] == "user":
                message["content"] = f"{message['content']}"
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