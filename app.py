import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# Streamlit Configuration
st.set_page_config(
    page_title="LaTeX Expert Chat, by LlamaIndex",
    page_icon="./assets/icon.jpg",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

openai.api_key = st.secrets.openai_key  # API Key Setup

# Title and Info
st.title("Chat about LaTeX, powered by LlamaIndex 💬🦙")
st.info(
    "This application is only for learning purpose and based on [LaTeX documentation](https://texdoc.org/serve/latex2e.pdf/0).",
    icon="📃",
)

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about LaTeX!",
        }
    ]


# Load data and initialize LlamaIndex
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(
        text="Loading and indexing the LaTeX docs - hang tight! This should take 1-2 minutes."
    ):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        Settings.llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,
            system_prompt="""
                You are a knowledgeable LaTeX expert. Your job is to assist users with their LaTeX-related questions by providing accurate, concise, and informative answers. Ensure your responses are technically correct and based on factual information from LaTeX documentation and best practices.

                Guidelines:
                - Focus on LaTeX topics only.
                - Provide step-by-step instructions if applicable.
                - Use clear and simple language, avoiding jargon unless it's necessary.
                - Offer examples where possible to illustrate your points.
                - Remain polite, helpful, and patient in your responses.
                - If you don't know the answer, suggest checking official LaTeX documentation or community forums for more information.

                """,
        )

        index = VectorStoreIndex.from_documents(docs)
        return index


index = load_data()

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
