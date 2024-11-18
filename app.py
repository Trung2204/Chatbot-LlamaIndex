import llama_parse as lp
import streamlit as st
import openai
import hashlib
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, Settings

# Streamlit Configuration
st.set_page_config(
    page_title="Chat about LaTeX, powered by LlamaIndex",
    page_icon="📄",
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


# Define the Document Class
class Document:
    def __init__(self, content, doc_id):
        self.content = content
        self.doc_id = doc_id
        self.hash = self.compute_hash()

    def get_doc_id(self):
        return self.doc_id

    def get_content(self):
        return self.content

    def compute_hash(self):
        return hashlib.md5(self.content.encode()).hexdigest()


# Load data using LlamaParse
@st.cache_resource(show_spinner=False)
def load_data():
    # Parse the PDF file
    parsed_docs = lp.load_data(input_file="./data/latex2e.pdf")
    documents = [
        Document(content=doc, doc_id=str(i)) for i, doc in enumerate(parsed_docs)
    ]

    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="""You are an expert on LaTeX
        and your job is to answer technical questions.
        Assume that all questions are related to LaTeX.
        Keep your answers technical and based on facts
        - do not hallucinate features.""",
    )
    index = VectorStoreIndex.from_documents(documents)
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
