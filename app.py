import streamlit as st
from src.data_converter import dataconverter
from src.data_ingestion import ingest_data
from src.retrieval_generation import generation

# Cache vector store creation so it runs once
@st.cache_resource
def load_vector_store():
    return ingest_data("connect")

# Cache chain creation, but ignore vstore hashing
@st.cache_resource
def load_chain(_vstore):
    return generation(_vstore)

# Load once
vstore = load_vector_store()
chain = load_chain(vstore)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat UI
st.title("RAG Chatbot")

user_input = st.chat_input("Type here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = chain.invoke(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
