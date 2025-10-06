import streamlit as st
import json
from config import GROQ_API_KEY, EMBEDDING_MODEL_NAME
from groq import Groq
from datasets import Dataset
import time

# LangChain components for RAGAS configuration
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# RAGAS and its metrics
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextRecall,
    ContextPrecision,
)

#custom modules
from document_processing import process_documents
from index import FAISSIndex
from config import GROQ_API_KEY

st.set_page_config(page_title="Product Comparison RAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "chat_name" not in st.session_state:
    st.session_state.chat_name = ""


#Groq Client and RAG Components
client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_and_index():
    """Loads documents, creates chunks, and builds a FAISS index."""
    folder_path = "docs"
    chunks = process_documents(folder_path)
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)    
    index = FAISSIndex(embedding_model=embedder)
    index.build_index(chunks)
    return index, embedder

index, embedder = load_and_index()

#Sidebar for Control, History, and Evaluation
with st.sidebar:
    st.title("Control Panel")

    st.header("Chat Management")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.chat_name = ""
        if "ragas_result" in st.session_state:
            del st.session_state.ragas_result
        st.rerun()

    # Chat History and Saving
    chat_name_input = st.text_input("Save current chat as:", value=st.session_state.get("chat_name", ""))
    if st.button("Save Chat"):
        if chat_name_input:
            st.session_state.chat_history[chat_name_input] = st.session_state.messages.copy()
            st.session_state.chat_name = chat_name_input
            st.success(f"Chat '{chat_name_input}' saved!")
        else:
            st.warning("Please enter a name to save the chat.")

    chat_keys = list(st.session_state.chat_history.keys())
    if chat_keys:
        selected_chat = st.selectbox("Load saved chat:", [""] + chat_keys)
        if st.button("Load Selected Chat"):
            if selected_chat:
                st.session_state.messages = st.session_state.chat_history[selected_chat]
                st.session_state.chat_name = selected_chat
                if "ragas_result" in st.session_state:
                    del st.session_state.ragas_result
                st.rerun()

    st.markdown("---")
    st.header("Evaluate with RAGAS")
    st.markdown("""
        Click to run a comprehensive evaluation using your `eval_data.json` file.
    """)

    if st.button("Run RAGAS Evaluation"):
        with st.spinner("Running RAGAS Evaluation... Please wait."):
            try:
                # 1. Load evaluation data
                with open("src/eval_data.json", "r") as f:
                    eval_data = json.load(f)
                
                eval_questions = [item["query"] for item in eval_data]
                eval_ground_truths = [item["ground_truth_answer"] for item in eval_data]

                # 2. Generate answers and retrieve contexts
                answers = []
                contexts = []
                for question in eval_questions:
                    retrieved_docs = index.search(question, k=3)
                    retrieved_contexts = [doc.page_content for doc in retrieved_docs]
                    context_str = "\n\n".join(retrieved_contexts)
                    contexts.append(retrieved_contexts)

                    full_prompt = [
                        {"role": "system", "content": "You are a helpful product comparison assistant."},
                        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion:\n{question}"}
                    ]
                    response = client.chat.completions.create(messages=full_prompt, model="llama-3.1-8b-instant")
                    reply = response.choices[0].message.content or ""
                    answers.append(reply)
                    time.sleep(3)

                # 3. Structure data for evaluation
                data_samples = {"question": eval_questions, "answer": answers, "contexts": contexts, "ground_truth": eval_ground_truths}
                dataset = Dataset.from_dict(data_samples)

                # 4. Defining LLM and Embedding models for RAGAS
                groq_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
                
                # 5. Initializing metrics plainly
                metrics = [
                    Faithfulness(),
                    AnswerCorrectness(),
                    ContextRecall(),
                    ContextPrecision(),
                ]

                # 6. Running evaluation, passing models to the evaluate call
                result = evaluate(
                    dataset=dataset, 
                    metrics=metrics,
                    llm=groq_llm,
                    embeddings=embedder
                )
                st.session_state.ragas_result = result.to_pandas()

            except Exception as e:
                st.error(f"RAGAS Evaluation Error: {e}", icon="🚨")

    # Display the stored RAGAS result if it exists
    if "ragas_result" in st.session_state:
        st.markdown("---")
        st.subheader("RAGAS Evaluation Results")
        st.dataframe(st.session_state.ragas_result)

# Main Chat Interface
st.title("Product Comparison RAG Assistant")
st.write("Ask a question about product comparisons to start a conversation.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("e.g., 'Compare the display quality of the iPhone 15 Pro and Galaxy S24 Ultra'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.spinner("Thinking..."):
        retrieved_docs = index.search(prompt, k=6)
        retrieved_context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt_messages = [
            {"role": "system", "content": "You are a helpful product comparison assistant. "
            "Use the provided context and conversation history to answer the user's query precisely. "
            "If the context does not contain the answer, say that you don't have enough information."},
            {"role": "user", "content": f"Context:\n{retrieved_context_str}\n\nQuestion:\n{prompt}"}
        ]
        
        history = st.session_state.messages[:-1]
        full_prompt = history + prompt_messages

        try:
            response = client.chat.completions.create(messages=full_prompt, model="llama-3.1-8b-instant")
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"An error occurred: {e}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").markdown(reply)

        with st.expander("Show Retrieved Context"):
            st.markdown(retrieved_context_str)
        st.rerun()