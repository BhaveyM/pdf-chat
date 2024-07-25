import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import shutil

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="PDF Chat", layout="wide")
st.title("PDF Chat with Context")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectors' not in st.session_state:
    st.session_state.vectors = None

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context and chat history.
    Please provide the most accurate and concise response based on the question.
    
    Chat History:
    {chat_history}
    
    Context:
    {context}
    
    Question: {input}
    """
)

# Function to process and embed documents
def vector_embedding():
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.loader = PyPDFDirectoryLoader("./Docs")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Create a directory to store uploaded files temporarily
os.makedirs("Docs", exist_ok=True)

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload PDF Files")
    uploaded_files = st.file_uploader("Upload your PDF files", type=['pdf'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join("Docs", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded successfully!")

        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                vector_embedding()
            st.success("Vector Store DB Is Ready")

# Main chat interface
st.subheader("Chat with your PDFs")

# Display chat history
for i, (question, answer) in enumerate(st.session_state.chat_history):
    with st.chat_message(f"user"):
        st.write(question)
    with st.chat_message(f"assistant"):
        st.write(answer)

# User input
user_question = st.chat_input("Ask a question about your documents:")

if user_question:
    if st.session_state.vectors is None:
        st.error("Please upload and process documents first.")
    else:
        with st.spinner("Thinking..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            chat_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
            
            start = time.process_time()
            try:
                response = retrieval_chain.invoke({
                    'input': user_question,
                    'chat_history': chat_history
                })
                
                process_time = time.process_time() - start

                # Display user question
                with st.chat_message("user"):
                    st.write(user_question)

                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(response['answer'])
                    st.caption(f"Response time: {process_time:.2f} seconds")

                # Add to chat history
                st.session_state.chat_history.append((user_question, response['answer']))

                # Show relevant document chunks
                with st.expander("Relevant Document Chunks"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"Chunk {i + 1}:")
                        st.write(doc.page_content)
                        st.write("---")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please try rephrasing your question or check if the documents are processed correctly.")

# Clean up the Docs folder after processing
if st.sidebar.button("Clear uploaded documents"):
    shutil.rmtree("Docs")
    os.makedirs("Docs", exist_ok=True)
    st.session_state.vectors = None
    st.success("Uploaded documents cleared. You can now upload new documents.")