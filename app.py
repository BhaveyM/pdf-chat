# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import os
# load_dotenv()

# ## load the GROQ And OpenAI API KEY 
# groq_api_key=os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

# st.title("Document Q&A")

# llm=ChatGroq(groq_api_key=groq_api_key,
#              model_name="Llama3-8b-8192")

# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )

# def vector_embedding():

#     if "vectors" not in st.session_state:

#         st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#         st.session_state.loader=PyPDFDirectoryLoader("./Docs") ## Data Ingestion
#         st.session_state.docs=st.session_state.loader.load() ## Document Loading
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings





# prompt1=st.text_input("Enter Your Question From Doduments")


# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# import time



# if prompt1:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=st.session_state.vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever,document_chain)
#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':prompt1})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")


import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import random
import logging
from io import StringIO, BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
import numpy as np

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
log_stream = StringIO()
handler = logging.StreamHandler(log_stream)
logger.addHandler(handler)

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0.1)

# Streamlit layout configuration
st.set_page_config(page_title="Document & CSV Chat", layout='wide')

st.title("PDF & CSV Chatbot")


# Upload files
st.sidebar.header("Upload Files")
uploaded_files = st.sidebar.file_uploader("Upload your PDF or CSV files", type=['pdf', 'csv'], accept_multiple_files=True)

# Define a simple document class
class SimpleDocument:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata else {}

# Function to chat with CSV
def chat_with_csv(df, query):
    try:
        logger.info("Initializing AI model...")
        pandas_ai = SmartDataframe(df, config={"llm": llm})
        logger.info("Processing query with AI model...")
        result = pandas_ai.chat(query)
        logger.info("Query processed successfully.")
        
        # Convert numpy types to Python native types
        if isinstance(result, np.generic):
            result = result.item()
        
        # Ensure result is string
        return str(result)
    except Exception as e:
        error_message = f"Error: {str(e)}"
        logger.error(error_message)
        return error_message

# Function to generate sample queries for CSV
def generate_sample_queries(columns):
    sample_queries = [
        f"What is the total count of {random.choice(columns)}?",
        f"Show the average value of {random.choice(columns)} over time.",
        f"Which {random.choice(columns)} has the highest value?",
        f"Compare the {random.choice(columns)} between different categories."
    ]
    return sample_queries

# Function to handle PDF embeddings
def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        docs = []
        for file in uploaded_files:
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                docs.append(SimpleDocument(text, metadata={"page_num": page_num, "file_name": file.name}))
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

# Initialize prompt for document Q&A
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Interface to handle PDF and CSV files
if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            with st.spinner(f"Processing {file.name}..."):
                vector_embedding([file])
            st.success(f"{file.name} uploaded successfully. Vector store DB is ready.")
            st.subheader("Ask a Question from the PDF")
            prompt1 = st.text_input("Enter your question:")
            if prompt1:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                response = retrieval_chain.invoke({'input': prompt1})
                st.write(response['answer'])
                with st.expander("Document Similarity Search"):
                    for chunk in response["context"]:
                        st.write(chunk.page_content)
                        st.write("--------------------------------")
        elif file.type == "text/csv":
            with st.spinner(f"Loading {file.name}..."):
                data = pd.read_csv(file)
            st.success(f"{file.name} uploaded successfully.")
            st.subheader("Data Summary")
            st.write(data.describe())
            st.subheader("Data Preview")
            st.dataframe(data.head(), use_container_width=True)
            if st.sidebar.checkbox("Show Basic Statistics"):
                st.subheader("Basic Statistics")
                st.write(data.describe())
            if 'sample_queries' not in st.session_state:
                st.session_state.sample_queries = generate_sample_queries(data.columns)
            sample_queries = st.session_state.sample_queries
            st.sidebar.subheader("Sample Queries")
            selected_sample_query = st.sidebar.radio("Select a sample query to execute:", sample_queries)
            st.subheader("Chat with CSV")
            input_text = st.text_area("Enter your query or select a sample query from the sidebar:")
            if st.button("Chat with CSV") or st.sidebar.button("Execute Selected Sample Query"):
                query = input_text if input_text else selected_sample_query
                st.info("Your Query: " + query)
                result = chat_with_csv(data, query)
                if isinstance(result, str) and result.startswith("Error:"):
                    st.error(result)
                else:
                    st.success(result)
                st.session_state.sample_queries = generate_sample_queries(data.columns)

# Additional instructions and log download
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload one or more PDF or CSV files.
2. For PDFs, ask questions about the document.
3. For CSVs, view basic statistics and interact with the data using AI.
4. Enter your query or select a sample query from the sidebar.
""")
st.sidebar.header("Download Logs")
if st.sidebar.button("Download Logs"):
    log_contents = log_stream.getvalue()
    st.sidebar.download_button(label="Download Logs", data=log_contents, file_name='logs.txt', mime='text/plain')

