# import streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import TextLoader
# from langchain.chains import RetrievalQA
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

# # Load documents
# @st.cache_resource
# def load_documents(file_path):
#     loader = TextLoader(file_path)
#     documents = loader.load()
#     return documents

# # Split documents into chunks
# @st.cache_resource
# def split_texts(_documents):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
#     return text_splitter.split_documents(_documents)

# # Create vector store
# @st.cache_resource
# def create_vector_store(_texts):
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model='models/embedding-001',
#         google_api_key='AIzaSyAUH70gKFSmR52QAbZq4fJFM3WSbTYCHp8',
#         task_type="retrieval_query"
#     )
#     vectordb = Chroma.from_documents(documents=_texts, embedding=embeddings)
#     return vectordb

# # Set up safety settings
# safety_settings = {
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
# }

# # Prompt template
# prompt_template = """
# ## Safety and Respect Come First!

# You are programmed to be a helpful and harmless AI. You will not answer requests that promote:

# * **Harassment or Bullying**
# * **Hate Speech**
# * **Violence or Harm**
# * **Misinformation and Falsehoods**

# **How to Use You:**
# 1. Provide context.
# 2. Ask your question.

# **Response to violations:** "I'm here to assist with safe and respectful interactions..."

# Context: \n {context}
# Question: \n {question}
# Answer:
# """
# prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# # Set up LLM
# def get_chat_model():
#     return ChatGoogleGenerativeAI(
#         model="gemini-1.5-pro",
#         google_api_key='AIzaSyAUH70gKFSmR52QAbZq4fJFM3WSbTYCHp8',
#         temperature=0.7,
#         safety_settings=safety_settings
#     )

# # Streamlit UI
# st.title("📚 Smart Q&A Chatbot")
# st.write("Ask anything based on the uploaded document!")

# uploaded_file = st.file_uploader("Upload a text file", type="txt")

# if uploaded_file:
#     with open("example.txt", "wb") as f:
#         f.write(uploaded_file.read())

#     documents = load_documents("example.txt")
#     texts = split_texts(documents)
#     vectordb = create_vector_store(texts)

#     chat_model = get_chat_model()

#     retriever = MultiQueryRetriever.from_llm(
#         retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
#         llm=chat_model
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=chat_model,
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type="stuff",
#         chain_type_kwargs={"prompt": prompt}
#     )

#     question = st.text_input("Ask a question about the content:")
#     if question:
#         response = qa_chain.invoke({"query": question})
#         st.markdown("### ✅ Answer:")
#         st.write(response['result'])

#         with st.expander("📄 Source Documents"):
#             for doc in response['source_documents']:
#                 st.text(doc.page_content)



import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

# Load documents
@st.cache_resource
def load_documents(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

# Split documents into chunks
@st.cache_resource
def split_texts(_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(_documents)

# Create vector store
@st.cache_resource
def create_vector_store(_texts):
    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key='AIzaSyAUH70gKFSmR52QAbZq4fJFM3WSbTYCHp8',
        task_type="retrieval_query"
    )
    vectordb = Chroma.from_documents(documents=_texts, embedding=embeddings)
    return vectordb

# Set up safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

# Prompt template
prompt_template = """
## Safety and Respect Come First!

You are programmed to be a helpful and harmless AI. You will not answer requests that promote:

* **Harassment or Bullying**
* **Hate Speech**
* **Violence or Harm**
* **Misinformation and Falsehoods**

**How to Use You:**
1. Provide context.
2. Ask your question.

**Response to violations:** "I'm here to assist with safe and respectful interactions..."

Context: \n {context}
Question: \n {question}
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Set up LLM
def get_chat_model():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key='AIzaSyAUH70gKFSmR52QAbZq4fJFM3WSbTYCHp8',
        temperature=0.7,
        safety_settings=safety_settings
    )

# Streamlit UI
st.title("📚 LLM CHAT bot trained on Story book")
st.write("Ask anything about the existing and General LLM queries")

# Automatically use the existing file
FILE_PATH = "example.txt"

documents = load_documents(FILE_PATH)
texts = split_texts(documents)
vectordb = create_vector_store(texts)
chat_model = get_chat_model()

retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    llm=chat_model
)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

question = st.text_input("Ask a question about the content:")
if question:
    response = qa_chain.invoke({"query": question})
    st.markdown("### ✅ Answer:")
    st.write(response['result'])

    with st.expander("📄 Source Documents"):
        for doc in response['source_documents']:
            st.text(doc.page_content)
