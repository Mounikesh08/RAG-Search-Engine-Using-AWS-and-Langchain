import json
import os
import sys
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
model_id="amazon.titan-embed-text-v1", client=bedrock
)

# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("Data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000, chunk_overlap=1000
)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

# Claude LLM
def get_claude_llm():
    return Bedrock(
    model_id="anthropic.claude-v2:1",
    client=bedrock,
    model_kwargs={"max_tokens_to_sample": 512},
)

# Llama2 LLM
def get_llama2_llm():
    return Bedrock(
    model_id="meta.llama3-8b-instruct-v1:0",
    client=bedrock,
    model_kwargs={"max_gen_len": 512},
)

# Prompt Template
prompt_template = """
human: Use the following pieces of context to provide a concise answer to the question at the end but use at least 250 words
with detailed explanations. If you don't know the answer, just say you don't know. Do not make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Get Response
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    answer = qa({"query": query})
    return answer["result"]

# Streamlit App
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using AWS Bedrock 🤖")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Menu:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if user_question:
        with st.spinner("Generating answer..."):
            llm = get_claude_llm()  # or use get_llama2_llm()
            vectorstore_faiss = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            response = get_response_llm(llm, vectorstore_faiss, user_question)
            st.write(response)


if __name__ == "__main__":
    main()
