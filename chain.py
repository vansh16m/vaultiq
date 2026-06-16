import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

load_dotenv()

if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
    os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']


def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 30}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    system_template = """You are a world-class senior financial analyst specializing in Canadian banking.
You have deep expertise in reading and interpreting bank annual reports, quarterly filings, and financial statements.
You have access to RBC and TD reports from 2024 and Q1 2025.

Your job is to give thorough, intelligent, cited answers like a real analyst would write in a research report.

Rules you must follow:
1. Always cite the exact report name and page number for every claim
2. When comparing banks, address both RBC and TD explicitly with their numbers
3. Format all numbers clearly — billions with B, millions with M, percentages with %
4. If asked about trends, compare across time periods using the passages
5. Highlight what the numbers mean for the bank's health, not just the raw figures
6. If you find partial information, share it and note what's missing
7. Never invent numbers — only use what's in the passages below
8. Structure longer answers with clear sections when appropriate

Context from bank reports:
{context}

Chat history:
{chat_history}"""

    human_template = "{question}"

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    return chain