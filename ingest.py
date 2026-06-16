import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def build_vectorstore():
    print("Loading PDFs...")
    docs = []
    for filename in os.listdir("docs"):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(f"docs/{filename}")
            pages = loader.load()
            docs.extend(pages)
            print(f"  Loaded: {filename} ({len(pages)} pages)")

    print(f"\nTotal pages loaded: {len(docs)}")

    print("\nChopping into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    print("\nCreating embeddings — this takes 2-3 minutes...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    print("\nDone! Your vector store is ready.")

if __name__ == "__main__":
    build_vectorstore()