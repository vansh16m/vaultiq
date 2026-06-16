from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "What did RBC say about mortgage risk?"
print(f"Searching for: '{query}'\n")

results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    source = doc.metadata.get('source', 'unknown')
    page = doc.metadata.get('page', '?')
    print(f"--- Result {i+1} ---")
    print(f"Source: {source} | Page: {page}")
    print(doc.page_content[:300])
    print()