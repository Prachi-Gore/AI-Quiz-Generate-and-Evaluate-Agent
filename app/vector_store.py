from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from app.prompts import summary_chain

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# print("OPENAI_KEY:", OPENAI_KEY)

PERSIST_DIR = "vector_db/chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

# Initialize embedding function
embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_KEY)

# Test embedding creation
sample_vector = embedding.embed_query("This is a test text.")
# print("Embedding created successfully. Length:", len(sample_vector))

# Initialize Chroma vector store
vectorstore = Chroma(
    collection_name="book_summaries",
    embedding_function=embedding,
    persist_directory=PERSIST_DIR
)

async def add_book_to_vector_db(book_id: str, title: str, author: str):
    """Generate and store a book summary with embeddings in Chroma"""
    summary = await summary_chain.ainvoke({"title": title, "author": author})
    summary_text = summary.get("text") 
    print('book summary_text',summary_text)
    vectorstore.add_texts(
        texts=[summary_text],
        metadatas=[{"book_id": book_id, "title": title, "author": author}],
        ids=[book_id]   # assign a unique id
    )
    vectorstore.persist()
    print(f"Stored book '{title}' (ID: {book_id}) in Chroma DB")

def get_relevant_summary(book_id: str):
    """Retrieve relevant summary for RAG"""
    docs = vectorstore.similarity_search(query="irrelevant", # mandatory to pass otherwise no use when we have filter
                                          k=1,filter={"book_id": book_id})
    return docs[0].page_content if docs else ""
