from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from bs4 import BeautifulSoup as Soup
import os

# Define paths
url = "https://mydukaan.io/"
chroma_db = 'vectorstore/db_chroma'

def process_url_content(content):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # Check if the content is a tuple (content, metadata)
    if isinstance(content, tuple) and len(content) >= 1:
        content = content[0]  # Extract the content from the tuple
    
    print(content)
    texts = text_splitter.split_documents([content])  # Wrap content in a list for processing
    
    # Initialize HuggingFaceEmbeddings using a specific model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    
    # Create a vector store using FAISS from the text content and embeddings
    db = Chroma.from_documents(texts,embeddings, persist_directory=chroma_db)
    
    # Save the vector store locally
    # db.save_local(os.path.join(chroma_db, f"url_content_db"))
    print(f"Vector store saved for content from URL")

def process_urls_sequentially():
    # Ensure the directory exists
    if not os.path.exists(chroma_db):
        os.makedirs(chroma_db)
        print(f"Created directory: {chroma_db}")

    # Load content from URL using RecursiveUrlLoader
    loader = RecursiveUrlLoader(
        url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text,
    )
    contents = loader.load()
    
    print(f"URL content loaded from {url}")
    
    for content in contents:
        process_url_content(content)
        print(f"Processing content from URL sequentially")

if __name__ == "__main__":
    process_urls_sequentially()
