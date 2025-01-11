import os
import faiss
import pickle
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.document_loaders import DataFrameLoader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pandas as pd


def setup_rag_pipeline(dataframe, save_path="rag_pipeline"):
    """
    Sets up the RAG pipeline using HuggingFace Transformers and FAISS.
    Saves the pipeline to disk for future use.

    Args:
        dataframe (pd.DataFrame): Knowledge base data.
        save_path (str): Path to save the pipeline.

    Returns:
        function: A function to retrieve answers from the RAG pipeline.
    """
    os.makedirs(save_path, exist_ok=True)

    # Load the documents using DataFrameLoader
    loader = DataFrameLoader(dataframe, page_content_column="text")
    documents = loader.load()

    # Use SentenceTransformers for embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode([doc.page_content for doc in documents], show_progress_bar=True)

    # Create an in-memory docstore
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})

    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Map indices to document IDs
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    # Save FAISS index and associated data
    faiss.write_index(index, os.path.join(save_path, "faiss_index"))
    with open(os.path.join(save_path, "docstore.pkl"), "wb") as f:
        pickle.dump(docstore, f)
    with open(os.path.join(save_path, "index_to_docstore_id.pkl"), "wb") as f:
        pickle.dump(index_to_docstore_id, f)

    print(f"Pipeline saved to {save_path}")


def load_rag_pipeline(load_path="rag_pipeline"):
    """
    Loads a saved RAG pipeline from disk with the embedding function.

    Args:
        load_path (str): Path to load the pipeline from.

    Returns:
        FAISS: The loaded FAISS vector store.
    """
    # Load FAISS index and associated data
    index = faiss.read_index(os.path.join(load_path, "faiss_index"))
    with open(os.path.join(load_path, "docstore.pkl"), "rb") as f:
        docstore = pickle.load(f)
    with open(os.path.join(load_path, "index_to_docstore_id.pkl"), "rb") as f:
        index_to_docstore_id = pickle.load(f)

    # Reconstruct FAISS vector store
    embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Reuse the same embedding model
    vectorstore = FAISS(
        embedding_function=embedder.encode,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore

if __name__ == "__main__":
    data = pd.read_csv("data/further_reduced_train.csv")

    # Uncomment the line below to set up and save the pipeline
    # setup_rag_pipeline(data)

    # Uncomment the lines below to load the saved pipeline
    vectorstore = load_rag_pipeline()
    print("Loaded pipeline successfully!")
