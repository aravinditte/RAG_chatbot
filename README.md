# RAG_chatbot
This project implements a Retrieval-Augmented Generation (RAG) chatbot using Python and LangChain. The chatbot leverages external data sources to provide accurate and context-aware responses to user queries.




1️⃣ chatbot.py (Main Streamlit App)
    Handles user interaction via Streamlit.
    Calls the retrieval pipeline for answering user queries.

2️⃣ retriever.py (Retrieval Logic)
   Loads the retriever from ChromaDB.
   Embeds the query and retrieves relevant document chunks.

3️⃣ rag_pipeline.py (Setting Up RAG)
   Loads dataset and creates embeddings.
   Uses FAISS or ChromaDB for retrieval.
   Can be run once to preprocess data.

4️⃣ requirements.txt (Dependencies)
   Defines required Python libraries for deployment.

5️⃣ README.md (Project Documentation)
   Installation Instructions
   How to Run the Chatbot
   How to Deploy on Streamlit Cloud

Refer Documentation for more information
