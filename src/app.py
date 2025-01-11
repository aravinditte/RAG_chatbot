import streamlit as st
from rag_pipeline import load_rag_pipeline
from transformers import pipeline


@st.cache_resource
def initialize_pipeline():
    """
    Load the saved RAG pipeline and initialize the QA model.

    Returns:
        tuple: The FAISS vectorstore and HuggingFace QA pipeline.
    """
    # Load the saved RAG pipeline
    vectorstore = load_rag_pipeline("rag_pipeline")

    # Load HuggingFace question-answering pipeline
    qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    return vectorstore, qa_model


def main():
    """
    Run the Streamlit app for the chatbot.
    """
    st.title("RAG-based Chatbot")
    st.write("Ask any question based on the knowledge base!")

    # Initialize the pipeline (vectorstore + QA model)
    vectorstore, qa_model = initialize_pipeline()

    # User input section
    user_query = st.text_input("Enter your question:")
    if user_query:
        with st.spinner("Fetching response..."):
            # Retrieve relevant documents
            docs = vectorstore.similarity_search(user_query, k=3)

            if not docs:
                st.error("I couldn't find anything relevant in my knowledge base. Please try rephrasing.")
                return

            # Combine content from top documents
            context = " ".join([doc.page_content for doc in docs])

            # Generate the answer using the QA model
            result = qa_model(question=user_query, context=context)

            # Display the response
            st.subheader("Answer:")
            st.write(result["answer"])

            # Optionally display retrieved documents for transparency
            with st.expander("Retrieved Context (Top 3 Documents)"):
                for i, doc in enumerate(docs):
                    st.write(f"**Document {i + 1}:** {doc.page_content}")


if __name__ == "__main__":
    main()
