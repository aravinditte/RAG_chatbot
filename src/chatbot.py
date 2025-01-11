from rag_pipeline import load_rag_pipeline
from transformers import pipeline


def main():
    """
    Runs the chatbot for user interaction using the free LLM pipeline.
    """
    print("Loading the RAG pipeline...")
    vectorstore = load_rag_pipeline("rag_pipeline")
    print("Pipeline loaded successfully!")

    print("Loading the QA model...")
    qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    print("Chatbot is ready! Type 'exit' to quit.")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve relevant documents
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            print("Bot: I couldn't find anything relevant in my knowledge base. Please try rephrasing.")
            continue

        # Combine content for context
        context = " ".join([doc.page_content for doc in docs])

        # Generate the answer using the QA model
        result = qa_model(question=query, context=context)
        print(f"Bot: {result['answer']}")


if __name__ == "__main__":
    main()
