"""
Simple RAG (Retrieval-Augmented Generation) Example
"""

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

def rag_example():
    # Sample documents
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of AI that enables computers to learn from data.",
        "React is a JavaScript library for building user interfaces."
    ]

    # Get embeddings for all documents
    print("Creating embeddings for documents...")
    doc_embeddings = []
    for doc in documents:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=doc
        )
        doc_embeddings.append(response.data[0].embedding)

    # User query
    query = "What is Python?"

    # Get embedding for query
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = query_response.data[0].embedding

    # Calculate similarities and find best match
    similarities = []
    for doc_emb in doc_embeddings:
        similarity = np.dot(query_embedding, doc_emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
        )
        similarities.append(similarity)

    # Get the most relevant document
    best_doc_idx = np.argmax(similarities)
    relevant_context = documents[best_doc_idx]

    print(f"Query: {query}")
    print(f"Most relevant document: {relevant_context}")

    # Generate answer using the relevant context
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Answer the question based on the provided context."},
            {"role": "user", "content": f"Context: {relevant_context}\n\nQuestion: {query}"}
        ]
    )

    print(f"Generated answer: {response.choices[0].message.content}")

# running this code:
if __name__ == "__main__":
    rag_example()
    print("RAG example executed successfully.")
