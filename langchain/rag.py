# Minimal LangChain RAG Example

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Initialize components
model = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings()

# Sample documents
documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "RAG stands for Retrieval Augmented Generation, which combines retrieval with generation.",
    "Vector databases store embeddings and enable semantic search over documents.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors."
]

# Split and embed documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.create_documents(documents)

# Create local vector database
vectorstore = FAISS.from_documents(splits, embeddings)


# RAG function
def rag_query(question: str) -> str:
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=2)
    context = "\n".join([doc.page_content for doc in docs])

    # Generate answer
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on the context. Context: {context}"),
        ("user", "{question}")
    ])

    chain = prompt | model
    result = chain.invoke({"context": context, "question": question})
    return result.content

if __name__ == "__main__":
    # Example query
    answer = rag_query("What is RAG?")
    print(f"Answer: {answer}")
