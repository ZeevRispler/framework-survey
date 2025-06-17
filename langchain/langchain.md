# LangChain - Components

## 🔗 Prompt Templates

Create dynamic prompts with variables and multi-role conversations.

**Class**: `ChatPromptTemplate`, `PromptTemplate`  
**Base Class**: `BasePromptTemplate`  
**Docs**: [Prompt Templates](https://python.langchain.com/docs/concepts/prompt_templates/)

```python
from langchain_core.prompts import ChatPromptTemplate

# Custom prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Your expertise is in {domain}."),
    ("user", "{input}")
])

# Usage
formatted = prompt.invoke({
    "role": "data scientist", 
    "domain": "machine learning",
    "input": "Explain neural networks"
})
```

## 🧠 LLMs / Chat Models

Configure language models with custom parameters and behavior.

**Classes**: `ChatOpenAI`, `init_chat_model`, various provider classes  
**Base Class**: `BaseChatModel`, `BaseLLM`  
**Docs**: [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

```python
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

# Universal model initialization
model = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    temperature=0.7,
    max_tokens=1000
)

# Direct provider configuration
openai_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
    timeout=30,
    max_retries=2
)

# Using with custom system messages
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Hello!")
]
response = model.invoke(messages)
```

## 🛠️ Tools

Define functions that LLMs can call to perform specific tasks.

**Class**: `BaseTool`, `@tool` decorator  
**Base Class**: `BaseTool`  
**Docs**: [Custom Tools](https://python.langchain.com/docs/how_to/custom_tools/)

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="Math expression to evaluate")

@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return str(eval(expression))
```

## 📄 Document Loaders

Load data from custom sources or formats into LangChain documents.

**Class**: `BaseLoader`  
**Base Class**: `BaseLoader`  
**Docs**: [Document Loaders](https://python.langchain.com/docs/how_to/document_loader_custom/)

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class CustomLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        # Your custom loading logic
        with open(self.file_path) as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": self.file_path})]
```

## ✂️ Text Splitters Usage

Configure how documents are chunked for processing and retrieval.

**Available Classes**: `RecursiveCharacterTextSplitter`, `CharacterTextSplitter`, `TokenTextSplitter`  
**Docs**: [Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure splitter with custom parameters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
    add_start_index=True
)

# Split documents
documents = text_splitter.split_documents(docs)

# Split text directly
chunks = text_splitter.split_text("Your long text here...")
```

## 🔍 Retrievers Usage

Configure how relevant documents are found and returned for queries.

**Available Classes**: `VectorStoreRetriever`, `MultiQueryRetriever`, `ParentDocumentRetriever`  
**Docs**: [Retrievers](https://python.langchain.com/docs/concepts/retrievers/)

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# Create vector store retriever
embeddings = OpenAIEmbeddings()
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(documents)

# Configure retriever with custom parameters
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Use different search strategies
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "lambda_mult": 0.5}
)

# Retrieve documents
results = retriever.invoke("your query here")
```

## 🗃️ Vector Stores Usage

Store and search document embeddings with different backends and configurations.

**Available Classes**: `FAISS`, `Chroma`, `InMemoryVectorStore`, `Pinecone`  
**Docs**: [Vector Stores](https://python.langchain.com/docs/concepts/vectorstores/)

```python
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# In-memory vector store (simple)
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(documents)

# FAISS vector store (persistent)
faiss_store = FAISS.from_documents(documents, embeddings)
faiss_store.save_local("./faiss_index")

# Load existing FAISS index
loaded_store = FAISS.load_local("./faiss_index", embeddings)

# Search with different parameters
results = vectorstore.similarity_search(
    query="your query",
    k=5,
    filter={"source": "document.pdf"}
)

# Search with scores
results_with_scores = vectorstore.similarity_search_with_score(
    query="your query",
    k=3
)
```