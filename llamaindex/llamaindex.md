# LlamaIndex - Components

## 🔧 LLM Models

Configure different language models and providers that LlamaIndex supports.

**Available**: OpenAI, Anthropic, Gemini, Ollama, HuggingFace, Local models  
**Docs**: [LLMs](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/)

```python
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings

# Configure different LLMs
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.7)

# Or use Anthropic
Settings.llm = Anthropic(model="claude-3-sonnet-20240229")

# Use with specific parameters
llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=512
)
```

## 📄 Document Loaders

Load data from various file formats and sources.

**Available**: PDF, Word, CSV, JSON, Web scraping, APIs, Databases  
**Docs**: [Data Connectors](https://docs.llamaindex.ai/en/stable/module_guides/loading/)

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader, DocxReader

# Load from directory
documents = SimpleDirectoryReader("./data").load_data()

# Load specific file types
pdf_reader = PDFReader()
pdf_docs = pdf_reader.load_data("document.pdf")

# Load from web
from llama_index.readers.web import SimpleWebPageReader
web_docs = SimpleWebPageReader().load_data(["https://example.com"])
```

## ✂️ Text Splitters / Node Parsers

Configure how documents are chunked into searchable nodes.

**Available**: Sentence, Token, Semantic, Code, Markdown splitters  
**Docs**: [Node Parsers](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)

```python
from llama_index.core.node_parser import (
    SentenceSplitter, 
    TokenTextSplitter,
    SemanticSplitterNodeParser
)

# Sentence-based splitting
splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20
)

# Token-based splitting  
token_splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

# Apply to documents
nodes = splitter.get_nodes_from_documents(documents)
```

## 🏗️ Index Types

Choose from different indexing strategies for your data.

**Available**: Vector, Tree, Keyword, Knowledge Graph, Summary indices  
**Docs**: [Indices](https://docs.llamaindex.ai/en/stable/module_guides/indexing/)

```python
from llama_index.core import (
    VectorStoreIndex,
    TreeIndex, 
    KeywordTableIndex,
    KnowledgeGraphIndex
)

# Vector index (most common)
vector_index = VectorStoreIndex.from_documents(documents)

# Tree index for hierarchical summaries
tree_index = TreeIndex.from_documents(documents)

# Keyword index for exact matches
keyword_index = KeywordTableIndex.from_documents(documents)
```

## 💾 Vector Stores

Configure different vector storage backends.

**Available**: FAISS, Chroma, Pinecone, Weaviate, Qdrant, Redis  
**Docs**: [Vector Stores](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/)

```python
from llama_index.vector_stores import (
    ChromaVectorStore,
    PineconeVectorStore,
    FaissVectorStore
)
from llama_index.core import StorageContext, VectorStoreIndex

# Chroma vector store
import chromadb
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create index with custom vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

## 📊 Embedding Models

Configure different embedding models for text representation.

**Available**: OpenAI, HuggingFace, Cohere, Local embeddings  
**Docs**: [Embeddings](https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/)

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# OpenAI embeddings
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# HuggingFace embeddings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

# Use with specific dimensions
embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    dimensions=1536
)
```

## 🔍 Query Engines

Configure different querying strategies.

**Available**: Vector, Tree, Transform, Router, Multi-step query engines  
**Docs**: [Query Engines](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/)

```python
from llama_index.core.query_engine import (
    RouterQueryEngine,
    TransformQueryEngine,
    MultiStepQueryEngine
)

# Basic query engine
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact"
)

# Transform query engine with custom transforms
query_engine = TransformQueryEngine(
    query_engine=base_engine,
    query_transform=custom_transform
)
```

## 🔄 Retrievers

Configure different document retrieval strategies.

**Available**: Vector, BM25, Knowledge Graph, Fusion retrievers  
**Docs**: [Retrievers](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/)

```python
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    BM25Retriever,
    QueryFusionRetriever
)

# Vector retriever with custom parameters
retriever = VectorIndexRetriever(
    index=vector_index,
    similarity_top_k=10,
    vector_store_kwargs={"filter": {"category": "tech"}}
)

# BM25 retriever for keyword search
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
```

## 📝 Prompts

Configure prompts and prompt templates for different query types and contexts.

**Available**: Simple prompts, Chat prompts, Conditional prompts, Custom templates  
**Docs**: [Prompts](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/)

```python
from llama_index.core import PromptTemplate
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT

# Custom text QA prompt
qa_prompt = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in a {style} manner.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# Use with query engine
query_engine = index.as_query_engine(
    text_qa_template=qa_prompt
)

# Chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert assistant. Answer based on the context: {context_str}"),
    ("user", "{query_str}")
])
```

## 📝 Response Synthesizers

Configure how retrieved information becomes final responses.

**Available**: Refine, Compact, Tree Summarize, Generation modes  
**Docs**: [Response Synthesis](https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/)

```python
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    ResponseMode
)

# Different response modes
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,  # or REFINE, TREE_SUMMARIZE
    use_async=True,
    streaming=True
)

# Custom query engine with specific synthesizer
query_engine = index.as_query_engine(
    response_synthesizer=response_synthesizer
)
```