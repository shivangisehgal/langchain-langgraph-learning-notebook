# LangChain Course Notebook — Parts 8–12

> Self-contained learning notes for Eden Marco's *LangChain & LangGraph* Udemy course.  
> Covers: Function Calling theory → RAG gist → Documentation Helper → Prompt Engineering → LLM Apps in Production.  
> Code drawn from `course-code/project-rag-gist/` and `documentation-helper/`.

---

## 8. Function Calling

### Intro

Function calling (also called **tool calling**) is the capability of an LLM to emit a *structured* function invocation—name plus arguments—instead of (or in addition to) free-form text. The call appears in a dedicated slot in the model response, not as prose you have to scrape with regex.

This section is theory-first. Earlier course layers already built ReAct agents three ways: LangChain abstractions, raw Ollama SDK loops, and a hand-rolled ReAct prompt. Function calling is the production successor to that prompt-based approach.

### \[Theory\] Understanding Function Calling for LLMs

**Definition.** The model is fine-tuned to detect when an external function should run, then format a valid JSON object that names the function and fills its schema. Developers supply function *definitions* (name, parameters, descriptions). The application parses the JSON, executes the real function, and feeds the result back into the conversation.

**Classic example.** User: “What’s the weather in Paris?” Bound tool: `get_current_weather`. Model response (conceptually):

```json
{
  "name": "get_current_weather",
  "arguments": { "location": "Paris", "unit": "celsius" }
}
```

Your app runs the weather API, appends the observation, and lets the model finish the answer in natural language.

**Motivation.** The ReAct *prompt* pioneered tool use, but it is fragile: models invent formats, miss closing tags, and break parsers. Function calling moves the heavy lifting to the vendor. You get machine-readable JSON that is far more deterministic.

**Two capabilities unlocked**

| Capability | What you get |
|---|---|
| External tools | Connect LLMs to APIs, DBs, search, calculators |
| Structured output | Force fields into JSON / Pydantic objects for downstream code |

**Advantages**

1. **Structured, reliable integration** — Fine-tuned to adhere to schemas; fewer formatting errors than ReAct text.
2. **Token efficient** — Skips verbose chain-of-thought in the visible output; returns the call only.
3. **Industry standard** — OpenAI (2023), Anthropic, Google, and peers all ship SOTA models with tool calling. Almost nobody ships production agents on raw ReAct prompts anymore.

**Tradeoff: opaque reasoning.** When the model decides to call a tool, you typically see *only* the name and args—not the internal justification. That makes auditing “why this tool / why these args?” harder than with an explicit ReAct thought trace. In practice, the reliability win dominates.

💡 **Extended Notes**

- Function calling ≠ “the model runs code.” The model *proposes* a call; your runtime executes it. Never treat model output as trusted code.
- Structured output (JSON schema / `with_structured_output`) often reuses the same fine-tuning surface as tool calling—schema adherence is the shared skill.
- Compare approaches:
  - **ReAct prompt:** full visibility of thoughts; brittle parsing; high token cost.
  - **Native function calling:** reliable schema; black-box decisions; lower token cost.
  - **Hybrid (LangGraph later):** graph edges decide *when* tools run; model fills args inside controlled nodes.

### Test Yourself — Section 8

1. What does the model actually return when it wants to use a tool?  
2. Name two problems function calling solves relative to a raw ReAct prompt.  
3. What is the main *downside* of native function calling for debugging?  
4. True/False: Function calling means the LLM executes your Python function inside the GPU.  
5. Why is function calling considered more token-efficient than ReAct prompting?

**Answers**

1. A structured JSON (or vendor-native) tool call with function name + arguments, in a dedicated response field.  
2. Reliable parseability / schema adherence; less brittle formatting; often better determinism.  
3. Opaque reasoning—you don’t see the chain-of-thought that led to the call.  
4. False—the app (or agent runtime) executes the function after parsing the call.  
5. It skips verbose thought/action traces and returns primarily the call payload.

---

## 9. The GIST of RAG — Embeddings, Vector Databases & Retrieval

### Introduction to Retrieval Augmented Generation (RAG)

**Problem.** You have a large private document (Harry Potter, a 200-page financial contract, an internal wiki). You want Q&A over *specific* paragraphs. The LLM was never trained on that private data, so it cannot answer from weights alone.

**Naive solution: stuff the whole document into the prompt.** Fails for four reasons:

1. **Hard token limit** — Even 1–2M context windows are finite.  
2. **Needle-in-a-haystack** — Research shows quality degrades as context grows; relevant facts get lost mid-prompt.  
3. **Cost** — You pay per token.  
4. **Latency** — Longer prompts take longer to process.

**RAG solution (high level).** Pre-process the corpus into chunks. At query time, retrieve only the *most relevant* chunks, **augment** the prompt with them, and **generate** the answer grounded on that context.

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG AT A GLANCE                         │
│                                                             │
│   Documents ──► Chunk ──► Embed ──► Vector DB               │
│                                      │                      │
│   User Query ──► Embed ──► Similarity Search (top-k)        │
│                                      │                      │
│                         Relevant chunks                     │
│                                      │                      │
│              Prompt = Instruction + Context + Question      │
│                                      │                      │
│                                   LLM ──► Answer            │
└─────────────────────────────────────────────────────────────┘
```

**Drawbacks of RAG (honest list).** Chunking is hard (wrong boundaries destroy meaning). Retrieval can miss the right chunk. Multi-doc and code repos need different strategies. Dynamic user uploads need an online ingestion path.

💡 **Extended Notes**

- RAG = **R**etrieval + **A**ugmentation + **G**eneration. Memorize the three letters as three pipeline stages.  
- “RAG is dead because of long context” is overstated. Long context *complements* RAG: you can retrieve richer snippets, still save cost, still cite sources, still reduce positional bias. Eden’s later stance: long windows amplify RAG; they do not replace it for production knowledge bases.

### Introduction to RAG Implementation

Building blocks introduced here:

| Component | Role |
|---|---|
| **Document loaders** | Uniform interface over PDF, Notion, Drive, WhatsApp, plain text → `Document(page_content, metadata)` |
| **Text splitters** | Chunk long text while trying to keep semantic coherence |
| **Embeddings** | Text → dense vector; similar meaning ⇒ nearby vectors |
| **Vector store (e.g. Pinecone)** | Persist vectors; nearest-neighbor search at query time |
| **Retriever** | LangChain wrapper that turns “query string” into top-k `Document`s |

**Embeddings intuition.** An embedding model is a black box: text in, float array out. Good models place “I want an extra large coffee,” “I’ll have a tall coffee,” and Spanish “quiero pedir café extra grande” near each other—even across languages—because semantics align.

**Vector DB role.** Stores embeddings and returns closest neighbors to a query vector in milliseconds. Without it, you’d embed Wikipedia yourself and scan linearly—impractical at scale.

```
                    EMBEDDING / VECTOR SPACE (2D sketch)

     "tall coffee" ●
                    \
                     ● "extra large coffee"
                      \
                       ● "quiero pedir café..."

                              ●  query: "how tall is Burj Khalifa?"
                             /
              Wikipedia para ●  (closest → used as context)
```

**End-to-end mental model for a book**

1. Split book → thousands of chunks.  
2. Embed each chunk → vectors.  
3. Upsert into Pinecone (or Chroma).  
4. Embed user question.  
5. Top-k similar chunk vectors → text.  
6. Prompt: question + chunks → LLM answer.

### Medium Analyzer — Boilerplate Project Setup

Project: `course-code/project-rag-gist/` (branch `project/rag-gist`).

**Setup flow Eden walks through**

1. Clone course repo; checkout starting commit for the RAG gist project.  
2. Dependencies via **uv** (`pyproject.toml` + `uv.lock`): `uv lock` then `uv sync` → creates `.venv`.  
3. Point the IDE interpreter at `.venv` (`which python3` after sync).  
4. `.env` with `OPENAI_API_KEY`, LangSmith vars (`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=RAG GIST`, `LANGSMITH_TRACING=true`), `INDEX_NAME`, `PINECONE_API_KEY`.  
5. Create a Pinecone index (example name: `medium-blogs-embeddings-index`):
   - Dense vectors, cosine similarity (default).  
   - Dimensions: **1536** for `text-embedding-3-small` (more capacity than the UI default 512).  
   - Serverless capacity; pick cloud/region carefully in prod (latency + egress + GDPR).  
6. Corpus: `mediumblog1.txt` — pasted Medium article on vector DBs.

💡 **Extended Notes — Pinecone index knobs**

- **Dimension must match** the embedding model. Mismatch → silent failure or reject.  
- Longer vectors hold more semantic detail but cost more storage/query.  
- Same region as the app reduces egress cost and latency.  
- LangChain Pinecone integration expects the env var name **`PINECONE_API_KEY`** exactly.

### Medium Analyzer — Class Review: TextLoader, TextSplitter, OpenAIEmbeddings, Pinecone

**Document loaders.** Thin wrappers that normalize third-party data into `Document`. Peeking at LangChain source: `TextLoader` opens a path, sets `metadata["source"] = filepath`, returns `[Document(...)]`. `WhatsAppChatLoader` adds regex parsing for sender/date. Same `.load()` interface for PDF, Slack, YouTube, Notion, etc.—that uniformity is the point.

**CharacterTextSplitter.** Splits on a separator (default newlines) with `chunk_size` and `chunk_overlap`. Overlap preserves boundary context so meaning isn’t cut mid-thought. `length_function` defaults to `len` (characters); you can swap in a token counter.

**OpenAIEmbeddings.** Uniform LangChain interface over embedding providers (OpenAI, Cohere, Hugging Face…). Under the hood: HTTP `/embeddings`. Historical note: `text-embedding-ada-002` was dramatically cheaper than prior OpenAI embeddings—price matters when you embed an entire warehouse.

**PineconeVectorStore.** Persistence + similarity search + upserts. Free tier is enough for the course.

### Medium Analyzer — Ingestion Implementation

Course code (updated path uses `UnstructuredLoader`; older videos used `TextLoader`—same pipeline):

```python
# course-code/project-rag-gist/ingestion.py
import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = UnstructuredLoader(
        file_path="mediumblog1.txt",
        chunking_strategy="basic",
        max_characters=1_000_000,
    )
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
```

**How it works**

1. **Load** → list of `Document`s (`page_content` + `metadata.source`).  
2. **Split** → ~20 chunks at `chunk_size=1000`, `overlap=0` for the Medium article (exact count depends on separators).  
3. **Embed + upsert** via `from_documents`: iterate chunks, call embedding API, write vectors + text + metadata to Pinecone.  
4. Inspect Pinecone UI: each row stores **text**, **source**, and the **vector**. Embeddings are one-way—you *must* store original text to retrieve readable context later.

**Chunk size heuristic.** Small enough that several chunks fit in the context window; large enough that a human reading one chunk still understands it. Garbage-in-garbage-out still applies: irrelevant context hurts quality *and* cost even with million-token models.

**Encoding gotcha.** Some OSes throw `UnicodeDecodeError`. Fix: `encoding="utf-8"` or `autodetect_encoding=True` on text loaders.

**Why use LangChain’s `from_documents`?** One interface across vector stores/embeddings; built-in batching, async, and rate-limit handling you’d otherwise rewrite.

### RECAP

Indexing is done. Next: embed the user query → vector DB top-k → augment prompt → generate. Eden notes retrieval videos were refilmed for Cursor + **uv** while ingestion code stayed best-practice.

### Medium Analyzer — Naive Retrieval Implementation

```python
# course-code/project-rag-gist/main.py (core pieces)
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    docs = retriever.invoke(query)           # 1. similarity search
    context = format_docs(docs)              # 2. stringify
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)          # 3. generate
    return response.content
```

**Demo query:** `"what is Pinecone in machine learning?"`

| Approach | Result Eden shows |
|---|---|
| Raw LLM (GPT-3.5) | Hallucinates a “pinecone algorithm” for hyperparameter search |
| Raw LLM (newer GPT) | Knows Pinecone vector DB from training data |
| RAG + GPT-3.5 | Correct: managed vector DB for ML apps |

**Lesson.** RAG grounds weaker / older models *and* private data. Even strong models benefit when the corpus is proprietary or post-cutoff.

**Limitations of the manual function.** No streaming, no async, hard to compose, fragmented LangSmith traces (each component is a separate root run).

### Medium Analyzer — 2 Step RAG (LCEL)

```python
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def create_retrieval_chain_with_lcel():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain

# invoke
chain = create_retrieval_chain_with_lcel()
answer = chain.invoke({"question": query})
```

```
  INPUT: {"question": "..."}
           │
           ▼
  ┌────────────────────────────────────────┐
  │ RunnablePassthrough.assign(context=…)  │
  │   itemgetter("question")               │
  │        │                               │
  │        ▼                               │
  │   retriever (top-k docs)               │
  │        │                               │
  │        ▼                               │
  │   format_docs → string                 │
  │                                        │
  │ OUTPUT: {question, context}            │
  └────────────────────────────────────────┘
           │
           ▼
     prompt_template → llm → StrOutputParser
           │
           ▼
        answer string
```

**How `assign` works.** `RunnablePassthrough` is identity on the input dict. `.assign(context=subchain)` *adds* a `context` key while keeping `question`. The subchain `itemgetter("question") | retriever | format_docs` pulls the string, retrieves, formats. Plain Python functions like `format_docs` are auto-wrapped as `RunnableLambda` so they gain `.invoke` / stream / batch.

Eden flags this LCEL block as the trickiest syntactic sugar in the RAG section—worth rewatching until the dict shape `{question, context}` feels inevitable.

**Naive vs LCEL — side-by-side**

| Step | Manual function | LCEL chain |
|---|---|---|
| 1 Retrieve | `retriever.invoke(query)` | `itemgetter("question") \| retriever` |
| 2 Format | `format_docs(docs)` | `\| format_docs` |
| 3 Prompt | `prompt_template.format_messages(...)` | `\| prompt_template` |
| 4 LLM | `llm.invoke(messages)` | `\| llm` |
| 5 Parse | `response.content` | `\| StrOutputParser()` |
| Tracing | Scattered root runs | Single runnable sequence |

**Why LCEL wins for this pipeline**

- Declarative composition with `|`  
- Streaming / async / batch for free  
- **One LangSmith trace** for the whole runnable sequence—bottlenecks and intermediate I/O (assign → retriever → format → prompt → LLM) are visible under a single parent

Answers match the naive path; observability and maintainability do not. Prefer LCEL whenever the pipeline is a fixed DAG of runnables.

### LangChain RAG Documentation (critique)

Eden’s take on official LangChain RAG docs post-v1.0:

1. **Semantic search tutorial** — Solid on Documents / split / embed / store; light on full apps.  
2. **“Build a RAG agent”** — Wraps retrieval as a *tool* on a ReAct agent. Eden **dislikes this as default production advice**:
   - LLM may skip retrieval when needed or call it unnecessarily.  
   - Extra inference call → latency + cost.  
   - Jailbreak / off-topic risk for customer support.  
   - Good when greetings shouldn’t hit the KB, or when query rewriting from chat history helps—but you can rewrite deterministically without full agent freedom.  
3. **Two-step chain in docs** — Matches what the course built; single inference; less flexible. Docs’ “agent without tools + middleware injection” is too opaque for Eden’s taste.  
4. **Custom RAG with LangGraph** — Eden endorses this path (corrective / self / adaptive RAG later in the course).

| Pattern | Control | Flexibility | Latency | Prod bias (Eden) |
|---|---|---|---|---|
| Always-retrieve 2-step LCEL | High | Low | Lowest | Great for fixed Q&A |
| Retrieval-as-tool agent | Low | High | Higher (extra LLM) | Risky default |
| Hybrid LangGraph RAG | Medium–High | Medium | Medium | Preferred in enterprises |

### Test Yourself — Section 9

1. List the four problems with stuffing an entire book into the prompt.  
2. Why must vector DBs store original text alongside embeddings?  
3. What does `k=3` mean on a retriever?  
4. What does `RunnablePassthrough.assign(context=...)` add to the input dict?  
5. When would Eden prefer a fixed 2-step RAG over a retrieval-tool agent?

**Answers**

1. Token limits, needle-in-haystack quality drop, cost, latency.  
2. Embeddings are not invertible—you need the text to show the LLM and the user.  
3. Return the top 3 most similar chunks.  
4. Keeps original keys (e.g. `question`) and adds computed `context`.  
5. Domain Q&A / support where retrieval should *always* run and agent freedom is a liability.

---

## 10. Building a Documentation Assistant

*(Embeddings, VectorDBs, Retrieval, Memory, Streamlit)*

### What are we building? A lightweight Cursor-like docs feature (RAG)

Goal: a slim **chat.langchain.com**—ingest LangChain docs, answer questions with citations, Streamlit UI, conversational memory / coreference later.

Pipeline stages:

1. Crawl docs (Tavily).  
2. Chunk + embed + index (Pinecone / Chroma).  
3. Retrieval agent (or chain) for answers + sources.  
4. Streamlit chat UI.  
5. Glance at production Agentic RAG (chat.langchain.com).

### Quick Note: Pipenv vs uv

Transcript for this short lecture is empty in the source dump; the surrounding videos still make the tradeoff clear by *which tool Eden uses when*.

| | **Pipenv** (`Pipfile` / `Pipfile.lock`) | **uv** (`pyproject.toml` / `uv.lock`) |
|---|---|---|
| Speed | Adequate | Extremely fast resolve/install |
| Docs helper repo | Uses Pipenv historically (`pipenv install`) | Newer RAG gist videos: `uv lock` → `uv sync` |
| Lock semantics | Pipfile.lock pins transitive deps | uv.lock same idea, Astral resolver |
| Commands | `pipenv install`, `pipenv run streamlit …` | `uv sync`, `uv run streamlit …` |
| IDE | Select Pipenv-created venv | Select `.venv` created by uv |
| Ecosystem momentum (2025+) | Mature but quieter | Becoming default in new LangChain samples |

💡 **Extended Notes**

- **Same Python, different workflow.** Both give you an isolated venv + locked deps. Don’t mix them in one project unless you know why.
- **Course inconsistency is intentional history.** Older docs-helper footage is Pipenv/PyCharm; refilmed RAG retrieval is Cursor + uv. Code semantics didn’t change—only packaging UX.
- **Team rule of thumb.** Greenfield → uv. Matching an existing `Pipfile` branch for the course → stay on Pipenv so versions match Eden’s lockfile.

### Environment Setup

1. Clone `documentation-helper` at branch `1-start-here`.  
2. Pinecone index e.g. `langchain-doc-index` / `langchain-docs-2025` / `langchain-docs-2026` (videos differ by re-record): **1536** dims, cosine, `text-embedding-3-small`, serverless. Region/cloud matter for GDPR and latency.  
3. `.env`: `PINECONE_API_KEY`, `OPENAI_API_KEY`, later `TAVILY_API_KEY`.  
4. `pipenv install` (or uv if you migrate).  
5. Boilerplate: `logger.py`, empty `backend/`, create `ingestion.py`.

### Ingestion Pipeline Intro

History: early course versions manually scraped docs; brittle across machines. Migrated Firecrawl → **Tavily** for better LangChain integration and scaling.

Division of labor:

- **Tavily** — crawl / map / extract documentation HTML → markdown-ish raw content.  
- **LangChain** — `Document`s, chunking, metadata, vector indexing.  

Tavily free tier is enough for the course.

### Imports (ingestion)

From `documentation-helper/ingestion.py`:

```python
import asyncio, os, ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import Colors, log_error, log_header, log_info, log_success, log_warning

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,  # critical under rate limits
)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# vectorstore = PineconeVectorStore(index_name="langchain-docs-2025", embedding=embeddings)
tavily_crawl = TavilyCrawl()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_extract = TavilyExtract()
```

**How it works.** `certifi` avoids SSL failures under heavy concurrent HTTPS. `retry_min_seconds=10` backs off when OpenAI embedding TPM limits hit (429). Chroma persists under `chroma_db/`; Pinecone is a one-line swap—same LangChain vector store interface.

### Tavily Crawling

**Web crawling** = follow links depth-first/breadth-first to discover pages. For agents, crawling reaches content beyond one-shot search.

```python
res = tavily_crawl.invoke(
    {
        "url": "https://python.langchain.com/",
        "max_depth": 2,
        "extract_depth": "advanced",
    }
)

all_docs = []
for item in res["results"]:
    all_docs.append(
        Document(
            page_content=item["raw_content"],
            metadata={"source": item["url"]},
        )
    )
```

**Knobs Eden emphasizes**

| Param | Meaning | Practice |
|---|---|---|
| `max_depth` | Link hops from base URL (max 5) | Start at 1–2; raise after reviewing results. Depth 1 ≈ 18 pages; 2 ≈ 75; 5 ≈ 251 (numbers vary over time). |
| `extract_depth="advanced"` | Tables + embedded content | Higher success, higher latency |
| `instructions` | Natural-language *URL filter* during mapping | e.g. “only AI agents content” → ~23 pages. Write filters, not questions. |

**Why offload crawling.** Rate limits, bot protection, JS rendering—unless crawling *is* your product, buy a specialist.

`metadata["source"]` enables citations → user trust.

### \[Optional\] TavilyMap + TavilyExtract (high customizability)

Two-phase alternative to `TavilyCrawl`:

1. **`TavilyMap`** — traverse site graph → sitemap URL list (`max_depth`, `max_breadth`, `limit`).  
2. **`TavilyExtract`** — scrape URL batches → `{url, raw_content}`.

Use when you need to inspect/filter the URL list before paying for extraction, or to tune concurrency yourself. Crawl is preferred for most course/prod “just ingest this docs site” cases.

### \[Optional\] Crawling Deep Dive

Production-shaped pattern:

1. Map → up to hundreds of URLs.  
2. `chunk_urls(urls, chunk_size=20)` → batches (respect API URL-count limits).  
3. `asyncio.gather` over `extract.ainvoke({"urls": batch})` → **two levels of parallelism** (API batch + async I/O).  
4. Convert each result to `Document(page_content=raw_content, metadata={"source": url})`.  
5. Log failed batches; continue.

Manual sequential download used to take hours; concurrent extract finishes in minutes.

### Recap (ingestion)

Hard part of many production RAG systems is **getting clean data in**. You now have LangChain `Document`s. Remaining: chunk → embed → index.

### Chunking (Text Splitting)

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
splitted_docs = text_splitter.split_documents(all_docs)
```

**RecursiveCharacterTextSplitter** tries separators in order (paragraphs → newlines → spaces → chars) so chunks stay as semantic as possible under a size cap.

**“Is RAG dead?” counter-arguments (Eden)**

1. Cost & latency of million-token prompts.  
2. Precision / noise reduction / less positional bias.  
3. **Citations and provenance**—mandatory in regulated settings.  
4. Long context *amplifies* RAG by allowing richer retrieved packs.

Chunking is not a silver bullet: explore small-to-big, semantic chunking, code-aware splitters as you mature.

### Batch Indexing

```python
async def index_documents_async(documents: List[Document], batch_size: int = 50):
    batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            return True
        except Exception as e:
            log_error(f"Failed batch {batch_num} - {e}")
            return False

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # ... success counts / warnings ...

# main:
await index_documents_async(splitted_docs, batch_size=500)
```

**How it works.** Each `aadd_documents` embeds then upserts. Batch size is a **sweet spot** against embedding TPM and vector-store write limits—not a magic constant. Eden shows that removing `retry_min_seconds` triggers **429 rate limits** mid-ingest. Switching to Chroma creates local `chroma_db/` (SQLite-backed) with the same call pattern.

Example scale from the video: ~6506 chunks indexed (depends on crawl depth / date).

### Retrieval Agent Implementation

```python
# documentation-helper/backend/core.py
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(
    index_name="langchain-docs-2026", embedding=embeddings
)
model = init_chat_model("gpt-5.2", model_provider="openai")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs  # content → LLM; artifact → app only

def run_llm(query: str) -> Dict[str, Any]:
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    answer = response["messages"][-1].content

    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}
```

**How it works**

1. `init_chat_model` — one string switches OpenAI / Gemini / etc.  
2. `content_and_artifact` — serialized string goes to the model; raw `Document` list stays in the ToolMessage artifact for the UI (no context pollution).  
3. `create_agent` runs a LangGraph ReAct loop under the hood.  
4. System prompt forces tool use + citations + “say I don’t know” anti-hallucination.

### Run, Debug, Trace RAG Agent

Example query: `"what are deep agents?"`

Message timeline in debug:

1. HumanMessage — user query.  
2. AIMessage tool_call — `retrieve_context` (often **rephrased**, e.g. “LangChain deep agents definition”).  
3. ToolMessage — `content` = serialized sources; `artifact` = `[Document, …]` (not sent back to the LLM).  
4. Final AIMessage — grounded answer.

LangSmith shows the full LangGraph run. Prefer `as_retriever()` over raw `similarity_search` so retrieval appears cleanly as a retriever span in traces.

### Frontend with Streamlit (UI)

```python
# documentation-helper/main.py (structure)
import streamlit as st
from backend.core import run_llm

def _format_sources(context_docs):
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]

st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

with st.sidebar:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Ask me anything about LangChain docs…",
        "sources": [],
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer…"):
                result = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))
            st.markdown(answer)
            # render sources expander…
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)
```

**How it works.** Streamlit `session_state` is a dict surviving reruns—chat history lives there. Forgetting to append the assistant message makes history vanish on the next input (classic Streamlit footgun). Run with `pipenv run streamlit run main.py` or `uv run streamlit run main.py`.

**Disclaimer.** Streamlit is a prototype UI. Production chat needs **generative UI** (tool progress, streaming state, human-in-the-loop)—covered via CopilotKit later.

### Documentation Helper In Production

[chat.langchain.com](https://chat.langchain.com) is the production sibling: open-source LangChain + LangGraph + Next.js.

Observable behavior:

1. User asks “What is LangChain?”  
2. System generates **sub-queries**, retrieves for each, filters/reranks.  
3. Generates answer + sources.  
4. UI exposes intermediate context (generative UI → trust).  
5. Coreference works (“Who created it?” → Harrison Chase).

Backend: multi-agent retrieval graph; prompts pulled from LangChain Hub (`router`, `generate_queries`, …). This is the “next level” after the course’s Streamlit prototype.

### RAG Architecture

```
  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────────┐
  │  2-Step RAG      │   │  RAG Agent       │   │  Hybrid / Corrective RAG │
  │  (LCEL)          │   │  (tool call)     │   │  (LangGraph)             │
  ├──────────────────┤   ├──────────────────┤   ├──────────────────────────┤
  │ always retrieve  │   │ LLM decides when │   │ query rewrite            │
  │ then generate    │   │ & how to search  │   │ retrieve                 │
  │ 1 LLM call       │   │ 2+ LLM calls     │   │ grade docs               │
  │ high control     │   │ high flexibility │   │ web fallback / rewrite   │
  │ low latency      │   │ higher latency   │   │ answer validation        │
  └──────────────────┘   └──────────────────┘   └──────────────────────────┘
         ▲                        ▲                        ▲
         │                        │                        │
    Medium Analyzer          Docs Helper              Later section
    (course default          (create_agent)           (recommended
     for fixed Q&A)           demo)                    for enterprises)
```

Eden’s production stance: pure retrieval-tool agents are **too free** for most Q&A products; hybrid graphs that keep control while allowing query enhancement and validation are what enterprises actually ship. Use cases: internal docs, knowledge bases—not every problem needs an agent.

### Docs Helper Architecture (ASCII)

```
                    ┌─────────────────────────────────────┐
                    │         Streamlit (main.py)         │
                    │  session_state chat + source cites  │
                    └─────────────────┬───────────────────┘
                                      │ run_llm(query)
                                      ▼
                    ┌─────────────────────────────────────┐
                    │         backend/core.py             │
                    │  create_agent + retrieve_context    │
                    └──────────────┬──────────┬───────────┘
                                   │          │
                     tool call     │          │ artifact Documents
                                   ▼          ▼
                    ┌──────────────────┐   UI sources expander
                    │ Pinecone / Chroma│
                    │  (embeddings)    │
                    └────────▲─────────┘
                             │ aadd_documents
                    ┌────────┴─────────┐
                    │  ingestion.py    │
                    │ TavilyCrawl →    │
                    │ split → batch    │
                    └──────────────────┘
```

### Test Yourself — Section 10

1. Why store `metadata["source"]` at ingest time?  
2. What is the difference between TavilyCrawl and Map+Extract?  
3. What does `response_format="content_and_artifact"` buy you?  
4. Why is Streamlit insufficient as a production agent UI?  
5. Which RAG architecture does Eden recommend for most enterprise Q&A?

**Answers**

1. Citations / provenance / trust when answering.  
2. Crawl is one-shot map+scrape; Map+Extract splits discovery and scraping for control and custom batching.  
3. String for the LLM + raw Documents for the app without polluting model context.  
4. Limited transparency into agent state/tools; generative UI needs richer streaming/HITL.  
5. Hybrid LangGraph-style RAG with controlled steps—not a fully free retrieval agent.

---

## 11. Prompt Engineering Theory

### The GIST of LLMs

A **language model** estimates \(P(x_{t+1} \mid x_1,\ldots,x_t)\) over vocabulary \(V\)—next-token prediction, aka “super autocomplete.”

An **LLM** is that model trained on enormous corpora, so next-token guesses become fluent answers. Every prompt is a conditioning prefix; the model samples (or argmaxes) token by token. That statistical nature is *why* hallucinations exist: high-probability text ≠ true text.

### What is a Prompt? Composition of a Formal Prompt

Shared vocabulary matters for collaboration. A prompt guides the model with four components:

| Component | Role | Example |
|---|---|---|
| **Instruction** | Task heart | “Summarize…”, “Classify…” |
| **Context** | Background that improves accuracy | Role, company, domain constraints |
| **Input data** | Payload to process | The email / ticket / paragraph |
| **Output indicator** | Signal that generation should start / format | `JSON:`, `Answer:` |

Sometimes the indicator is implicit in the instruction; sometimes you make it explicit.

### Zero Shot Prompting

Ask for a task **with no examples**—rely on pretrained knowledge.

```
Create a list of the 10 must-visit cities in the world in no particular order.
```

**Pros.** Fast, intuitive, most common starting point.  
**Cons.** Less control, weaker accuracy on niche formats, hard to fine-tune behavior without examples or stronger instructions.

### Few Shot Prompting

Provide **n examples** (one-shot = n=1) of desired I/O, then the real input.

Eden’s Blue Willow demo: generate compressed image-description prompts for a Yorkshire dog in Brazilian winter.

- **Zero-shot** — creative, long description; model guesses style.  
- **One-shot** — one adjective/noun example → more compressed, adjective-heavy.  
- **Few-shot** (blue/red/green dog examples) — model learns to lead with a *color* and add a motion descriptor (fur fluttering)—pattern induction from shots.

**Tradeoff.** More shots → less artistic freedom, more precision to your style. Shots cost tokens; pick representative, diverse examples.

### Chain of Thought Prompting

From Google research: for multi-step math / common-sense reasoning, force **intermediate steps**.

Classic failure: toy count works; dog-hours-per-week fails under zero-shot (model does \(10\times5=50\) instead of \(10\times0.5\times7=35\)).

**Few-shot CoT:** show a worked solution with steps, then ask the analogous problem—the model copies the decomposition pattern.

**Zero-shot CoT:** append “Let’s think step by step.”

| Flavor | What you supply |
|---|---|
| Zero-shot CoT | Magic phrase only |
| Few-shot CoT | Worked examples with reasoning traces |

### ReAct Prompting

Paper: **Re**ason + **Act**. Combine CoT-style thoughts with **actions** that hit external tools (search, APIs), then **observations**, looping until a final answer.

Apple Remote multi-hop question: zero-shot, CoT-only, and act-only all fail; ReAct search → observe → rethink → search again → keyboard function keys.

This paradigm seeded frameworks like LangChain: parse thoughts/actions, execute tools in code, re-prompt with observations. Native function calling later made the “act” channel reliable without brittle text formats.

### Prompt Engineering Quick Tips

1. **Context** — Don’t make the model invent the scenario. “Senior DevOps at a fast-paced cloud startup” yields deeper interview questions than a bare ask.  
2. **Clear, non-ambiguous tasks** — “Improve UX” is vague; “Identify pain points to raise CSAT and conversion” is actionable. Treat prompting like precise human communication (Eden’s supermarket-apples analogy).  
3. **Specificity** — More precise detail → more targeted outputs.  
4. **Iterate** — Lean-startup loop: run → evaluate → refine prompt → repeat. Time spent crafting prompts saves time overall.

### Context Engineering

**Evolution of prompt engineering.** Prompts are static; real agent context is dynamic—developer instructions, user input, chat history, tool results, retrieved docs. “Garbage in, garbage out” still rules.

Failure modes as context grows in long agent runs:

| Failure | Meaning |
|---|---|
| **Context poisoning** | A hallucinated tool result enters history and spreads |
| **Context confusion** | Irrelevant tokens steer the answer |
| **Context clash** | Contradictory pieces fight each other |

Also: cost, latency, and hard window limits. Context engineering = selecting, compressing, and structuring the *right* dynamic context (tools included)—for both app developers *and* end users of coding agents.

### Context Engineering a System Prompt

Evidence: leaked SOTA agent system prompts (Claude Code, Cursor, Devin) are **hundreds of lines**, continuously iterated—serious engineering, not an afterthought.

**Goldilocks zone** (Anthropic framing):

```
  Too specific ◄──────────────────●──────────────────► Too vague
  (brittle if/else               (Goldilocks:         ("do the right
   state machine)                 principles +         thing")
                                  boundaries)
```

- **Too specific:** Treats the LLM like a state machine (`ask exactly 3 follow-ups`); exhaustive escalation lists; maintenance nightmare; maybe you wanted a workflow, not an agent.  
- **Too vague:** “Act consistent with brand essence”; undefined “escalate if needed”; inconsistent runs.  
- **Just right:** Clear identity/scope; empower with goals + heuristics; reasoning *framework* (identify → gather → resolve → confirm); compressed principles (prefer simplest solution); no contradictory overlapping rules.

Good system prompts teach **principles** that generalize; bad ones hard-code scripts or say nothing.

### Test Yourself — Section 11

1. Name the four formal components of a prompt.  
2. What is the difference between zero-shot and few-shot prompting?  
3. Give one zero-shot CoT trick.  
4. What does ReAct add on top of chain-of-thought?  
5. Name three context-failure modes in long agent runs.

**Answers**

1. Instruction, context, input data, output indicator.  
2. Zero-shot: no examples; few-shot: n≥1 worked examples.  
3. Add “Let’s think step by step.”  
4. External actions + observations in a reason→act→observe loop.  
5. Poisoning, confusion, clash (plus overflow/cost/latency).

---

## 12. LLM Applications In Production

### LLM Applications in Production

Challenges when agents leave the notebook:

1. **Sequential multi-call latency** — Each tool decision needs an LLM call; deep tasks become long-running. Mitigations (mentioned, not deep-dived): semantic / LLM caches.  
2. **Context window** — Huge prompts per step; even 100k+ models suffer “lost in the middle.” Limits how many steps you can afford.  
3. **Hallucinations & compounded error** — If each tool choice is ~90% correct, six independent steps → \(0.9^6 \approx 0.53\) joint success. Fine-tuning for tool selection can raise per-step accuracy. RAG reduces factual hallucination by grounding.  
4. **Pricing** — Fat prompts × millions of users; GPT-4-class reasoning is slow/expensive. Mitigations: cache; **RAG over tools** when the toolbelt is huge (retrieve candidate tools before the reasoner runs).  
5. **Response validation** — Wrong *format* breaks apps even if content is fine; automated eval remains hard.  
6. **Security** — Prompt injection + powerful tools (SQL, APIs) = breach risk. Least privilege; prompt guardrails; tools like **LLM Guard**.  
7. **Don’t over-agent** — If the workflow is deterministic, write Python. Agents are for non-deterministic branching. Prototypes are easy; production is careful.

### LLM Application Development Landscape

Four complexity tiers:

| Tier | Pattern | Example Eden cites |
|---|---|---|
| 1 | Single LLM call (± light postprocess) | Children’s story generator |
| 2 | RAG + vector store | Quiver / “second brain” over personal data |
| 3 | Agents + tools | Torq “Socrates” remediating security alerts |
| 4 | Agents + vector memory | AutoGPT / GPT Engineer-style long-term memory |

Course goal: give you the building blocks for tiers 2–3 (and later LangGraph for advanced RAG).

### Privacy & Data Retention

**Disclaimer (course + this notebook):** Not legal advice. Read each vendor EULA; involve legal/privacy teams.

For **enterprise APIs** (not consumer ChatGPT UI):

- Top vendors often **do not train** on API data by default; training is opt-in.  
- Retention varies (e.g. OpenAI abuse monitoring ~30 days; some customers get **zero retention**). Policies differ and change.  
- Highly regulated orgs (banks, insurers, health) may still reject vendor guarantees → **self-host open-source**, or host OSS models in *their* cloud account (Bedrock / Vertex-style) to keep controls while shifting ops burden.

Self-hosting trades privacy for GPU ops, availability, security patching, and cost.

### Generative UI/UX featuring CopilotKit

Backend quality ≠ product trust. Users know GenAI is flaky; UX must show:

- Where answers come from (RAG sources).  
- Which tools ran and why.  
- Intermediate state / streaming progress.  
- Human-in-the-loop pause/resume.

**CopilotKit** — OSS React components/hooks for generative UI, with **CoAgents** bridging LangGraph state, parallel nodes, and HITL. Eden has no affiliation; recommends it as current best building blocks. Streamlit docs-helper is intentionally *not* this bar.

### Official LangChain Academy Courses

Free courses on LangChain Academy (esp. **LangSmith**): tracing, monitoring, evaluation, human feedback—exactly the LLMOps loop for POC → production. Eden points here for depth beyond the course’s LangSmith intro.

### Open Source LLMs vs Managed Providers (Deepseek)

| | Open source (Deepseek, Llama, …) | Managed (OpenAI, Anthropic, Gemini) |
|---|---|---|
| **Apparent cost** | “Free” weights | Pay per token |
| **Real cost** | GPUs, SRE, availability, security | Usually predictable API bill |
| **Privacy** | Full control if self-hosted | Data leaves to vendor (unless private cloud offering) |
| **Customization** | Full fine-tune freedom | Vendor fine-tune APIs exist; Eden rarely recommends FT first |
| **Ops burden** | High | Low (plug-and-play) |
| **Quality trend** | Closing fast; sometimes wins benches | Still strong; getting cheaper/faster |

Nuance: hosting OSS on Groq-like hosts **reintroduces** third-party dependency and blunts the privacy argument. Cloud-native enterprises already trust AWS/GCP—using Claude on Bedrock or Gemini in-project may be consistent with existing risk posture. Prefer **prompting + few-shot** over fine-tuning until metrics demand otherwise.

### Confidence in AI Results (Assaf Elovic & Harrison Chase)

Arabic-narrated lecture summarizing their article: product adoption hinges less on peak model accuracy and more on **care** (trust calculus):

\[
\text{Care} \approx \frac{\text{Value}}{\text{Risk} \times \text{Correction Effort}}
\]

- **Value** — time/money/creative upside when AI works.  
- **Risk** — blast radius of a wrong answer.  
- **Correction** — how hard undo is.

**Cursor:** high value, low risk (local editor, not auto-pushing to main), low correction (delete suggestion) → high care.  
**Creative assistants (e.g. Jasper):** AI as assistant, human final say → high care.  
**Monday.com AI board mutations:** medium risk × medium correction → medium care; **preview mode** drops risk → care rises—*without changing the model*.

Design for trust in high-stakes domains (finance, health) by lowering risk and undo cost.

### \[NEW\] AI FOMO is the New Normal

Andrej Karpathy’s late-2025 note resonated industry-wide: “I’ve never felt this much behind as a programmer.” The profession is being refactored; bits written by humans are sparser; failing to claim a 10× boost from the last year’s tooling *feels* like a skill issue—even when no human can track every announcement.

**New stack to hold in your head (partial):** agents, sub-agents, prompts, context, memory, modes, permissions, tools, plugins, skills, hooks, MCP, LSP, slash commands, workflows, IDE integrations—plus a mental model for strengths/pitfalls of stochastic, fallible, changing entities mixed into “good old-fashioned engineering.”

**The paradigm shift**

```
Punch cards → Assembly → C → Python → … → English/prompts + agents
                                              ↑
                                    You are the orchestrator
```

Software work looks more like **tech-lead work**: assign tasks to agents, review outputs, supply tools/skills, give feedback, loop. Review weight rises; raw syntax typing weight falls. Privilege: you live through the abstraction jump in real time.

**Eden’s personal aside (regex).** Classical engineering still matters—agents fail at fiddly deterministic tasks (e.g. painful regex) the same way junior engineers do; you still need taste and verification.

**Survival advice**

1. Accept FOMO as the steady state—Karpathy has it; Eden has it; you will too.  
2. **Focus** beats infinite Twitter. Skim; only deep-dive what unblocks *your* problem.  
3. Roll up sleeves: experimentation beats passive consumption.  
4. Distill noise: most “revolutionary” launches are not on your critical path.

💡 **Extended Notes**

- FOMO is a **product of abundance**, not of your inadequacy. Treat the feed as a radar, not a syllabus.  
- Build a personal “allowlist” of layers you *will* master this quarter (e.g. RAG eval + LangSmith) and consciously ignore the rest until needed.

### Finished course? What’s next!

Recap of the pre-LangGraph arc: **agents** + **RAG**—the two dominant LLM app patterns (plus plain LLM calls).

Next skills: **LLMOps**—prompt management across model changes, latency/cost monitoring, debugging agents, automated evaluation. Tools: **LangSmith** (unified, proprietary); **Pezzo** (OSS alternative mentioned). Security: prompt injection, over-privileged tools; LangChain moved unsafe integrations to experimental. Keep learning via LangChain blog + X/Twitter; course updates continuously.

LangGraph section starts next in the outline (~lecture 85).

### Test Yourself — Section 12

1. Why do sequential agent steps compound failure probability?  
2. Name the four landscape tiers of LLM apps.  
3. What product-design lever raised “care” for Monday.com-style AI without model changes?  
4. Give one reason regulated banks self-host OSS LLMs.  
5. What is Eden’s rule of thumb before reaching for agents?

**Answers**

1. Per-step success multiplies (e.g. \(0.9^n\)); errors accumulate across dependent calls.  
2. Single call → RAG → agents → agents+vector memory.  
3. Preview / confirm-before-apply (lower risk).  
4. Control over data, retention, and residency beyond vendor contractual guarantees.  
5. If you can do it deterministically in code, don’t use an agent.

---

## Appendix — Quick Reference Cheatsheet

```
INGEST:    Load → Split → Embed → Upsert
RETRIEVE:  Embed(query) → top-k → format
GENERATE:  Prompt(instruction + context + question) → LLM

2-STEP:    always retrieve then generate   (LCEL assign pattern)
AGENTIC:   LLM may call retrieve tool      (create_agent)
HYBRID:    graph: rewrite → retrieve → grade → generate → validate
```

**Key env vars:** `OPENAI_API_KEY`, `PINECONE_API_KEY`, `INDEX_NAME` / index string, `TAVILY_API_KEY`, LangSmith (`LANGSMITH_*`).

**Key packages:** `langchain`, `langchain-openai`, `langchain-pinecone`, `langchain-chroma`, `langchain-tavily`, `langchain-text-splitters` / classic splitters, `streamlit`, `python-dotenv`.

---

*End of notebook parts 8–12. Continue with LangGraph introduction in the next notebook part.*
