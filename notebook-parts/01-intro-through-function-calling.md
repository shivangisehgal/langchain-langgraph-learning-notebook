# LangChain & LangGraph — Self-Contained Learning Notebook
## Sections 1–7: Introduction through Function Calling / Agents Under The Hood (Layers 1–3)

> Based on Eden Marco's *LangChain — Agentic AI Engineering with LangChain & LangGraph* (Udemy).  
> Code excerpts are taken from the course `course-code/` projects. Treat this notebook as a full replacement for watching lectures 1–41 (through Theory: Function Calling). RAG begins after this document.

---

## 1. Introduction

### Course Introduction

Eden Marco (backend engineer turned AI engineer; LangChain ambassador) filmed this as the **third edition** of the course, updated for **LangChain 1.0** (filmed against ~1.0.2 / targeting compatibility through ~1.11.x). The remake incorporates student feedback: more production/security content, reorganized lectures, and code aligned with modern LangChain best practices (composability, robustness).

**Who he is (context that matters for tone):** years of backend work in cybersecurity companies, **zero ML experience before 2023**, jumped into generative AI when LLMs became accessible. His pitch for LangChain: it **commoditized** LLM app development so engineers without PhDs can build agents and RAG systems.

### Course Objectives

By the end of the full course you should be able to:

1. Build **LLM-powered applications** with LangChain — especially the two dominant shapes: **agents** and **RAG**.
2. Understand those systems **under the hood** (no magic): dive into LangChain source, ReAct loops, function calling, etc.
3. Use the ecosystem stack: **LangSmith** (tracing), **LangGraph** (workflow / durable agents).
4. Apply **prompt engineering** history and techniques (zero-shot, few-shot, CoT, ReAct).
5. Think about **production** concerns: testing, logging, monitoring, alerting, security — woven through the course, not bolted on at the end.

**Two application types (mental model):**

```
┌─────────────────────────────────────────────────────────┐
│                  LLM Applications                       │
├──────────────────────┬──────────────────────────────────┤
│       AGENTS         │              RAG                 │
│  LLM = reasoner that │  Retrieve private/external docs  │
│  chooses tools/steps │  then generate grounded answers  │
│  dynamically         │                                  │
└──────────────────────┴──────────────────────────────────┘
```

**Audience:** software engineers / data scientists comfortable with code. **No ML prerequisite.** Surprising audiences (lawyers, doctors) have succeeded, but the course is technical.

**Prerequisites (assumed, not taught):**

- Solid Python (functions, classes, run a program)
- Basic Git (`clone`, `commit`)
- Virtual environments + environment variables

This is **not** a Python 101 course — that flexibility lets the course go deep on LangChain quickly. Udemy 30-day refund applies; Eden also offers personal refunds after that if needed (course meta — skip if you're reading this notebook standalone).

### Course Structure + How to Get the Best Out of It

Practical learning advice from the lecture (stripped of Udemy UI fluff):

- Watch sections **in order** (1 → 2 → 3 …). Each builds on prior abstractions.
- Check each video's **Resources** for Discord links, code snippets, docs, and commit links matching the recording.
- Use the **Troubleshooting** section when stuck; use the **Theory** section when terms like *chain-of-thought* or *ReAct paper* appear.
- Contact paths that matter for learning: Discord (preferred for course Q&A), LinkedIn, Udemy DM — community Q&A often already answers your bug.

### Course's Community

Discord is the support surface: config issues, error messages, conceptual clarifications. Eden is most available weekends; weekdays ~twice daily. Rule of thumb: **ask early** — one person's question usually unblocks many silent students. If Discord invite fails, log into Discord *before* clicking the invite.

### Course Resources

[No transcript for this lecture.] Expect: GitHub repo with per-video commit links, Discord invite, docs bookmarks, shared LangSmith traces. Prefer the **exact commit** linked in a lecture over "latest main" when following along.

### Test Yourself

**Q1.** What are the two primary application categories Eden frames for LLM apps?  
**Q2.** What is *not* required to take this course (ML vs software)?  
**Q3.** Why remake the course for LangChain 1.0 instead of patching old videos?

<details>
<summary>Answers</summary>

1. **Agents** and **RAG** applications.  
2. Prior **machine learning** knowledge is not required; **software/Python comfort** is.  
3. LangChain APIs and best practices moved fast; remaking keeps code current (1.0+), integrates feedback, and reduces "deprecated API" friction.

</details>

---

## 2. The GIST of LangChain — Hello World Chain

### What is LangChain? (Under ~6 Minutes)

**Definition:** LangChain is an **open-source framework** that simplifies building LLM-powered applications via tools and abstractions. Industry adoption is high among developers who treat models as **black boxes** (use, don't train).

**Why it exists — the "stitching" problem.** Alone, an LLM app often needs:

- Private data the model was never trained on (PDFs, email, Notion)
- Dynamic prompts from user input
- Conversation / message history
- Easy **vendor switching** (Claude → Mistral → local Llama)
- Tool use (Google Search, arbitrary APIs)

Stitching that yourself is a lot of moving parts. LangChain owns the heavy lifting via modules:

| Module | Role |
|--------|------|
| **Chat models** | One interface across vendors; swap models "like socks" |
| **Prompts / prompt templates** | Manage, optimize, serialize prompts; inject variables |
| **Document loaders** | Load Notion/PDF/email/… into a unified `Document` interface |
| **Agents / tools / LangGraph** | Reasoning + tool invocation ("LLM superpowers") |
| **Tracing (LangSmith)** | Debug/monitor production LLM apps |

```
User input ──► PromptTemplate ──► ChatModel ──► (optional parse/tool/chain)
                    │                  │
                    └──── LCEL pipe ───┘
```

You will implement agents and RAG end-to-end in this course, including diving into implementations.

💡 **Extended Notes — Vendor lock-in**

Chat model abstraction is not just DX sugar. Production teams routinely A/B models, fall back on outages, or move to local models for privacy. A single interface (`invoke` / messages / tool binding) makes that a **config change** instead of a rewrite — until you peel under the hood (Sections 4–7) and see how much schema/format work LangChain was hiding per vendor.

### What Are We Building? LangChain Hello World Chain

**Goal:** Learn by doing. Build a tiny chain that:

1. Takes text about **Elon Musk**
2. Formats it through a **prompt template**
3. Sends it to an LLM
4. Returns a **short summary + two interesting facts**

You'll meet: prompt templates, chat models, chains (LCEL), debugging/tracing, **OpenAI GPT-5** (or any first-tier model), and **Ollama** local open-weights (**Gemma 3**).

### Project Setup

Bootstrap pattern used throughout the course:

1. Clone the course GitHub repo (linked in resources / intro).
2. Create an orphan working branch, e.g. `project/hello-world` (`git checkout --orphan …`), optionally wipe files for a clean slate. (When you clone later, the branch may already exist — use a new name if needed.)
3. Use **uv** (Rust-based, fast pip alternative) — or Poetry / Pipenv; course only needs install + venv + run.
4. `uv init` → creates `main.py` + `pyproject.toml`.
5. Install packages:

```bash
uv add langchain
uv add langchain-openai          # OpenAI integration package
uv add python-dotenv             # load .env
uv add black isort               # formatting (optional hygiene)
# later: langchain-ollama for local models
```

**Why separate integration packages?** Historically providers were bundled in core LangChain. They were split so you don't download 100 providers to use one, and so each vendor maintains its own package. Same idea for `langchain-tavily`, `langchain-ollama`, etc.

6. Add a standard Python `.gitignore` (never commit `.venv` or secrets).
7. Create `.env`:

```bash
OPENAI_API_KEY=sk-...
# Optional later:
# GOOGLE_API_KEY=...          # Gemini via Google AI Studio + langchain-google-genai
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=...
# LANGSMITH_PROJECT=hello-world
```

**Security:** API keys are passwords. Malicious scrapers hunt leaked keys on GitHub. Set a **budget limit** on the OpenAI dashboard. Without billing/credits you often hit **429**-style failures. Variable name **`OPENAI_API_KEY` must be exact** — LangChain looks for that name.

Verify load:

```python
from dotenv import load_dotenv
import os

load_dotenv()
print(os.environ.get("OPENAI_API_KEY"))  # should print your key locally — remove before commit
```

Commit boilerplate as something like `environment setup`. Course commits are linked per video.

### LangChain Fundamentals: Prompt Templates, ChatModels, and Chains

**Prompt** = text (and increasingly multimodal) input the model processes. Formal prompt composition is covered later in Prompt Engineering Theory.

**PromptTemplate** = wrapper that parameterizes a prompt so you can reuse it:

- Product = "cat food" vs "piano" → different outputs from one template.
- Core job: **format variables into the string** eventually sent to the LLM (plus ecosystem features: validation, reuse, tracing).

**Chat models** (`ChatOpenAI`, `ChatOllama`, …) are the primary LLM interface. Historical LLMs: string in → string out. Modern chat models: **list of messages** (system / human / AI / tool) in → **AI message** out. Beyond text: tool calling, token metadata, etc. (Glossary later.)

**Habit:** Cmd/Ctrl-click into LangChain source. This course repeatedly opens implementations on purpose.

**Chain** = workflow connecting components so **output of step N → input of step N+1**. Steps can be LLM calls, prompts, transforms, tool calls, or nested chains. Composition is why LangChain took off: you can format → LLM → parse → API → another LLM without spaghetti.

### Building a LangChain Chain to Summarize Text

Real course code (`project-hello-world/main.py`):

```python
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Elon Reeve Musk FRS ...  # Wikipedia-style bio (abbreviated in this notebook)
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    llm = ChatOpenAI(temperature=0, model="gpt-5")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
```

#### How it works (line by line)

1. **`load_dotenv()`** — loads `OPENAI_API_KEY` (and later LangSmith vars) into the process env.
2. **`information`** — raw context the model has not "memorized" for this call; you inject it via the template (same pattern as private docs later in RAG).
3. **`summary_template`** — instruction + `{information}` placeholder. Curly braces are **template variables**, not Python f-string evaluation at definition time.
4. **`PromptTemplate(input_variables=[...], template=...)`** — declares expected keys. Misspell `information` → clean error instead of a broken prompt silently sent to the API.
5. **Why not f-strings?** Templates enforce required vars, are reusable across chains, are first-class for **logging/tracing**, and play nicer with output parsers / injection-aware formatting later.
6. **`ChatOpenAI(temperature=0, model="gpt-5")`** — chat model wrapper over OpenAI SDK. Credentials from `OPENAI_API_KEY`.
7. **`temperature`** — randomness dial:
   - **0–0.3**: more deterministic (summaries, code, tests)
   - **≥0.8**: more creative (poetry, brainstorming)
8. **`chain = summary_prompt_template | llm`** — **LCEL** (LangChain Expression Language). `|` builds a **Runnable** pipeline: left output → right input.
9. **`chain.invoke({"information": information})`** — runs the runnable:
   - Invokes prompt template → `PromptValue` (fancy string)
   - Pipes into `llm.invoke(...)`
   - Returns an **`AIMessage`**; **`.content`** is the text.
10. Print the content.

```
┌──────────────────┐     PromptValue      ┌─────────────┐     AIMessage
│ PromptTemplate   │ ───────────────────► │  ChatModel  │ ──────────────► content
│ {information}    │                      │  (OpenAI)   │
└──────────────────┘                      └─────────────┘
         ▲
         │ invoke({"information": bio})
```

💡 **Extended Notes — LCEL is the hard part (on purpose)**

Eden explicitly says LCEL/`|` is often **harder to internalize than agents**. Mental model for now: **left-to-right dataflow**, not magic. Everything is a `Runnable` with `.invoke` / `.stream` / `.batch`. You do *not* need the full LCEL graph theory to finish Hello World — you need "template formats → model generates."

### Debugging and Tracing Our LangChain Chain

In the debugger:

- `response` is an **`AIMessage`**.
- Text lives in **`response.content`**.
- Also inspect: `type`, `response_metadata` (model name, finish reason), token usage — essential later for agents/cost monitoring.

Message types and richer fields are covered in the Glossary section of the full course.

### Using Local Open-Weights Models with LangChain and Ollama

**Strength of LangChain:** swap models by changing the chat model client; the rest of the chain stays put.

**Setup:**

1. Install [Ollama](https://ollama.com) for your OS.
2. `ollama pull gemma3:270m` (or a larger model).
3. `ollama list` / `ollama run <model>` to verify CLI chat works.
4. In code:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(temperature=0, model="gemma3:270m")
# Comment out ChatOpenAI; keep the same chain = template | llm
```

**Tradeoff observed in lecture:** tiny local models are **fast and free**, but may **under-follow instructions** (e.g. summary without a clean "two facts" section). For agentic work later, Eden recommends something with stronger reasoning + **function calling** (e.g. GPT-OSS locally if you have disk/RAM).

Cloud open-weights alternatives (e.g. Groq) follow the same pattern: install provider package → swap chat model → same chain.

### Integrating LangSmith for LangChain Application Tracing

LangSmith = tracing/debugging platform for LLM apps. Free tier is enough for the course.

**Setup steps:**

1. Sign up at LangSmith → **Set up tracing**.
2. Generate API key.
3. Env vars:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=hello-world
# If outside US: set LANGSMITH_ENDPOINT to the EU endpoint (or you'll get auth errors)
```

4. No code changes required for basic tracing — use normal LangChain objects; runs appear in the project.

**What you see in a Hello World trace:**

- A **RunnableSequence** (prompt → chat model)
- Inputs as Human/formatted messages; outputs as AIMessage
- Latency, TTFT, success/fail, token totals
- Nested steps: PromptTemplate then ChatOpenAI / ChatOllama

Commit the working chain (e.g. `hello World Chain`). Shareable public traces can be linked in resources.

### Semantic Versioning in LangChain

Course filmed against **LangChain ~1.0.2**. Check your `uv.lock` for the exact resolved version.

- **Patch/minor** within 1.0.x: usually compatible.
- **Breaking major changes:** Eden updates the course; ping Discord if videos diverge from current APIs.

💡 **Extended Notes — SemVer in a moving ecosystem**

Pin via lockfile for reproducibility. When upgrading majors, re-run smoke tests on chains/agents — especially tool calling and `create_agent` APIs, which evolved hard between classic LangChain agents and 1.0.

### Test Yourself

**Q1.** What does `summary_prompt_template | llm` create, and what method runs it?  
**Q2.** Name two benefits of `PromptTemplate` over an f-string.  
**Q3.** Which env var must exist for `ChatOpenAI` credentials by default?  
**Q4.** What is the main quality tradeoff when switching Hello World to a tiny Ollama model?

<details>
<summary>Answers</summary>

1. An LCEL **Runnable** chain; run with **`.invoke(...)`**.  
2. Variable validation/errors, reuse, first-class tracing/logging, safer structured formatting.  
3. **`OPENAI_API_KEY`**.  
4. Speed/cost improve; instruction-following / answer quality often drop.

</details>

---

## 3. THE GIST of AI Agents

### What Are AI Agents? A High-Level Overview

Ask ten people what an "AI agent" is — get ten answers. Shared core (aligned with LangChain):

> **An agent is a software system that uses an LLM as a reasoning engine to decide what actions to take, then executes those actions.**

**Agents vs chains:**

| | Chain / Runnable | Agent |
|--|------------------|-------|
| Control flow | **Developer-hardcoded** sequence | **LLM decides** next step |
| LLM role | Often one step (summarize, extract) | **Reasoner + actor** over tools |
| Flexibility | Predictable pipeline | Dynamic tool use for open-ended tasks |

**Tools** = capabilities you pre-write (API call, DB query, code execution, search). The LLM doesn't "have internet" by itself — you grant it.

**ReAct agents** (Reason + Act): combine chain-of-thought-style reasoning with tool actions in an **iterative loop** until the task is done. LangChain/LangGraph ship prebuilt ReAct-style agents with state, tools, and customization.

This section teaches the **`create_agent` interface** (how to equip an LLM with tools). The next major block peels the onion under the hood.

### What Are We Building? AI Job Search Agent

**Demo parallel:** ChatGPT with Web Search — query for 3 Bay Area AI engineer / LangChain job postings on LinkedIn, return details **with source URLs**.

**Why sources matter:** LLMs hallucinate. URLs let users verify grounding and trust. (Also a preview of **generative UI** — the app reflects agent steps.)

**Core idea:** Models are text-in/text-out (or multimodal) but **static in time** without tools. Search tools add real-time external capability. You'll build that with LangChain + Tavily.

Historical note: when Eden first taught this (~2022), chat UIs lacked built-in search; now it's table stakes — same agent pattern underneath.

### The Evolution of LangChain ReAct Agents

Timeline of agent APIs (remember the shape, not every date):

```
2022  ReAct prompting only (text Thought/Action/Observation)
  │
  ▼
Native function/tool calling on models
  │
  ▼
Tool-calling agents (structured calls > fragile prompts)
  │
  ▼
LangGraph-backed ReAct (durable execution, persistence, control)
  │
  ▼
LangChain 1.0  create_agent(...)   ← high-level API over LangGraph ReAct
```

**Course pedagogy:** start with **latest `create_agent`** (interface only), then walk **backward** through layers until you rebuild from ReAct prompts. Goal: intuition for production agents, not cargo-culting one API.

### Setting Up the Environment for a LangChain Search Agent

Branch: `project/search-agent` (or related ReAct search branch — follow resources).

```bash
uv init
uv add langchain langchain-openai langchain-tavily tavily-python python-dotenv black isort
```

**Tavily:** popular web search API for agents (featured in LangChain docs). Free tier ~1000 requests/month. Also offers Crawl / Map / Extract (used later in RAG/docs assistant). Playground demo: query → URLs + content snippets.

`.env` additions:

```bash
OPENAI_API_KEY=...
TAVILY_API_KEY=...          # exact name — LangChain looks this up
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=search-agent
```

`load_dotenv()` at top of `main.py` as before.

### Creating Your First LangChain Agent: Tools and LLMs

**Minimum agent ingredients:** (1) **tools**, (2) **LLM** as reasoner.

**What is a tool?** Any Python function the agent can execute — API, DB, code, search. You control the implementation → unbounded capabilities.

**Creating a tool:**

1. Write a normal function with **type hints** + **docstring**.
2. Decorate with `@tool` from LangChain.
3. Under the hood, name/args/docstring become metadata the LLM uses (via **function calling**) to decide *whether* and *how* to call it.

First lecture uses a **stub** search tool (returns static "Tokyo weather is sunny") to prove the loop before wiring Tavily.

Conceptual shape:

```python
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

@tool
def search(query: str) -> str:
    """Search the internet. Args: query. Returns search results."""
    print(query)
    return "Tokyo weather is sunny right now."

llm = ChatOpenAI()  # default model at filming could be gpt-3.5; later set gpt-5
tools = [search]
agent = create_agent(model=llm, tools=tools)

result = agent.invoke(
    {"messages": HumanMessage(content="What's the weather in Tokyo?")}
)
# LangChain accepts a single message or a list under "messages"
```

Rule of thumb for docstrings: **explicit and unambiguous** — the model is choosing tools from language.

### From Query to Answer: How a LangChain Agent Thinks

When you run the stub agent and inspect `result["messages"]`:

1. **HumanMessage** — user query  
2. **AIMessage** with **tool_calls** — model chose `search(query="weather in Tokyo")` (not the final answer yet)  
3. **ToolMessage** — observation from executing the tool ("Tokyo weather is sunny…")  
4. **AIMessage** — final natural language answer (no further tool calls)

LangSmith shows the run as **LangGraph** under the hood (don't panic — covered later). First LLM call includes **tool schemas** in the request. The model returns a structured tool call; LangChain's runtime **executes** the Python function; then another LLM call sees history + observation and answers.

```
User ──► LLM(+tools) ──► tool_call? ──yes──► execute tool ──► ToolMessage ──┐
              ▲                              │                              │
              └──────────── append history ──┴──────────────────────────────┘
                         no tool_call → Final Answer
```

Two conceptual pieces:

1. **Reasoning engine** (LLM) — which tool + args  
2. **Agent runtime** (LangChain/LangGraph) — execute tools, manage messages, loop

### Integrating Real-World Search with Tavily and LangChain Tools

**Custom tool wrapping Tavily SDK:**

```python
from tavily import TavilyClient

tavily = TavilyClient()  # reads TAVILY_API_KEY

@tool
def search(query: str) -> str:
    """Search the internet..."""
    return tavily.search(query=query)
```

Same agent invoke; LangSmith now shows real weather/job result payloads in tool observations.

Job query example:

> search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?

GPT-class models may emit **multiple parallel tool calls** (several LinkedIn/search queries). Final LLM turn consumes all ToolMessages.

**Best practice — prefer vendor LangChain tools:**

```python
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)
```

Why: Tavily's team writes better descriptions/args (`include_domains`, `search_depth=advanced`, etc.). The agent makes more precise calls than a thin DIY wrapper. Course commits show evolution: custom `@tool` → `TavilySearch` → structured output.

### Structured Output with LangChain Agents Using Pydantic

Most agent answers are free text. Production apps need **downstreamable** structure (JSON / Pydantic) for APIs and UIs.

Pass `response_format=` into `create_agent` with a Pydantic schema:

```python
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
            )
        }
    )
    print(result)  # includes structured_response: AgentResponse
```

#### How it works

- **`Field(description=...)`** — metadata for the model (what belongs in each field). Typos in descriptions often still work; clarity helps.
- **`default_factory=list`** — empty sources if omitted.
- **`response_format=AgentResponse`** — agent result gains **`structured_response`** typed as `AgentResponse` (`answer` + `sources[].url`).
- LangSmith shows the structured payload on the final turn.
- Hint for later: implemented via **function calling / structured output strategies** under the hood.

### [THEORY] Predictable Agent Responses with LangChain Structured Output

Two implementation strategies:

1. **Provider strategy** (default when supported) — use the vendor's native structured-output API. Schema can be Pydantic, dataclass, TypedDict, or JSON Schema. Responsibility for validity sits with the **model provider**.
2. **Tool strategy** — for models without native structured output but with tool calling: expose **one tool** whose schema *is* the desired object and force the model to call it. Classic evolution path of structured output.

LangChain picks provider strategy when available unless you override. Both accept the same schema family.

### Test Yourself

**Q1.** What is the key control-flow difference between a chain and an agent?  
**Q2.** In a tool-using run, what does the first AIMessage usually contain before the final answer?  
**Q3.** Why prefer `TavilySearch` over a hand-rolled `@tool` wrapping the SDK?  
**Q4.** Where does a Pydantic `response_format` show up in the agent result dict?  
**Q5.** Name the two structured-output strategies LangChain uses.

<details>
<summary>Answers</summary>

1. Chains hardcode the sequence; agents let the **LLM choose** next actions/tools.  
2. **`tool_calls`** (which tool + arguments), not the final user-facing answer.  
3. Better schemas/descriptions/args maintained by the vendor → better tool selection.  
4. **`structured_response`**.  
5. **Provider strategy** and **tool strategy**.

</details>

---

## 4. Agents Under The Hood (1/4) — Core Architecture

### Introduction to The Core Architecture of AI Agents

**Pedagogy: peel the onion.**

| Layer | What you build | Abstractions away |
|-------|----------------|------------------|
| **0** (done) | `create_agent(model, tools)` | Everything |
| **1** | Manual agent **while-loop** with LangChain primitives (`@tool`, `bind_tools`, messages) | Vendor JSON schemas, raw SDKs |
| **2** | Same loop with **raw Ollama SDK** + hand-written JSON tool schemas | LangChain chat/tool APIs |
| **3** | Same loop with **ReAct prompt + regex** (no function calling) | Native tool-calling APIs |

```
┌─────────────────────────────────────────────┐
│  Layer 1: LangChain                         │  ← @tool, bind_tools, ToolMessage
│  ┌────────────────────────────────────────┐  │
│  │  Layer 2: Raw Function Calling         │  │  ← JSON schemas, ollama.chat()
│  │  ┌─────────────────────────────────┐   │  │
│  │  │  Layer 3: Raw ReAct Prompt      │   │  │  ← prompt + regex + scratchpad
│  │  └─────────────────────────────────┘   │  │
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

Do this **hands-on**. Models: Ollama **Qwen3** (function-calling capable), optionally OpenAI. Any FC-capable model works.

### What Are We Building? An E-Commerce Agent

Theme: hardware store bot — **laptop / headphones / keyboard** with **bronze / silver / gold** discount tiers.

**Two tools:**

1. `get_product_price(product)` → catalog price  
2. `apply_discount(price, discount_tier)` → discounted price  

Example query: *"What is the price of a laptop after applying a gold discount?"*

With `create_agent` you'd pass both tools and be done. Here you implement a **lean agent loop** yourself so the diagram becomes code.

**Discount percentages are deliberately weird** (5% / 12% / 23%) so the model cannot "guess" math — it must call tools.

### [Theory] The Gist of ReAct

**Agent loop ≈ ReAct loop ≈ ReAct algorithm** — the engine behind Cloud Code, Gemini CLI, Codex, Devin-style agents. Originated in the paper *ReAct: Synergizing Reasoning and Acting in Language Models* (Princeton + Google researchers; Eden covers the paper in Prompt Engineering Theory).

**Loop steps:**

1. **User query** enters  
2. **Thought** — LLM sees system prompt + **tool descriptions** + history; decides tool call **or** final answer  
3. **Action** — application executes the chosen tool (LLM only *names* the call; your code runs it)  
4. **Observation** — tool result  
5. Append history (**scratchpad** / message list) and repeat  
6. Exit when LLM returns **no tool call** / Final Answer  

E-commerce walkthrough:

```
Iter 1: Thought → Action get_product_price("laptop") → Observation 1299.99
Iter 2: Thought → Action apply_discount(1299.99, "gold") → Observation 1000.99
Iter 3: Thought → Final Answer (no tool)
```

It's "just" a while loop — simple and extremely powerful.

### Setup

Branch: `project/agents-under-the-hood`.

```bash
uv init
# delete unused main.py if desired
uv add langchain langchain-ollama langchain-openai python-dotenv black isort
```

`.env`:

```bash
OPENAI_API_KEY=...          # optional; for model switch demos
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=ReAct Under The Hood
LANGSMITH_TRACING=true
# Ollama needs no API key
```

Pull and serve model:

```bash
ollama pull qwen3:1.7b      # ~1.4GB; 0.6b was too weak in Eden's tests
ollama run qwen3:1.7b       # smoke test, then /bye
ollama serve                # ensure server is up for Python clients
```

### Test Yourself

**Q1.** Order the four layers from most abstract to most raw.  
**Q2.** Who executes the tool — the LLM or your application?  
**Q3.** Why use non-obvious discount percentages in the demo catalog?

<details>
<summary>Answers</summary>

1. Layer 0 `create_agent` → Layer 1 LangChain loop → Layer 2 raw FC → Layer 3 ReAct prompt.  
2. **Your application** executes; the LLM only selects name/args (or Final Answer).  
3. So the model **cannot hallucinate** the math — it must call `apply_discount`.

</details>

---

## 5. [Layer 1] The ReAct Loop

### Writing Tools

File: `1_agent_loop_langchain_tool_calling.py`.

Key imports:

- `init_chat_model` — create a chat model from a **string** like `"ollama:qwen3:1.7b"` or `"openai:gpt-5"` (provider packages must be installed).
- `@tool` — function → LangChain tool (auto JSON schema from hints/docstring).
- `HumanMessage`, `SystemMessage`, `ToolMessage` — portable message types across providers.
- `MAX_ITERATIONS = 10` — safety cap (heuristic).
- `@traceable(name="LangChain Agent Loop")` — nest the whole run under one LangSmith span (manual agent loops don't auto-nest like `create_agent`).

Tools from course code:

```python
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)
```

Docstrings + type hints **are** the tool API the model sees after `@tool` normalizes them per vendor.

### Tool Binding and Defensive Prompting

```python
tools = [get_product_price, apply_discount]
tools_dict = {t.name: t for t in tools}

llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
llm_with_tools = llm.bind_tools(tools)
```

**`bind_tools`:** every subsequent invoke attaches tool schemas. Requires a **function-calling-capable** model. Same code path works for OpenAI/Anthropic after changing the `init_chat_model` string.

**`tools_dict`:** map LLM-chosen name → runnable tool object for `invoke`.

**Defensive system prompt** (critical for smaller open-weights models):

1. NEVER guess prices — must call `get_product_price` first.  
2. Only call `apply_discount` **after** a real price observation; pass that exact number.  
3. NEVER compute discounts with mental math — always use the tool.  
4. If tier unspecified — ask; don't assume.

Eden added these after watching Qwen hallucinate prices. Exercise: remove the rules and observe failures.

Messages seed:

```python
messages = [
    SystemMessage(content="You are a helpful shopping assistant. ... STRICT RULES ..."),
    HumanMessage(content=question),
]
```

### Understanding the ReAct Agent Loop in LangChain

Full loop from course code:

```python
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=("You are a helpful shopping assistant. ...")),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None
```

#### How it works

1. **Thought:** `llm_with_tools.invoke(messages)` → `AIMessage`.  
2. **Exit:** empty `tool_calls` → return `.content`.  
3. **Simplify:** only first tool call (models can return many; parallel FC exists — kept serial for clarity).  
4. **Action:** `tools_dict[name].invoke(args)` → observation.  
5. **Memory:** append AIMessage (decision) + `ToolMessage(content, tool_call_id)` so the next Thought sees history.  
6. **`tool_call_id`** links result to the call for tracing/correctness.  
7. **`@tool` auto-traces** tool spans in LangSmith.

Map to state machine:

| Diagram node | Code |
|--------------|------|
| Thought | `llm_with_tools.invoke` |
| Action | `tool_to_use.invoke` |
| Observation | `ToolMessage` |
| Final Answer | `if not tool_calls` |

This is a lean version of what `create_agent` / LangGraph ReAct does for you.

Related simpler teaching variant in `project-ReAct-Algo/main.py` (text-length tool + callback printer) — same bind_tools loop pattern with `while True`.

### Model Switch

Change one string:

```python
llm = init_chat_model("openai:gpt-5", temperature=0)
# vs ollama:qwen3:1.7b
```

LangSmith shows `ChatOpenAI` instead of Ollama. **Caveat:** easy switching ≠ production readiness. Eden saw **GPT-5.2** ask clarifying questions / underperform on this agent vs GPT-5 / Qwen for the same prompt — **evaluate before you swap**. Model marketing ≠ best for your loop.

### Test Yourself

**Q1.** What does `bind_tools` attach to each LLM request?  
**Q2.** Why append both the AIMessage and ToolMessage each iteration?  
**Q3.** What happens if `tool_calls` is empty?  
**Q4.** Why is "never calculate discounts yourself" in the system prompt?

<details>
<summary>Answers</summary>

1. Tool **schemas/descriptions** so the model can emit structured tool calls.  
2. So the next Thought has the **decision + observation** history (agent memory / scratchpad).  
3. Treat `.content` as the **final answer** and return.  
4. Defensive prompting against **hallucinated math**, especially on smaller models.

</details>

---

## 6. [Layer 2] Raw Function Calling

### Manual JSON Schemas vs LangChain Tool Abstraction

File: `2_agent_loop_raw_function_calling.py` — copy Layer 1, then strip LangChain chat/tools/messages; use `import ollama`.

Without `@tool`, you must either:

- Hand-write vendor JSON schemas, or  
- Pass Python functions (Ollama-specific) with **Google-style docstrings** (poorly documented; Eden digs into Ollama source)

Ollama schema shape (manual):

```python
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]
```

**Tradeoff table:**

| Concern | LangChain `@tool` | Raw schemas |
|---------|-------------------|-------------|
| Schema authoring | Auto from hints/docstrings | Manual, verbose, error-prone |
| Multi-vendor | One interface | Anthropic ≠ Ollama ≠ OpenAI formats |
| Docs clarity | Relatively consistent | Often incomplete (Ollama formal schema) |
| Switching models | Change string | Rewrite SDK + message roles + schemas |

That's a major reason LangChain won early: **integration cost**.

Tools stay plain functions; use `@traceable(run_type="tool")` because you lost automatic LangChain tool tracing.

### Building a ReAct Agent Loop with the Raw Ollama SDK

Differences vs Layer 1 (from course comments + transcript):

```python
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    messages = [
        {"role": "system", "content": "...STRICT RULES..."},
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name          # attribute access, not dict
        tool_args = tool_call.function.arguments

        observation = tools_dict[tool_name](**tool_args)  # direct call, not .invoke

        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(observation)})
        # Note: Ollama path may lack tool_call_id that LangChain ToolMessage needs

    return None
```

#### How it works / what LangChain was doing for you

1. **Message roles:** `user`/`assistant`/`system`/`tool` dicts vs typed `HumanMessage`/`AIMessage` (LangChain normalizes vendor naming: human vs user, AI vs assistant).  
2. **Tracing:** manual `@traceable` on chat + tools; LangChain chat models trace out of the box.  
3. **Tool dispatch:** `fn(**args)` vs `tool.invoke(args)`.  
4. **Tool call object shape:** `tool_call.function.name` vs `tool_call["name"]`.  
5. Same ReAct loop structure — only the "wires" change.

Rename the outer span to **"Ollama Agent Loop"** so LangSmith doesn't still say LangChain Agent Loop.

### Recap (Layer 2)

Same shopping question, same three iterations, same answer path — without LangChain. You paid for it with JSON schemas, SDK-specific messages, and manual tracing. Switching to Anthropic would mean another schema dialect and client.

### Test Yourself

**Q1.** What does LangChain's `@tool` generate that Layer 2 writes by hand?  
**Q2.** Why is multi-vendor support expensive without LangChain?  
**Q3.** How do you invoke a tool in Layer 2 vs Layer 1?

<details>
<summary>Answers</summary>

1. Per-vendor **JSON tool schemas** from signatures/docstrings.  
2. Each vendor has different **schema shapes, message roles, and SDKs**.  
3. Layer 2: `tools_dict[name](**args)`; Layer 1: `tool.invoke(args)`.

</details>

---

## 7. [Layer 3] The ReAct Prompt — Foundation of Function Calling

### What Are We Building? Function Calling (via ReAct Prompt)

**ReAct prompt** = the prompt that launched LangChain agents (Harrison Chase's `hwchase17/react` on LangChain Hub — millions of downloads). Before native function calling (OpenAI ~2023), agents were **pure prompt engineering**: model emits text in a strict format; your code parses and executes.

You will power the e-commerce agent with this prompt — **no `tools=` API**.

Canonical format:

```
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:
```

**Agent scratchpad** = accumulating history of Thought/Action/Observation injected each iteration. Techniques embedded: few-shot-ish format demonstration + chain-of-thought ("Thought:") — see Prompt Engineering Theory for depth.

### Generating Dynamic Tool Descriptions in Python

File: `3_raw_react_prompt.py`. Delete JSON schemas. Build text descriptions with `inspect`:

```python
import re
import inspect

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())
```

#### How it works

1. **`__wrapped__`** — if `@traceable` wrapped the function, signature would otherwise include tracer kwargs; unwrap to the real `(product: str) -> float`.  
2. **`inspect.signature` / `getdoc`** — metadata the LLM needs, previously supplied by `@tool` or JSON schema.  
3. Inject into the ReAct f-string along with defensive STRICT RULES from Layer 1.

### Understanding the ReAct Prompt: Building Agents Without Function Calling

[No dedicated transcript.] Conceptually: the LLM is **not** told it is an agent via API. Agency = prompt format + your parser. The model continues the text after `Thought:`; you stop it before it invents Observation.

### Implementing Manual Tool Calling for LLMs

Critical Ollama option: **`stop=["\nObservation"]`**.

```python
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


prompt = react_prompt.format(question=question)
scratchpad = ""

full_prompt = prompt + scratchpad
response = ollama_chat_traced(
    model=MODEL,
    messages=[{"role": "user", "content": full_prompt}],
    options={"stop": ["\nObservation"], "temperature": 0},
)
output = response.message.content
```

Without the stop token, the model often **hallucinates Observation** (fake tool results). With it, generation ends after Action Input; **you** inject the real observation.

Single user message contains instructions + tools + question + scratchpad — no separate system/user tool-calling protocol.

### Agent Loop With ReAct Prompt

[No transcript; reconstructed from course code.]

```python
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        full_prompt = prompt + scratchpad
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            return final_answer_match.group(1).strip()

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)
        if not action_match or not action_input_match:
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. ..."
        else:
            observation = str(tools[tool_name](*args))

        # Note: apply_discount casts price=float(price) — regex args are strings
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    return None
```

#### How it works

1. **Final Answer arrow:** regex on `Final Answer:` → return.  
2. **Tool arrow:** regex `Action:` + `Action Input:` → parse comma-separated / `key=value` args → call `tools[name](*args)`.  
3. **Scratchpad:** append model output + real Observation + `Thought:` so the next completion continues CoT.  
4. **Fragility:** if the model drifts from the format, regex fails — the historical motivation for native function calling.

```
              ┌──────────────────────────────────────┐
              │           full_prompt (text)         │
              └──────────────────┬───────────────────┘
                                 ▼
                         LLM (no tools=)
                                 │
              ┌──────────────────┴───────────────────┐
              │                                      │
      "Final Answer: ..."                   "Action: ..."
              │                                      │
           return                          regex parse args
                                                     │
                                              execute tool
                                                     │
                              scratchpad += output + Observation
                                                     │
                                              loop ──┘
```

### Cross-Layer Comparison (Same Agent)

| Step | Layer 1 LangChain | Layer 2 Raw FC | Layer 3 ReAct |
|------|-------------------|----------------|---------------|
| Reason | Structured `tool_calls` | Structured `tool_calls` | Text Thought/Action |
| Parse | `ai_message.tool_calls[0]` | `.function.name/arguments` | Regex |
| Execute | `tool.invoke` | `fn(**args)` | `fn(*args)` |
| Observe | `ToolMessage` | `{"role":"tool"}` | Scratchpad string |
| Finish | No tool_calls | No tool_calls | `Final Answer:` in text |

### Test Yourself

**Q1.** What does the stop token `\nObservation` prevent?  
**Q2.** What is the agent scratchpad in Layer 3?  
**Q3.** Why is ReAct-prompt tool calling less reliable than native function calling?

<details>
<summary>Answers</summary>

1. The model **hallucinating** Observation / fake tool results.  
2. A growing **string** of Thought/Action/Observation history re-sent each iteration.  
3. Free-text format is **brittle to parse**; FC returns schema-constrained structured calls.

</details>

---

## Bridge: [Theory] Understanding Function Calling for LLMs

*(Outline Section 8 / lecture 41 — included here because it closes the Layer 1–3 story.)*

**Function / tool calling** = model ability to emit a **structured function call** (name + args) in a dedicated response channel — not only prose in `content`.

- Capability of **some** models (now standard on SOTA OpenAI / Anthropic / Google).  
- Introduced by OpenAI in **2023**: you pass function definitions; model may return JSON specifying call + args.  
- Models are **fine-tuned** to detect when a function should run and to adhere to schemas.  
- App parses JSON → executes real function → feeds result back (the agent loop you already built).

**Motivation:** ReAct prompts are powerful but **unreliable** (bad formatting → parse failures). Function calling moves heavy lifting to the vendor → **deterministic, machine-readable** tool use.

**Two big wins:**

1. Connect LLMs to external tools reliably  
2. **Structured output** (same machinery as Section 3's `response_format`)

**Advantages vs ReAct prompt:** structured, fewer format errors, **token-efficient** (less verbose CoT in the wire format).

**Drawback:** **opaque reasoning** — you see the call, not always the justification (harder audit/debug). Still the industry default; almost nobody ships production agents on raw ReAct prompts alone anymore.

```
ReAct prompt era          Function calling era
─────────────────         ────────────────────
Text → regex → tool       Schema → JSON call → tool
Fragile, visible CoT      Reliable, opaque CoT
```

### Test Yourself (Function Calling Theory)

**Q1.** When did OpenAI introduce function calling to mainstream LLM APIs?  
**Q2.** Name one advantage and one drawback of FC vs ReAct prompts.  
**Q3.** How does FC relate to structured output / Pydantic agent responses?

<details>
<summary>Answers</summary>

1. **2023**.  
2. Advantage: reliable structured calls / fewer parse errors / fewer tokens. Drawback: **opaque** intermediate reasoning.  
3. Structured output often uses the same schema-constrained generation / tool-strategy machinery as FC.

</details>

---

## Appendix A — Project Map for These Sections

| Project folder | Maps to |
|----------------|---------|
| `course-code/project-hello-world/` | Section 2 Hello World / LCEL |
| `course-code/project-search-agent/` | Section 3 `create_agent` + Tavily + structured output |
| `course-code/project-ReAct-search-agent/` | Related search agent + `REACT_PROMPT` / schemas |
| `course-code/project-agents-under-the-hood/` | Sections 4–7 Layers 1–3 |
| `course-code/project-ReAct-Algo/` | Extra bind_tools loop + callbacks |

## Appendix B — Suggested Hands-On Checklist

- [ ] Run Hello World with OpenAI and with Ollama; compare LangSmith traces  
- [ ] Build stub search agent → Tavily custom tool → `TavilySearch` → Pydantic `response_format`  
- [ ] Run all three `project-agents-under-the-hood` scripts on the same laptop/gold question  
- [ ] Remove defensive rules in Layer 1 and note failure modes  
- [ ] Remove `stop=["\nObservation"]` in Layer 3 and inspect hallucinated observations  
- [ ] Sketch the ReAct diagram and label Thought / Action / Observation / Final Answer in each file  

---

*End of notebook part 01. Next course block: RAG (embeddings, vector DBs, retrieval) starting ~lecture "Introduction to Retrieval Augmentation Generation".*
