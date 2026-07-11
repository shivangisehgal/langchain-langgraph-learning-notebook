# Sections 17–28 — MCP, Deep Agents, Glossary, Industry Insights & Security

Self-contained learning notebook for Eden Marco's *LangChain — Agentic AI Engineering with LangChain & LangGraph*. Covers Model Context Protocol theory and practice, developer tooling, Deep Agents + Skills, the LangChain glossary, production insights from Assaf Elovic and Roy Miara, LLM app security, and the failure modes of vibe-coded apps.

---

## 17. Introduction to Model Context Protocol (MCP)

MCP (Model Context Protocol) is Anthropic's open standard for how AI *applications* (not raw models) discover and consume context: tools, resources, and prompts. The engineering insight is classic CS: when N agents each need M integrations, you do not write N×M custom adapters — you add a protocol layer and integrate once.

### Why MCP (Model Context Protocol)

Without MCP, every agent that wants Slack, Gmail, or a DB must wrap vendor APIs as local tools. That works for one product. The moment Windsurf, Claude Desktop, Cursor, Copilot, Lovable, and Bolt all want the same capability, you rewrite the integration for each host.

MCP flips the economics:

1. You implement capability once as an **MCP server**.
2. Any **MCP host** (Cursor, Claude Desktop, your LangGraph agent, etc.) connects via an **MCP client**.
3. Network effects kick in: thousands of community and official servers (Stripe, Cloudflare, docs fetchers, Uber Eats demos…) become plug-and-play.

Eden's social-media analogy holds: a protocol with few participants is interesting; a protocol with millions of consumers and producers is infrastructure.

💡 **Extended Notes**

- MCP is **not** a replacement for LangChain. LangChain solves orchestration, LCEL, memory, RAG, graphs. MCP solves *standardized tool/resource exposure across hosts*. They compose: LangChain agents can consume MCP tools via adapters.
- Think USB-C: the host (laptop) and the peripheral (device) agree on a connector. The model never speaks MCP directly; the application does.
- Prefer official vendor MCP servers over reinventing Stripe/Gmail wrappers. Supply-chain risk is real — fake "Stripe MCP" repos are an attack vector. Prefer verified registries when they mature.

### How LLMs Really Use Tools: Understanding Tool Calling

LLMs are token predictors. They cannot "call Slack." Tool use is application-layer behavior:

1. The host injects tool schemas into the prompt / API payload.
2. The model emits a structured tool call (name + args) instead of a final answer.
3. The application parses that call, executes real code, and feeds the observation back.
4. The model continues until it produces a user-facing answer.

Vendors differ on wire format (OpenAI tools, Anthropic tool_use, ReAct text prompts), but the loop is the same. Reliability is statistical — good enough for agents, never 100%.

MCP's job: let you *author* those tools once and let any tool-calling host discover and invoke them.

### MCP Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP HOST                             │
│   (Cursor / Claude Desktop / your LangGraph agent)          │
│                                                             │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │ MCP Client A │  │ MCP Client B │  │ MCP Client C │    │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└──────────┼─────────────────┼─────────────────┼────────────┘
           │ 1:1             │ 1:1             │ 1:1
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │ MCP Server  │   │ MCP Server  │   │ MCP Server  │
    │  (weather)  │   │  (math)     │   │  (docs)     │
    │ tools/      │   │ tools/      │   │ tools/      │
    │ resources/  │   │ resources/  │   │ resources/  │
    │ prompts/    │   │ prompts/    │   │ prompts/    │
    └─────────────┘   └─────────────┘   └─────────────┘
```

**Components**

| Role | Responsibility |
|------|----------------|
| **Host** | AI application that owns the LLM session and UX |
| **Client** | Lives *inside* the host; 1:1 with a server; speaks the protocol |
| **Server** | Exposes tools, resources, prompts; executes tools in *its* runtime |

**Why execute tools on the server?** Decoupling. Tool runtime can scale independently (K8s, serverless), log separately, update dynamically without redeploying the agent. Orchestration stays in the host; execution stays in the server.

### The GIST of the Protocol with Tool Calling

**Boot sequence (before any user message)**

1. Host starts → clients initialize connections to configured servers (stdio / SSE / streamable HTTP).
2. Server advertises capabilities: `list_tools`, `list_resources`, `list_prompts`, …
3. Host caches those schemas.

**Request sequence**

```
User ──query──▶ Host
                  │
                  ├─ augment query with discovered tool schemas
                  ▼
                LLM ──tool_call(name, args)──▶ Host
                  ▲                              │
                  │                              ▼
                  │                         MCP Client
                  │                              │
                  │                              ▼
                  │                         MCP Server ── executes tool
                  │                              │
                  └── tool result ◀──────────────┘
                LLM ── final answer ──▶ User
```

Key contrast with vanilla LangChain ReAct: in LangChain, tools usually run *inside* the agent process. With MCP, the host forwards the call; the **server** runs the function. LangChain can still *orchestrate* while MCP *executes*.

Dynamic tools: if clients re-initialize periodically, agents can pick up new tools without redeploy — useful for platform products.

### MCP Servers

Servers are wrappers that federate access to systems through three surfaces:

1. **Tools** — model-controlled functions (`get_weather`, `fetch_docs`). Full freedom: read/write APIs, side effects.
2. **Resources** — application-controlled data (PDFs, JSON, dynamic URLs). The app decides what to pull into context.
3. **Prompts** — user-controlled templates for standardized complex interactions.

**How to obtain servers**

- Hand-write with the MCP SDK (Python / Node)
- Generate with AI coding agents
- Clone community servers
- Use official vendor servers (Stripe, Cloudflare, …)

**Transports**

- **stdio** — local process, stdin/stdout (common for desktop hosts)
- **SSE / streamable HTTP** — remote, cloud-friendly
- **Docker** — package the server once, run anywhere

**Sampling** — server can ask the *host* to complete a prompt (powerful; security-sensitive).

**Composability** — an app can be both client and server → multi-layer agent systems.

**Future** — central registry/discovery, official verification (supply-chain defense), `.well-known` capability endpoints for websites (robots.txt for agents), OAuth 2.0 / session tokens.

### Test Yourself — Section 17

1. Why is MCP described as solving an N×M integration problem?
2. Where does tool *execution* happen in MCP vs a vanilla LangChain ReAct agent?
3. What is the cardinality between MCP clients and servers, and how do hosts talk to many servers?
4. Name the three primary surfaces an MCP server can expose.
5. What is "sampling," and why does it raise security concerns?

---

## 18. Using a Pre-built Server (mcpdoc) with AI Clients (Cursor & Claude)

Goal of this section: consume a **pre-built server** from **pre-built clients** so the protocol is tangible before you write anything.

### What are we building? MCP Doc

**mcpdoc** (LangChain) is an MCP server that keeps Cursor / Claude Desktop / Windsurf plugged into *fresh* LangChain & LangGraph docs via `llms.txt` indexes — docs that otherwise go stale between model training cutoffs.

Pattern:

1. Client connects to mcpdoc.
2. Agent lists doc sources → gets `llms.txt` URL(s).
3. Agent fetches the index → picks the relevant page URL.
4. Agent fetches that page → answers grounded in live docs.

### MCP Inspector

Anthropic's open-source **MCP Inspector** (`npx`-runnable) is the debugger you want while building servers:

- Connect to stdio or SSE servers
- **Resources** tab — list, metadata, content
- **Prompts** tab — templates + custom args
- **Tools** tab — schemas + live invoke
- **Notifications** — server logs

Use it as a unit-test UI before wiring Cursor/Claude.

```bash
# Typical local SSE inspect flow
# Terminal 1: run your MCP server (e.g. port 8082)
# Terminal 2:
npx @modelcontextprotocol/inspector
# Point inspector at http://localhost:8082 (SSE) and List Tools
```

### LLM.txt

`llms.txt` is a machine-readable Markdown map of a site — URLs + short descriptions — usually at the site root. Not an official IETF standard; widely adopted in GenAI docs (LangGraph ships both Python and JS variants).

LangChain often provides two flavors:

| File | Contents | Best for |
|------|----------|----------|
| `llms.txt` | Index: URLs + blurbs | Agent + scraper: pick page, fetch on demand (RAG-like, real-time, higher latency) |
| `llms-full.txt` | Full page text dump | Chunk + embed offline, or stuff into huge-context / cached models |

💡 **Extended Notes**

- Treat `llms.txt` as a **sitemap for agents**. Your docs site should expose one if agents are a distribution channel.
- Latency tradeoff: abbreviated + fetch is freshest but multi-hop (list → fetch index → fetch page → answer). Full dump is faster at query time after indexing, but drifts unless you re-crawl.
- Pair with Firecrawl / Tavily extract patterns when the agent needs arbitrary pages beyond the curated index.

### mcpdoc

Hands-on flow from the course:

1. Clone mcpdoc; `uv` venv + install.
2. Run locally against LangGraph's `llms.txt` (SSE on ~8082).
3. Sanity-check with MCP Inspector: tools `list_doc_sources` and `fetch_docs`.
4. Wire Claude Desktop via `claude_desktop_config.json` — prefer **absolute paths** for `uvx` / project dirs (relative paths cause `ENOENT`).
5. Restart host; ask *"What is LangGraph memory?"*
6. Observe tool chain: `list_doc_sources` → `fetch_docs(llms.txt)` → `fetch_docs(memory page)` → grounded answer.

Transport note: same server can speak **SSE** (Inspector / remote) or **stdio** (desktop hosts). Config chooses the transport.

```json
// Claude Desktop-style config sketch (paths must be absolute on your machine)
{
  "mcpServers": {
    "mcpdoc": {
      "command": "/absolute/path/to/uvx",
      "args": [
        "--from", "/absolute/path/to/mcpdoc",
        "mcpdoc",
        "--urls", "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt",
        "--transport", "stdio"
      ]
    }
  }
}
```

**How it works:** The host's MCP client spawns/connects to mcpdoc. On each user question about LangGraph, the model chooses tools from the advertised schemas. Execution stays in mcpdoc (HTTP fetch). The host only relays results into the next LLM turn — classic MCP decoupling with a documentation specialty.

### Test Yourself — Section 18

1. What two tools does mcpdoc typically expose, and in what order does a successful "memory" question use them?
2. When would you prefer `llms.txt` over `llms-full.txt`?
3. Why does Claude Desktop often fail with `ENOENT` on first MCP setup?
4. What is MCP Inspector for, and when should you use it vs jumping straight to Cursor?

---

## 19. Building MCP Servers and Clients with LangChain

### Intro

You now switch roles: build servers yourself, then consume them from a LangChain/LangGraph client via **langchain-mcp-adapters**.

### Boilerplate

Course setup pattern:

```bash
uv init
uv venv
source .venv/bin/activate
uv add langchain-mcp-adapters langgraph langchain-openai python-dotenv
```

`.env` holds model keys + optional LangSmith (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`). Always gitignore `.env`.

Skeleton:

```python
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    print("hello langchain mcp")
    # sanity: print(os.environ.get("OPENAI_API_KEY", "")[:8])

if __name__ == "__main__":
    asyncio.run(main())
```

Any function-calling model works (OpenAI, Anthropic, Gemini free tier, DeepSeek, …).

### Servers

Two teaching servers:

1. **math_server.py** — `add`, `multiply`; transport **stdio**
2. **weather_server.py** — dummy `get_weather` → `"hot as hell"`; transport **SSE** (HTTP)

Naming pitfall: do **not** name a module `math.py` — it shadows Python's stdlib `math`.

```python
# servers/math_server.py — pattern
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

```python
# servers/weather_server.py — pattern
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
def get_weather(city: str) -> str:
    """Return a weather string for the city (demo stub)."""
    return f"It is hot as hell in {city}"

if __name__ == "__main__":
    mcp.run(transport="sse")  # e.g. http://localhost:8000
```

```bash
uv run servers/math_server.py      # waits on stdio
uv run servers/weather_server.py   # listens on :8000
```

### What are we MCBuilding?

Focus: wire the **SSE** weather server (cloud-deployable) plus the **stdio** math server into one LangChain multi-server client. Enterprise angle: SSE servers live in your VPC; org agents call them. AuthN/AuthZ / RBAC for MCP is still maturing — treat open remote servers as hostile until OAuth lands properly.

### Simple MCP Server / Client Scaffold

```python
# langchain_client.py — scaffold
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

async def main():
    print("hello langchain mcp")

if __name__ == "__main__":
    asyncio.run(main())
```

`MultiServerMCPClient` hides the 1:1 client↔server rule: under the hood it still creates one client per server; you get one ergonomic API.

### Bridging the Gap: The LangChain MCP Adapter Explained

**Similarities**

- Both MCP and LangChain treat tools as typed functions with descriptions that the model reads.
- MCP server ≈ LangChain toolkit (a bag of tools).

**Differences**

| | LangChain `bind_tools` | MCP |
|--|------------------------|-----|
| Surfaces | Tools (primary) | Tools + resources + prompts |
| Binding target | An LLM object | An AI *application* / host |
| Execution locus | Usually in-process | Server process / remote |

**Adapter value**

- Convert MCP tools → LangChain/LangGraph-compatible tools
- Reuse community MCP servers without rewriting
- Multi-server client for one agent session

### Imports / Client / Tracing

💡 **Extended Notes** — these lectures have no transcript; patterns below match course intent and current `langchain-mcp-adapters` usage.

```python
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "uv",
                "args": ["run", "servers/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
        }
    )

    tools = await client.get_tools()
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = create_react_agent(llm, tools)

    result = await agent.ainvoke(
        {"messages": [("user", "What is (3+5)*2, and what's the weather in Dubai?")]}
    )
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
```

**How it works**

1. Client starts stdio subprocess for math; opens SSE to weather.
2. `get_tools()` lists tools from both servers and wraps them as LangChain tools.
3. ReAct agent binds those tools; model may call `add`/`multiply`/`get_weather`.
4. Each tool call is proxied to the owning MCP server; observations return into the graph.
5. With LangSmith env vars set, you see tool spans labeled by server — invaluable when debugging multi-server fan-out.

Tracing checklist:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=...
export LANGCHAIN_PROJECT=mcp-adapters
```

Inspect: which server answered, latency per tool, whether the model chose the wrong tool due to weak descriptions.

#### Imports (what each package is doing)

| Import | Role |
|--------|------|
| `MultiServerMCPClient` | Spawns/connects N MCP clients; `get_tools()` → LangChain tools |
| `create_react_agent` | LangGraph prebuilt ReAct loop over those tools |
| `ChatOpenAI` (or Anthropic/Gemini) | Any tool-calling chat model |
| `load_dotenv` | Local secrets without hardcoding |

If the weather SSE server is not already running, start it in another terminal before the client. Stdio servers are usually spawned *by* the client config (`command`/`args`), so you do not need a separate math process when using that pattern.

#### Client lifecycle notes

```python
# Alternative: async context manager style (versions vary — check adapters docs)
async with MultiServerMCPClient({...}) as client:
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)
    ...
```

**How it works (end-to-end for `(3+5)*2` + weather):**

1. Model sees tool schemas for `add`, `multiply`, `get_weather` (descriptions matter — write them like API docs for an intern).
2. Likely plan: `add(3,5)` → `multiply(8,2)` → `get_weather("Dubai")` (order may vary).
3. Math calls travel stdio JSON-RPC to the child process; weather calls POST/SSE to `:8000`.
4. Observations concatenate into message history; final AI message answers both questions.
5. LangSmith shows separate tool runs — if weather hangs, you see SSE latency isolated from math.

#### Tracing pitfalls

- Forgetting to export tracing vars → silent local-only runs.
- Mixing `LANGCHAIN_*` vs `LANGSMITH_*` env names across Deep Agents CLI vs classic LangChain — read the error from `/trace` or the docs for *that* harness.
- Project name filters: if you do not see runs, check you are in the right LangSmith project and time range.

### Test Yourself — Section 19

1. Why must you avoid naming a server module `math.py`?
2. What does `MultiServerMCPClient` abstract, and what does it *not* change about the protocol?
3. Give one reason to prefer SSE over stdio for an enterprise-shared tool.
4. Sketch the adapter's role between an MCP tool schema and `create_react_agent`.

---

## 20. Useful tools when developing LLM Applications

### Stop Writing Deprecated Code: LangChain's Official MCP Server

Coding agents train on yesterday's LangChain. APIs deprecate (`initialize_agent`, old `create_react_agent` paths, …). LangChain ships a **public Docs MCP server** (streamable HTTP, no API key) with tool `SearchDocsByLangChain`.

Install from docs "Copy MCP Server" into Cursor. Side-by-side demos in the course: with Docs MCP → modern `create_agent`; without → deprecated AgentExecutor patterns via generic web search.

Also: [chat.langchain.com](https://chat.langchain.com) uses the same search tool under the hood — traces show `SearchDocsByLangChain`.

**Rule:** When generating LangChain code with Cursor/Claude Code, enable LangChain Docs MCP (or Context7 for general libs; Docs MCP for LangChain-specific accuracy).

### LangChain Hub — Downloads prompt from the community

LangSmith Hub is a shared registry of prompts (agents, RAG QA, SQL, classification, …), filterable by use case and model family.

```python
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")  # example identifier
# Pass into your chain / agent as the prompt template
```

Playground: plug variables, compare vendors, inspect commit history of a prompt. Treat Hub like npm for prompts — reuse, then customize.

### TextSplitting Playground

Chunking is under-specified science: size, overlap, and splitter choice are data-dependent. LangChain's **Text Splitter Playground** (Streamlit, open source) lets you paste text, tweak `chunk_size` / `chunk_overlap` / splitter type, and *see* chunks + generated code.

Use it before locking ingestion hyperparameters. Visualize overlap boundaries so you do not split mid-thought on your corpus.

### LangChain vs LlamaIndex

| Dimension | LangChain | LlamaIndex |
|-----------|-----------|------------|
| Popularity / ecosystem | Broader adoption | Strong niche |
| Center of gravity | Agents + LCEL + graphs + RAG | Data / retrieval-first |
| Agents | Deep, research-active, LangGraph | Present; historically RAG-centric |
| Eden's default | Prefer LangChain even for RAG-heavy apps | Viable; less preferred here |

Both can build LLM apps. If the product is agentic, LangChain/LangGraph is the clearer bet in this course's framing.

### Test Yourself — Section 20

1. Why do coding agents emit deprecated LangChain APIs, and what MCP mitigates that?
2. What does `hub.pull` give you operationally?
3. Name two chunking parameters you should validate in the Text Splitter Playground before production ingestion.
4. When might LlamaIndex still be a reasonable choice?

---

## 21. Deep Agents

### Introduction to Deep Agents Section

**Deep agents** tackle long-horizon work: multi-minute/hour research, feature implementation, browse+test loops. Coding agents (Claude Code, Cursor CLI, Devin) are the industrial proof. This section taxonomizes them, then studies LangChain's open-source **Deep Agents** harness — rare because most coding agents are closed source.

### Taxonomy: Shallow Agents, Deep Agents, Coding Agents

```
                    ┌──────────────────────────┐
                    │      Agents (umbrella)   │
                    └────────────┬─────────────┘
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
   ┌───────────────┐    ┌────────────────┐    ┌────────────────┐
   │ Shallow/ReAct │    │  Deep Agents   │    │ Agentic apps   │
   │ short loops   │    │ long-horizon   │    │ (hybrid RAG…)  │
   └───────────────┘    └───────┬────────┘    └────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
           ┌────────────────┐      ┌─────────────────┐
           │ Deep research  │      │ Coding agents   │
           │ (Perplexity,   │      │ (Claude Code,   │
           │  GPT Researcher│      │  Cursor, Devin) │
           └────────────────┘      └─────────────────┘
```

**Shallow (classic ReAct)** — excellent for "book a flight"-scale tasks. Context grows each tool iteration → **context rot** (confusion, contradiction, pollution), cost, latency. Fine for many production agents; insufficient for deep research / multi-file features.

**Deep agents** — long-running, interruptible, user-in-the-loop capable. Four recurring capabilities:

1. **Planning tool** (dynamic to-do)
2. **Subagents** (hierarchical delegation, isolated context)
3. **Filesystem** (persist intermediate state off-prompt)
4. **Large system prompt** (harness instructions)

Innovation note: model quality improves gradually; harness / application-layer engineering is where leaps (Claude Code-class systems) currently come from.

### Dynamic To-Do Lists

Deep agents do **not** rely only on implicit CoT. They maintain an **explicit** markdown to-do: pending / in_progress / completed, revised between steps. Failures get replanned instead of blind ReAct retries. Users can influence the list; some products (Claude Code) keep the planner internal but visible in traces (`update_todo`).

Human parallel: break work down, track progress, get the dopamine of checking boxes — and keep the agent oriented across hours of execution.

```text
# Example living plan (conceptual)
- [x] Explore auth middleware patterns
- [x] Draft failing test for IDOR on /invoices/:id
- [ ] Implement org_id filter in repository
- [ ] Run e2e + update changelog
```

**How it works:** The plan is a *tool-backed artifact*, not vibes in the residual stream. Between tool batches the agent reads/updates the list, so a failed step becomes a new pending item with a revised approach instead of an infinite retry of the same call. That is the difference between "agentic" and "stubborn."

For research agents the to-do might be: outline questions → search → read → synthesize → cite. For coding agents: reproduce → locate → patch → test → lint. Same mechanism, different domains.

### Sub Agents and Hierarchical Delegation

```
                    ┌─────────────────────┐
                    │     Main Agent      │
                    │  (orchestrator)     │
                    │  lean context       │
                    └──────────┬──────────┘
                               │ task brief
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │ Subagent A   │ │ Subagent B   │ │ Subagent C   │
      │ own prompt   │ │ own prompt   │ │ own prompt   │
      │ own tools    │ │ own tools    │ │ own tools    │
      │ own ReAct    │ │ own ReAct    │ │ own ReAct    │
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             │                │                │
             └──────── condensed result only ──┘
                               ▼
                    ┌─────────────────────┐
                    │   Main continues    │
                    └─────────────────────┘
```

Eden's father-in-law analogy: you delegate a roof fix with a clear brief; the specialist brings tools and skills; you receive the result without absorbing every intermediate detail — **context isolation**.

Benefits: parallel work, specialized prompts/tools, no pollution of the main thread with exploratory tool spam.

### Subagents Context Flow

- Main thread tokens accumulate every user/assistant turn.
- Spawning a subagent sends **only** a crafted brief — not full history.
- Subagent burns its own context window; returns one condensed artifact (summary + diffs).
- Main stays lean longer → fewer `/compact` / `/clear` rituals.

Prompt quality to the subagent *is* the product quality. Invest in how the main agent phrases delegation.

Finite context (200K / 1M / …) always exists. Subagents raise effective capacity by side-chaining work.

### Deep Agents File Systems

Deep Agents expose Claude Code–like FS tools: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`. Backends are pluggable (local disk, Firestore, DynamoDB, …) — interface first.

Context-engineering diagram (LangChain blog framing):

- **Blue** — all available knowledge (repos, web, DBs)
- **Red** — what the agent pulled into the window
- **Green** — what the task actually needs

Failure modes: under-retrieval, over-retrieval (noise), misaligned retrieval, hard window limits. Sweet spot: smallest red disk that covers green.

Filesystem enables two context-engineering primitives:

1. **Write** — park intermediates on disk instead of the prompt
2. **Select** — `glob`/`grep`/`read` to pull only green-relevant bits

```python
# Conceptual deep-agent FS tool surface (matches course / Claude Code shape)
TOOLS = [
    "ls", "read_file", "write_file", "edit_file", "glob", "grep",
    "write_todos",  # planning
    "task",         # spawn subagent with brief
]
```

**How it works in a long coding task:** Agent greps for `authorize(`, writes a scratch `notes/auth-findings.md`, spawns an explore subagent with that note as the brief, receives a condensed report, edits production files with `edit_file`, runs tests via shell tool, updates todos. The main prompt never contains every grepped line — only what was selected into notes and summaries.

### Test Yourself — Section 21

1. Why can a tool-rich ReAct agent still be "shallow"?
2. List the four common deep-agent capabilities.
3. How do subagents compress context for the main agent?
4. Map `glob`/`grep`/`write_file` to write-vs-select context engineering.
5. Why is application-layer harness work currently outpacing raw model jumps for coding agents?

---

## 22. Deep Agents Skills

### The 3 Layers of AI Agent Skills: From Usage to Source Code

Skills = packaged expertise (markdown + assets) that agents load **progressively**. Study path:

| Layer | What you do | Tooling |
|-------|-------------|---------|
| 1 | Use skills as an end user | Deep Agents CLI |
| 2 | Observe disclosure in traces | LangSmith |
| 3 | Read the middleware | `skills.py` in deepagents |

Closed agents hide this; Deep Agents is open source — reverse-engineer the pattern for your own harness.

### Level 1: Using Agent Skills in the Deep Agents CLI

```bash
uv tool install deepagents   # or project install per docs
export ANTHROPIC_API_KEY=...
deepagents                   # interactive harness
```

Install a skill (course demo: Remotion best practices):

```bash
npx skills add remotion-dev/skills
# Prefer universal .agents/skills (Deep Agents, many CLIs)
# Claude Code may use .claude/skills — install targets matter
# Choose global vs project scope
```

Ask *"which skills do you have?"* → Skill Creator, Find Skills, Remotion best practices, …

Ask *"create a Remotion video on agent skills"* → agent **reads** `SKILL.md` and related rule files only when relevant → scaffolds React/Remotion project → renders.

You just experienced **progressive disclosure** from the outside.

### Layer 2: Tracing AI Agent Skills with LangSmith

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGCHAIN_PROJECT="my-deep-agent-execution"
deepagents
# /trace — confirm tracing enabled (docs may lag; env names matter)
```

What traces show:

1. **before_agent middleware** — discover skills once; put metadata (name, description, path) in agent state. Full bodies stay on disk.
2. On a plain "hello", system prompt lists skill *metadata* + instructions on progressive disclosure — no Remotion rules yet.
3. On "make a GIF with Remotion", model `read`s `SKILL.md`, then selectively reads `rules/*.md` (animations, compositions, timings… — maybe not `gifs.md`; the LLM chooses).
4. **wrap_model_call / skills middleware** — before each LLM call, inject the skills appendix into the system prompt from state metadata.

Two middleware moments:

- Discovery (session start) ≠ Injection (every model call)
- Decision to deepen into files = **model**, not harness

### RECAP — Skill Middleware

Standard ReAct loop, plus:

1. Session start → load skill metadata into state (`before_agent`)
2. Before each LLM call → append skills system appendix (`wrap_model_call`)
3. Model may `read_file` skill paths → content enters messages → next reason step

Harness prepares the menu; the model orders the dishes.

### Layer 3: Inside skills.py — Mechanics of Progressive Disclosure

```
Session start
    │
    ▼
before_agent ── list_skills(sources) ── parse YAML front matter
    │                                    update state.skills_metadata
    │
    ▼
┌──────────── agent loop ────────────┐
│  wrap_model_call                    │
│    modify_request:                  │
│      append SKILLS_SYSTEM_PROMPT    │
│      with skills_list + locations   │
│  LLM decides: answer | tool | read  │
│  tools: read_file / grep / …        │
└─────────────────────────────────────┘
```

Implementation notes from the course walkthrough:

- ~800 lines; mostly FS traversal + YAML front matter parsing — smart engineering, not exotic ML.
- `backend` abstraction: local FS today; Firestore/Bigtable-shaped skills tomorrow.
- Source order: later sources override same-named skills (harness policy choice).
- `SKILL.md` should act as an **index** so the model can pick the right rule file.
- Progressive disclosure beauty: harness never hardcodes "always load gifs.md"; the LLM routes.

```python
# Conceptual shape (illustrative — not a verbatim copy of deepagents)
class SkillsMiddleware:
    def before_agent(self, state):
        if state.get("skills_metadata"):
            return state  # already loaded this session
        meta = list_skills(self.sources, backend=self.backend)
        return {**state, "skills_metadata": meta}

    def wrap_model_call(self, request, state):
        appendix = render_skills_prompt(state["skills_metadata"])
        request.system = request.system + "\n\n" + appendix
        return request
```

**How it works:** Metadata is cheap and always present. Full instructions are opt-in via ordinary filesystem tools. That keeps the default context lean while allowing arbitrarily deep skill packs (Remotion rules, brand guidelines, internal runbooks).

### Test Yourself — Section 22

1. Define progressive disclosure for agent skills in one sentence.
2. What is loaded at session start vs on a Remotion GIF request?
3. Distinguish `before_agent` discovery from `wrap_model_call` injection.
4. Why should `SKILL.md` read like an index?
5. Where does responsibility lie for choosing which rule markdown files to open?

---

## 23. LangChain Glossary

Quick reference for objects you use constantly. Treat this as a field manual, not a tutorial redo.

### ChatModels

Primary interface to conversational LLMs (OpenAI, Anthropic, Gemini, Ollama, …). Input: list of messages. Output: an AI message.

Capabilities beyond text: **tool calling**, **structured output** (`with_structured_output`), **multimodality**, async / batch / stream, LangSmith integration.

Common methods: `invoke`, `stream`, `batch`, `bind_tools`, `with_structured_output`.

Init knobs (standardized where possible): `model`, `temperature`, `max_tokens`, `stop`, `timeout`, `max_retries`, API key / base URL. Always check provider-specific extras (e.g. Gemini).

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
ai_msg = llm.invoke([("system", "Be concise."), ("human", "Define LCEL in one line.")])
```

### Messages

Unit of chat I/O. Every message has **role** + **content** (text or multimodal blocks).

| Role | LangChain class | Purpose |
|------|-----------------|--------|
| system | `SystemMessage` | Behavior / context |
| user | `HumanMessage` | User or upstream system input (bare strings coerce to this) |
| assistant | `AIMessage` | Model output (+ tool_calls, usage metadata) |
| tool | `ToolMessage` | Tool result tied to a tool_call_id |

Ordering matters. Tool loop: Human → AI(tool_calls) → Tool → AI(final).

### RecursiveCharacterTextSplitter

Splits documents by trying separators from large semantic units down to characters: `\n\n` → `\n` → spaces → chars. Heuristic to keep related text together vs naive fixed-length cuts. Still a heuristic — validate on your corpus (Playground).

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
```

### Document

Standard container: `page_content` (str) + `metadata` (dict: source, page, URL, tags…). Loaders emit `Document`s; splitters emit smaller `Document`s. Metadata fuels filtered retrieval later.

### Token Limitation Strategies

Token budgets cover **prompt + completion**. Strategies for oversize inputs (classic summarization teaching example via `load_summarize_chain`):

| Strategy | Idea | Pros | Cons |
|----------|------|------|------|
| **Stuff** | Dump all docs into one prompt | Simple, 1 call | Hits limits fast |
| **Map-reduce** | Summarize each doc (map), then summarize summaries (reduce) | Scales, parallelizable | Many calls, info loss |
| **Refine** | Fold/left: iteratively refine running summary with next doc | Preserves sequential nuance | Serial, slower |

💡 **Extended Notes** — map-reduce ≈ map then reduce; refine ≈ `foldl` with "combine & summarize" as the binary op. Same ideas apply beyond summarization whenever context exceeds the window.

### Memory Intro — Coreference Resolution

LLMs are **stateless**. Without history, "videos about *him*" fails after "Who created LangChain?" → Harrison Chase. **Coreference resolution** needs prior entities in context. Memory = sophisticated ways to re-inject history so the model can resolve pronouns and preferences — subject to token limits on long chats.

### Memory Deepdive (LangGraph)

What to store:

1. **Stuff all messages** — fine for short threads
2. **Trim** old messages (`trim_messages`) — heuristic drop
3. **Summarize** older turns; keep recent raw messages

Where to store: **LangGraph checkpointers** (`MemorySaver`, Postgres, Redis, Mongo, …). Checkpoint after turns; reload by `thread_id`.

```python
from langgraph.checkpoint.memory import MemorySaver
# graph.compile(checkpointer=MemorySaver())
# invoke(..., config={"configurable": {"thread_id": "user-123"}})
```

Prompt pattern: `MessagesPlaceholder("messages")` + inject history dict. Checkpointing persists; trimming/summarization *shapes* what gets persisted or sent.

### Test Yourself — Section 23

1. Which ChatModel methods would you use for a streaming chat UI vs a cron batch job?
2. Write the message role sequence for one successful tool call.
3. Contrast stuff vs map-reduce vs refine for a 200-page PDF summary.
4. Why is coreference a memory problem rather than a "smarter model" problem alone?
5. What does a LangGraph checkpointer actually persist?

---

## 24. Industry Insights: Building Production Agents with Assaf Elovic

Assaf Elovic — co-founder of Tavily, creator of GPT Researcher, former Head of AI at monday.com — on production AI in 2026.

### The Core Architecture of Production-Grade AI

Pillars he highlights:

1. **Observability for agents ≠ APM for click UIs.** You need stack traces of *reasoning and tool paths* (LangSmith-class), plus understanding of natural-language user intent and whether the agent succeeded.
2. **AI gateway** — guardrails, permissions, prompt security, model routing, failover when providers rate-limit or brown out. Smart routing by cost/latency/capability.
3. **Memory** — observe it, govern it, monitor cross-company context leakage.
4. **Semantic search / ranking** — "RAG" evolves; retrieval quality remains foundational.
5. Application topology itself is **use-case specific** — don't cargo-cult one graph.

### How to Make Users Trust Your AI Agents

Technical reliability ≠ perceived reliability. Assaf + Harrison Chase's **FAIR** framing for trust UX:

| Letter (mnemonic) | Practice |
|-------------------|----------|
| **Explainability** | Show how the agent reached an action — black boxes destroy trust after mistakes |
| **Transparency** | Reveal what tools/data were used |
| **Feedback loops** | Let users correct the agent so next runs improve — underrated |
| **Evals** | Developer-side suites for critical paths on every release |

Analogy: you trust coworkers who explain decisions, show their work, accept feedback, and get performance-reviewed.

### Tutorial: Building a Lean AI Feedback Loop

Minimum viable loop (days, not quarters):

1. Keep a per-user or per-tenant **Markdown** preference/memory file (starts empty).
2. User gives feedback in natural language.
3. **The agent** (not the human) updates the file.
4. Middleware injects that file into future contexts (before model or tool calls).

Eden's mapping: LangChain/LangGraph **middleware** before an LLM call can read/update the markdown and patch state — same mechanical idea as skills middleware, different content.

💡 **Extended Notes**

- Separate **product-level** vs **user-level** agents when designing whose markdown gets loaded.
- Version and size-limit these files; unconstrained append → context rot.
- Pair FAIR UX (show the note you updated) so users see their feedback "landed."

### Test Yourself — Section 24

1. Name two ways agent observability differs from classic product analytics.
2. What problem does an AI gateway solve when OpenAI returns 429s?
3. List the FAIR trust practices and give one UI affordance for each.
4. In the lean feedback loop, who writes the markdown file, and why does that matter?

---

## 25. Industry Insights: Building Production Agents with Roy Miara

Roy Miara — Member of Technical Staff at Tenzai (autonomous offensive-security agents); formerly Pinecone applications lead.

### Intro

Tenzai builds autonomous "hacker" agents for penetration testing / security validation — surface vulns before real adversaries. Modern LLMs are strong enough at cyber reasoning that this product class is viable; execution remains hard.

### AI Agents in Cybersecurity CTF Competitions

Once the agent worked, Tenzai entered **CTFs** (including agent-vs-agent races). Competitive loops drove rapid iteration — CEO passionately running overnight challenges, bringing failure modes back to the team next morning.

Principle: training a **top-1% agent** is like training an elite athlete, not "going to the gym casually." Domain passion + ruthless evaluation environments beat generic agent tutorials.

### Harness Engineering

Multiple architecture iterations. Warning sign: you start **patching** because the architecture no longer supports progress.

Hard lesson: naively sharding a hard problem across 10–100 agents does **not** yield 10–100× success. Agents are not aware of fleet scale; **breaking context carelessly** hurts.

What you actually manage:

- Stable environment so *one* agent can take on harder tasks
- When context may be cleared vs must stay concentrated
- Which subproblems are **delegatable** vs must stay on the main thread

Harness engineering = designing those policies deliberately.

### Managing Variance and Hallucinations in Production Agents

Coding agents: variance across runs is often acceptable (many valid patches).

Autonomous security products: **comprehensiveness** matters — missing a vuln is a product failure. Tension:

- High temperature / creativity → exploration of weird exploit paths
- Forced exhaustive checklists → coverage but less discovery of novel issues

Other products hard-code comprehensiveness and miss deep edges. Expert researchers want best-practice scripts; Roy's counter: once running, the agent may know more about the live system than the operator — over-constraining kills that advantage.

Design explicitly for the creativity ↔ exhaustiveness tradeoff; measure coverage empirically (CTF suites, regression corpuses).

### Test Yourself — Section 25

1. Why are CTFs a better eval flywheel than ad-hoc demos for a security agent?
2. What fails when you "just add more agents" to a large attack surface?
3. How does acceptable variance differ between coding assistants and vuln scanners?
4. State the creativity vs exhaustiveness tension in one sentence.

---

## 26. Agent Security

### What is LLM App Sec?

LLM apps inherit **all** classic AppSec concerns (auth, injection into backends, SSRF, …) **and** add a new attack surface: the model that accepts untrusted text/images and may emit tool calls.

Two product shapes:

1. **Autonomous agents** (Claude Code-class) — large blast radius if tools are powerful
2. **Agentic workflows** — human-defined graphs where the LLM chooses branches — still injectable

Threat themes Eden flags for the section:

- **Prompt injection** (direct)
- **Indirect prompt injection** (malicious content in retrieved docs/emails)
- **Tool hijacking** / confused deputy via tool descriptions or poisoned results
- Minimizing **blast radius** when compromise happens (least privilege tools, sandboxing, human approval for irreversible actions)

Career note: most engineers deprioritize security while shipping. This section's job is to make insecure defaults emotionally and technically expensive to ignore.

💡 **Extended Notes** — secure-by-default checklist

- Treat every tool as a privileged API: authz at the tool boundary, not "the model will be careful"
- Separate system prompts from untrusted retrieved content; never concatenate blindly
- Allowlist tool args (URLs, SQL, shell); prefer structured I/O over free-form shell
- Log tool calls; require confirmation for spend / delete / email-send
- Threat-model MCP servers like dependencies — supply chain + data exfil
- Red-team with indirect injection fixtures in your eval set

### Test Yourself — Section 26

1. What new attack surface does introducing an LLM add beyond classic web AppSec?
2. Give an example of indirect prompt injection in a RAG email assistant.
3. Define blast radius and one concrete way to shrink it for an agent with a DB tool.
4. Why is "we use a strong system prompt" insufficient as a security control?

---

## 27. The Dark Side of "Vibe Coding": Vulnerabilities in AI-Generated Apps

Transcripts for this block are missing or empty in the source dump. The notes below are expanded from lecture titles plus production AppSec experience with AI-generated codebases. **Vibe coding** = accepting LLM-generated app code with light review because it "looks right" and demos well.

### Introduction

💡 **Extended Notes**

AI coding agents excel at scaffolding CRUD apps, React pages, and glue code. They pattern-match from public repos — including insecure tutorials and Stack Overflow answers that prioritize "it works." Security failures cluster where **intent is implicit**:

- Who is allowed to do what (authorization)
- What "done" means for a business rule (invariants, abuse cases)
- Where trust boundaries cross networks (SSRF, webhooks, SSR)

The sociological problem: vibe coding optimizes for **demo latency**. Security requires **adversarial imagination**. Those objectives conflict unless process forces the second.

Ship checklist before celebrating a vibe-coded MVP:

1. Draw trust boundaries (browser / API / worker / LLM tools / third parties)
2. Name the principal on every mutating endpoint
3. List three ways a malicious user abuses the happy path
4. Confirm those abuses fail in automated tests

Demos pass; production gets popped — usually via IDOR, SSRF, or missing authz, not via a clever SQLi string from 2005.

### AI Coding Rarely Writes SQL Injections or XSS Bugs

💡 **Extended Notes**

Modern stacks + ORM defaults + React's JSX escaping mean **classic OWASP Top 10 "easy mode"** bugs are less common in AI output than in 2012 PHP tutorials. Models have seen endless "use parameterized queries" and "don't use `innerHTML`" advice. Prisma/`Parameterized` SQLAlchemy queries look "correct" out of the box.

Do not celebrate yet. Agents still:

| Still common | Why agents ship it |
|--------------|--------------------|
| Shell / NoSQL injection | String-building `subprocess`, Mongo `$where`, Redis keys |
| Stored XSS in rich text | `dangerouslySetInnerHTML`, markdown→HTML, PDF HTML |
| Secret leakage | Logging headers, committing `.env`, verbose 500s |
| Auth bypass adjacent | "Skip auth in dev" flags left on in prod configs |
| Path traversal | `open(user_path)` for file download features |

**Takeaway:** Absence of textbook SQLi/XSS is not a secure app — residual risk moved up the stack into **authorization, logic, and server-side request** classes that models under-specify.

```python
# Looks "safe" (ORM) but still wrong if org scoping is missing
invoice = await db.invoice.find_unique(where={"id": invoice_id})
return invoice  # IDOR: any authenticated user who guesses UUIDs wins
```

### AI Agents Struggle with Role-Based Access Control

💡 **Extended Notes**

RBAC/ABAC requires consistent enforcement on **every** path: UI, API, background jobs, admin scripts, **and AI tools**. Agents are especially weak here because:

1. Tutorials show `if (user.role === 'admin')` in one React component and call that "RBAC."
2. Multi-tenancy (`org_id`) is a product concern rarely in the prompt.
3. Tool-calling agents get a `get_user` / `delete_user` toolkit with no authorization wrapper — the model becomes the policy engine (it is a terrible one under injection).

Common AI-generated failures:

- Check roles in the React router but not in the API
- Trust `user_id` from the client body instead of the session
- Add `isAdmin` on the JWT client-side without server verification
- Generate "admin" endpoints gated only by obscure URLs
- MCP/agent tools that call the same ORM without `authorize()`

Multi-tenant IDOR: `GET /invoices/124` returns another customer's invoice because the query filtered by `id`, not `(id AND org_id = principal.org_id)`.

**Hardening pattern:**

```python
# Single policy module — force agents to call this
def authorize(principal, action: str, resource) -> None:
    if not policy.allows(principal, action, resource):
        raise Forbidden(action)

@app.get("/invoices/{invoice_id}")
def get_invoice(invoice_id: str, principal=Depends(current_user)):
    inv = repo.get(invoice_id)  # may be None
    authorize(principal, "invoice:read", inv)
    return inv
```

Put this pattern in a repo **skill** / cursor rule *and* enforce with integration tests that swap principals across tenants. Rules without tests are suggestions; agents ignore suggestions when green paths are easier.

### AI Coding Agents Struggle with Business Logic and SSRF

💡 **Extended Notes**

**Business logic** flaws live in underspecified prompts ("build a refund flow"). Models invent happy paths and miss:

- Double refund / replay of webhook events
- Negative quantities, coupon stacking, currency mismatch
- TOCTOU races on inventory
- "Cancel subscription but keep paid features until period end" edge cases

Ask the agent for an **abuse case list** before implementation — then turn each into a test. Still verify yourself; models miss creative fraud.

**SSRF** shows up whenever agents add "fetch this URL" features: link previews, import-from-URL, PDF-from-HTML, webhook testers, "screenshot this page."

Attacker-supplied URL examples:

```text
http://169.254.169.254/latest/meta-data/iam/security-credentials/
http://127.0.0.1:8501/  (internal admin)
http://metadata.google.internal/
file:///etc/passwd
```

Mitigations (defense in depth):

1. Allowlist schemes (`https` only) and hosts
2. Block link-local, loopback, RFC1918, cloud metadata IPs **after DNS resolve** (DNS rebinding otherwise bypasses)
3. Fetch only through a locked-down egress proxy
4. Disable or strictly limit redirects
5. Never return raw fetch bodies to other tenants without sanitization

### AI Coding Agents Struggle with Rate Limiting and CSRF

💡 **Extended Notes**

**Rate limiting** is invisible in single-user demos, so agents omit it. Consequences:

- OTP / password spray
- LLM route cost bombs (`POST /chat` unbound)
- SMS / email toll fraud
- Credential stuffing on login

Apply limits at **edge + app**: per IP, per account, per route class (auth stricter than read-only).

**CSRF** — cookie-session SPAs need CSRF tokens or careful `SameSite` + method design. Agents often:

- Use cookie sessions with `fetch` and no CSRF token "because Content-Type is JSON" (not a complete defense in all browsers/plugins)
- Or pivot to localStorage JWTs (then XSS becomes game-over)
- Skip `SameSite=Lax/Strict` discussions entirely

Also watch in vibe-coded apps:

| Gap | Exploit |
|-----|---------|
| No webhook signature verify | Attacker forges Stripe/GitHub events |
| Magic links without one-time/nonce | Replay login |
| Unbounded uploads | Disk / AV / XSS via SVG |
| Missing CORS discipline | Confused browser calls |

### Prompt Engineering Won't Fix Insecure AI Code

💡 **Extended Notes**

Lecture title is the thesis: **you cannot prompt your way to a secure codebase.**

Why prompts fail as controls:

1. **Non-determinism** — same prompt, different insecure variants across runs (Roy's variance point applies to security too).
2. **Objective mismatch** — models maximize "user said it works"; security maximizes "attacker fails."
3. **No enforcement** — a system prompt is not an authorization middleware; injection can override or distract it.
4. **Scale** — one forgotten endpoint undoes a paragraph of "always check auth."

What actually works:

| Control | Role of AI |
|---------|------------|
| Golden-path templates / internal SDKs | Agent must use approved modules |
| CI: Semgrep, dependency audit, IaC scanners | Agent output blocked on fail |
| Authz + IDOR integration tests | Agent must make tests green |
| Threat model on trust boundaries | Human-owned; agent assists |
| Skills/rules describing policy API | Guidance only — tests enforce |

```text
Prompt engineering  →  discovery & draft patches
Tests + gates       →  enforcement
Humans on boundaries→  residual risk acceptance
```

Use prompts to ask *"how would you attack this diff?"* — then convert answers into tests. Do not confuse that conversation with a security program.

### Test Yourself — Section 27

1. Why might an AI-generated NestJS+Prisma app avoid SQLi yet still be unsafe?
2. Give an IDOR example an agent might ship in a multi-tenant SaaS.
3. How does a "preview this URL" feature become SSRF?
4. Why is "add security best practices to the system prompt" an insufficient control plane?
5. Name two CI checks that catch classes of vibe-coding failures prompts miss.

---

## 28. Bonus

💡 **Extended Notes** — no transcript in the dump. Treat this as a capstone briefing.

You now have a vertical slice of modern agent engineering:

| Theme | Course idea to keep |
|-------|---------------------|
| **Protocols** | MCP = integrate tools once; hosts plug in |
| **Harnesses** | Plans, subagents, FS, monstrous system prompts |
| **Skills** | Progressive disclosure via middleware + `SKILL.md` |
| **Glossary** | ChatModels, Messages, Documents, token strategies, checkpointers |
| **Production** | Observability, AI gateway, FAIR trust, lean feedback files |
| **Hard evals** | CTF-style loops beat vanity demos; manage variance deliberately |
| **Security** | LLM AppSec + vibe-coding failure modes; tests enforce, prompts suggest |

### One-week practice plan

| Day | Exercise |
|-----|----------|
| 1 | Run mcpdoc (or LangChain Docs MCP) in Cursor; answer a LangGraph question from live docs |
| 2 | Write math (stdio) + weather (SSE) servers; consume with `MultiServerMCPClient` |
| 3 | Trace the run in LangSmith; annotate which server each tool hit |
| 4 | Install Deep Agents; add a tiny internal skill (`SKILL.md` + one rule file); watch disclosure in traces |
| 5 | Add a per-user feedback markdown + before-model middleware that injects it |
| 6 | Write two IDOR tests and one prompt-injection fixture against an agent tool |
| 7 | Force an IDE agent to write LangChain code *with* Docs MCP on, then *with it off* — compare deprecations |

### Architectural rhymes worth remembering

- Assaf's feedback markdown ↔ Deep Agents skills middleware: **externalized memory injected before the model call**.
- MCP server runtime ↔ Deep Agent FS backend: **keep heavy state off the prompt, fetch by need**.
- Roy's harness engineering ↔ Eden's deep-agent taxonomy: **context policy is the product**.

### Test Yourself — Section 28

1. Write a one-week practice plan that touches MCP, one deep-agent skill, and one security test.
2. Which course idea most directly reduces deprecated-code generation in IDEs?
3. How do Assaf's feedback markdown and Deep Agents' skills middleware rhyme architecturally?
4. Pick one shallow agent you have shipped — what would you add first to make it "deeper": todos, subagents, or FS? Why?

---

## Appendix — Quick ASCII Cheatsheet

### MCP (recall)

```
Host[Client₁] ──stdio/SSE──▶ Server₁ (execute tools)
Host[Client₂] ──stdio/SSE──▶ Server₂
LLM ◀── schemas + observations via Host only
```

### Deep agent hierarchy (recall)

```
Main (plan + delegate + FS)
 ├─ Subagent (isolated ReAct) → summary
 ├─ Subagent → summary
 └─ To-do.md (living plan)
```

### Skills progressive disclosure (recall)

```
Metadata in system prompt  →  read SKILL.md  →  read rules/* as needed
        (always)                  (if relevant)     (model chooses)
```
