# Sections 13–16: LangGraph, Reflection, Reflexion & Agentic RAG

Companion notebook for Eden Marco's *LangChain & LangGraph* Udemy course. Source: course transcripts + real code from [`emarco177/langgraph-course`](https://github.com/emarco177/langgraph-course).

```bash
git clone https://github.com/emarco177/langgraph-course.git
cd langgraph-course
git checkout project/ReAct-Agent-Function-Calling   # §13 hands-on
git checkout project/reflection-agent               # §14
git checkout project/reflexion-agent                # §15
git checkout project/agentic-rag                    # §16
# Tip: git log --oneline  # each commit ≈ a lesson on agentic-rag
```

**How to use this notebook:** each `##` / `###` maps to the course outline. Lecture content is covered in full; 💡 **Extended Notes** expand with production judgment. Code blocks are taken from the repo branches. **Test Yourself** closes each major section.

---

## 13. Introduction To LangGraph

### What is LangGraph?

LangChain is the open-source stack for composing LLM apps—prompt templates, models, tools, RAG, agents—especially via LCEL. It has gotten more secure, flexible, and usable over time. You *can* build agents in LangChain alone; Eden covers that in the LangChain portion of the course.

LangGraph exists because complex agentic systems hit a wall that LCEL chains do not solve cleanly: **cycles**.

LangChain's own launch diagram for LangGraph frames a spectrum of control:

```
Deterministic code ──► LLM call ──► Chains ──► Router chains ──► [gap] ──► Autonomous agents
     (full control)                                    (acyclic)              (full freedom)
                                                              ▲
                                                         LangGraph
                                                      (scoped freedom + cycles)
```

| Layer | Who controls the flow? | Who controls the output? | Cycles? |
|-------|------------------------|---------------------------|---------|
| Deterministic code | Developer | Developer | N/A |
| Single LLM call | Developer (before/after) | LLM | No |
| Chains | Developer | LLM(s) at multiple steps | No (DAG) |
| Router chains / agents | LLM chooses branch | LLM | Still acyclic in LCEL |
| Autonomous agents (AutoGPT-style) | LLM invents tasks | LLM | Unbounded |
| **LangGraph** | Developer designs graph; LLM may choose edges | LLM inside nodes | **Yes** |

With LCEL you get acyclic graphs: a flow you write ahead of time. You cannot elegantly go back to an earlier node and re-run. LangChain *does* have ad-hoc loop implementations (classic ReAct `AgentExecutor` is literally a `while` loop in library code), but they are opaque and hard to customize.

LangGraph's pitch from the docs: **build language agents as graphs**—including graphs *with cycles*. That extra dimension of freedom is what you need for reflection loops, tool-retry loops, corrective RAG, and anything that says "try again until good enough."

💡 **Extended Notes:** Think of LangGraph as a **state machine runtime** specialized for LLM apps—not "LangChain 2.0 that replaces chains." You still use LangChain for models, tools, prompts, retrievers. LangGraph owns *orchestration*: nodes, edges, shared state, checkpoints, human-in-the-loop. Mixing them is the intended design.

---

### Why LangGraph? LangGraph VS LangChain

This lecture is intentionally theoretical: why LangGraph was created, and where it sits on the autonomy spectrum.

**Left pole — deterministic systems.** You write every step. Resilient and reliable; zero flexibility. No LLM.

**Right pole — autonomous agents.** BabyAGI, AutoGPT, GPT-Engineer-style systems that invent tasks, write code, re-plan. Extremely flexible; in practice not production-ready. LLMs are statistical next-token predictors—give them unbounded freedom and they wander.

**In between:**

1. **LLM inside your code** — you still own control flow; the model summarizes an alert, extracts entities, etc. One non-deterministic island in a deterministic sea. Example Eden uses: cybersecurity product explaining an alert.

2. **Chaining** — take one LLM output as input to the next. Classic RAG: question → embed → retrieve similar docs → augment prompt → generate. Multiple LLM-determined outputs; still a left-to-right DAG. This is where LangChain shines day-to-day.

3. **LLM router** — model chooses Branch 1 vs Branch 2 (database vs web). First time the LLM decides *which step to take*. Still **no cycles** in the LangChain router story—the diagram's dotted line marks the boundary. Below it: agentic applications.

Everything above the dotted line is well implemented in LangChain; Eden is a fan of building advanced systems with only those blocks. The **gap** between routers and fully autonomous agents is where LangGraph sits.

**Soft definitions:** ask three people "what is an agent?" get three answers; ask again tomorrow, get three more. Eden likes Andrew Ng / Harrison Chase writing on the topic; his own reduction: **an agent is a control flow where an LLM decides where to go**. Chains are one-directional; agents have cycles. Today's agents usually decide via **function calling** (tool schemas + arguments) rather than free-form ReAct text alone.

**Working definition of an agent (Eden's reduction):** a control flow where an **LLM decides where to go**. A chain is one-directional left-to-right; an agent has **cycles**. Modern agents usually use **function/tool calling** so the model picks tools with typed arguments rather than free-form ReAct text.

**Classic ReAct loop (paper → LangChain AgentExecutor):**

```
START → LLM: tool? → [yes] execute tool → feed result back to LLM → …
                  → [no]  return final answer
```

Flexible (any tool permutation) but **unreliable**. Anyone who has shipped agents has seen the death spiral: same tool invoked forever; wrong args; hallucinated tool names; weak models choosing poorly. Flexible like AutoGPT, unreliable like AutoGPT.

Production needs flexibility *and* reliability. LangGraph's idea: **scope freedom one dimension down**. Represent the agent as a **graph / state machine** you design. The LLM still decides branches and may run inside nodes, but *you* own the topology. Reliability comes from architecture—not from hoping the next token is lucky.

Why not Airflow / NetworkX? You could—but LangGraph is **opinionated for agentic apps**: controllability, parallel nodes, LLM conditional branching, persistence for HITL, time-travel/replay, LangSmith tracing. Nodes can run any Python (LangChain optional). Modeling as graphs also matches how papers illustrate agents—readable, maintainable, testable, monitorable.

Opinionated LangGraph capabilities vs rolling your own with Airflow / NetworkX:

- Controllability and conditional branching with LLMs
- Parallel node execution
- Built-in **persistence** (checkpoint state for HITL, resume, time-travel)
- First-class LangSmith tracing
- Nodes can run **any** Python—not only LangChain

Shared **state** across nodes/edges holds intermediate results and informs routing.

💡 **Extended Notes — AgentExecutor vs LangGraph (senior take):**

| Concern | LangChain `AgentExecutor` | LangGraph |
|---------|---------------------------|-----------|
| Control flow | Opaque `while` loop | Explicit graph you draw |
| Custom state | Awkward kwargs / config | Typed state schema + reducers |
| Observability | Limited insight into loop | Per-node checkpoints + LangSmith |
| Nesting agents | Painful | Graph-as-node composition |
| Cycles / reflection | Ad hoc | First-class |
| Production HITL | DIY | Persistence + interrupt patterns |

Rule of thumb: use `create_agent` / prebuilt ReAct when the loop is enough; drop to a custom `StateGraph` when you need reflection, multi-stage RAG, or non-ReAct topologies.

---

### What are Graphs?

Two primitives you will use constantly in this course:

**Graph (data structure):** a mathematical object for relationships. Nodes (also called vertices) plus edges that connect them. Graphs show up everywhere—social networks, transportation maps and routes between cities, and in Eden's cybersecurity career, graph databases modeling AWS/GCP/Azure assets to ask questions like "is this web server internet-facing and can it reach a database?" (cloud security posture). CS has deep algorithm research on graphs; you benefit from that when you model agents this way.

Formal definition (last math of the course, Eden promises): graph \(G = (V, E)\) with \(V\) a set of vertices and \(E\) a set of edges as pairs \((x, y)\) with \(x, y \in V\).

**State machine:** a model of computation made of states and transitions. By defining states and transition rules, you manage complex conditions and sequences. State machines *visualize* as graphs: states = nodes, transitions = edges. That visualization is why agent papers look like flowcharts—and why LangGraph's API feels natural.

LangGraph (built on LangChain) lets you describe agent flows with those nodes and edges and still keep systems readable, testable, and monitorable.

---

### LangGraph & Flow Engineering

**Flow engineering** is a new, still-informal idea in the GenAI community: a systematic approach to software that incorporates AI-driven decision-making by defining a clear **flow**—a sequence of operations that is not merely linear. Flows may include decision nodes, multiple candidate outputs, and iterative assess-and-refine cycles. The goal is to guide the AI through well-defined steps so output quality, reliability, and functionality improve—mimicking human development discipline (plan, implement, review) rather than one-shot prompting.

**Why this exists:** AutoGPT / BabyAGI-style agents receive a goal, invent tasks and subtasks, and execute indefinitely. In practice they fail: the model invents imaginary plans and drifts. Flow engineering flips the responsibility: **you** define tasks and scope; the LLM stays inside that context. It may still decide "is this good enough to release?" or "which legal next step?"—but it does not invent the program.

In state-machine terms: developers write the states (what can happen). The LLM may choose transition \(i\) vs \(i+2\) based on input. The topology is an engineering decision; statistics do not own it.

LangGraph is the middle ground between fully autonomous agents and fully deterministic LCEL chains. LLMs participate in two ways: (1) **inside** a step (e.g. generate a tweet), and (2) **choosing** the next step (conditional edges). Section 14's tweet generate→reflect→revise loop is the canonical teaching example.

Eden's speculative time split for future AI software work: ~**60% flow engineering / architecture**, ~**35% fine-tuning**, ~**5% prompt engineering**. Treat ratios as a provocation; the ranking (architecture over prompt fiddling) is the lesson. This lecture is abstract on purpose—by course end the idea should feel concrete.

💡 **Extended Notes:** Flow engineering is workflow design for non-deterministic workers. Apply backend instincts—explicit retries, budgets, idempotent steps, observable transitions—when a step is an LLM. Your graph *is* the product architecture document that happens to execute.

---

### LangGraph Core Components

You implement a well-scoped control flow (state machine / graph—pick your vocabulary). Inside that scope, LLMs either (a) choose the next hop or (b) execute work (LLM/agent calls) inside a node.

**Three building blocks:**

1. **Nodes** — literally Python functions. Deterministic code, LLM calls, tool I/O, or nested agents. Full flexibility of what runs inside.
2. **Edges** — connect nodes; define sequential execution.
3. **Conditional edges** — decide node A vs B dynamically (the power move).

Built-in **START** and **END** are no-op sentinels (entry / termination).

**State / agent state:** a dictionary (or typed schema) tracking whatever the run needs—message history, intermediates, feature flags. Local to the graph runtime (every node and edge can read it). Can also be **persisted** so you stop and resume with the same state.

Node contract (critical):

- Always receive the **current graph state** as input.
- Always return a **dict of updates** (keys to merge into state)—not "the whole new world" unless you design it that way.
- Therefore every node execution mutates the evolving state; edges/conditionals route based on that state.

Also previewed for later:

- **Cyclic graphs** — loops (awkward in pure LCEL; natural here)
- **Human-in-the-loop** — human feedback chooses the next node
- **Persistence helpers** — robustness, fault tolerance, and better UX (resume mid-flow)

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
           ┌───────►│  Node (fn)  │───────┐
           │        │ updates     │       │
           │        │   state     │       │
           │        └──────┬──────┘       │
           │               │              │
           │        conditional edge      │
           │         /           \        │
           │        ▼             ▼       │
           │   ┌────────┐   ┌────────┐    │
           │   │ Node A │   │ Node B │    │
           │   └───┬────┘   └───┬────┘    │
           │       │            │         │
           │       └─────►END◄──┘         │
           │                              │
           └──────── cycle edge ──────────┘
```

---

### [Hands On] Implementing ReAct AgentExecutor with LangGraph

**Build goal:** a ReAct-style agent executor as an explicit LangGraph—emphasizing how natural the ReAct loop looks as a graph, with custom state and tools (Tavily search + a custom `triple` tool). Query pattern: weather in a city, then multiply a number (hello-world of tool agents).

**Update (refilmed):** modern LangGraph + `ToolNode` + function calling. Understanding classic ReAct prompts still matters historically; if you already implemented a manual ReAct executor, this section clicks faster.

Branch: `project/ReAct-Agent-Function-Calling` (course also references related ReAct branches).

```
                    START
                      │
                      ▼
              ┌───────────────┐
         ┌───►│ agent_reason  │── tool_calls? ──yes──►┌──────┐
         │    │  (LLM+tools)  │                       │ act  │
         │    └───────┬───────┘                       │ToolNode│
         │            │ no                            └───┬──┘
         │            ▼                                   │
         │           END                                  │
         └──────────────────── reason again ◄─────────────┘
```

---

### Quick Note: poetry vs uv

*[No transcript for this lecture.]*

💡 **Extended Notes:** Poetry was the course default for lockfiles/`pyproject.toml`. **uv** is now a common, much faster alternative (`uv init`, `uv add`, `uv run`). Either works; match whatever the branch's `pyproject.toml` expects. Do not mix Poetry and uv lockfiles casually in one project.

---

### [Hands On] Get Started: Setting Up Your ReAct Agent Project Environment

Boilerplate:

1. Empty directory → `poetry init`
2. Standard Python `.gitignore` (never commit `.env`)
3. Dependencies: `langchain`, `langchain-openai`, `langchain-tavily`, `langgraph`, `python-dotenv`, plus `black` / `isort`
4. `.env`: `OPENAI_API_KEY`, LangSmith (`LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, project name e.g. `react-function-calling`), `TAVILY_API_KEY`
5. `main.py` with `load_dotenv()` sanity check
6. Stub files: `react.py` (reasoning / tools), `nodes.py` (graph nodes)

Commit on branch `project/ReAct-Agent-Function-Calling` (or the function-calling ReAct branch named in the video).

---

### [Hands On] Coding the Agent's Brain: Implementing the ReAct Runnable

`react.py` holds the reasoning engine: tools + an LLM with **bound tools** (function calling), not a hand-rolled ReAct prompt.

From the repo:

```python
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
```

**How it works:** `@tool` turns `triple` into a LangChain tool with schema/description for the model. `TavilySearch` ships with its own description. `bind_tools` attaches both schemas to every chat request; the vendor returns structured tool calls—you no longer parse `Action:` / `Action Input:` from free text. Function calling evolved from ReAct-style prompting; vendors now own parsing reliability.

---

### [Hands On] Building Blocks: Defining Your Agent's Nodes in LangGraph

`nodes.py` — two executables:

1. **`run_agent_reasoning`** — invoke the tool-bound LLM with a system message + `state["messages"]`; return `{"messages": [response]}` (append AIMessage, possibly with `tool_calls`).
2. **`tool_node`** — LangGraph prebuilt `ToolNode(tools)` executes whatever tool calls are on the last AI message (including parallel calls).

```python
from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSYEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    response = llm.invoke(
        [{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]]
    )
    return {"messages": [response]}

tool_node = ToolNode(tools)
```

**How it works:** `MessagesState` is a typed state with a `messages` list (and the `add_messages` reducer under the hood). Before `ToolNode`, tool execution was tedious boilerplate; the prebuilt node is the modern default.

---

### [Hands On] Bringing Your ReAct Agent to Life: Connecting Nodes into a Graph

Wire `StateGraph(MessagesState)`:

- Entry → `agent_reason`
- Conditional edge from `agent_reason`: if last message has `tool_calls` → `act`, else → `END`
- Edge `act` → `agent_reason` (cycle)
- Compile; optionally `draw_mermaid_png` / ASCII

```python
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)
flow.add_conditional_edges(
    AGENT_REASON, should_continue, {END: END, ACT: ACT}
)
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
```

**How it works:** `should_continue` is a **routing function**, not a node—it returns a destination name. The path map `{END: END, ACT: ACT}` makes visualization explicit; without it the graph may still run but Mermaid drawings can omit conditional destinations (you will hit this again in Reflection).

---

### [Hands On] Running Our LangGraph React Agent with Function Calling

Invoke:

```python
res = app.invoke({
    "messages": [
        HumanMessage(
            content="What is the temperature in Tokyo? List it and then triple it"
        )
    ]
})
print(res["messages"][LAST].content)
```

**Expected behavior:** agent reasons it needs live weather → `should_continue` sees `tool_calls` → `act` runs Tavily. If `max_results=1` returns a useless snippet (common), the model may search **again** with a refined query—smart, but spendy. Once it sees e.g. **15°C**, it tool-calls `triple(15)` → 45 → final natural-language answer.

**LangSmith walkthrough:** project `react-function-calling` → open trace → `agent_reason` (system + tools bound) → conditional `should_continue` → `act` (search) → back to reason → maybe second search → `triple` → reason with no tool calls → END. Attach public traces in course resources when available.

**Ops tip:** bump Tavily `max_results` (e.g. 5) so one search is likelier to contain the answer and you avoid retry loops.

Goal of the section: ReAct is *easy* to express as a graph, and function calling makes that agent substantially more stable than prompt-parsed ReAct.

---

### [IMPORTANT] Building Modern LLM Agents: From History to LangGraph v1.0

Eden's recap assumption: you should be able to "sing" the ReAct loop if woken at night—query → LLM picks tool → execute → repeat until final answer.

**Evolution diagram (course narrative):**

1. **ReAct paper + ReAct prompt** — LLM as reasoning engine; LangChain added fancy parsers to extract tools. Impressive demos; weak for production. Models were weaker; one wrong token broke brittle output parsing. Non-determinism made control illusory.

2. **Vendor function calling** — normalized "LLM as reasoner." No special ReAct prompt; vendors return which function to call in a dedicated response slot. Parsing reliability moves to the vendor.

3. **New problem: every vendor differed** — "function calling" vs "tool calling," different JSON shapes. **LangChain's tool-calling interface** unified them so one app API works across OpenAI, Anthropic, Gemini, etc.

4. **LangGraph architectural shift** — replace function-based agent loops (`AgentExecutor`'s abstracted `while`) with **graphs**: shared state dict, nodes as Python functions `(state) → partial update`, edges as control flow. Motivation: almost every agent paper already drew a graph. Explicit structure → printable diagrams, custom state without awkward kwargs, automatic checkpoints before nodes (monitor, rewind, time-travel), and **graph-as-node** composition (nest agents).

5. You built a close cousin of the old LangGraph prebuilt ReAct agent in this section.

6. **LangChain / LangGraph v1.0** — cleaner API via **`create_agent`**, which returns a **compiled LangGraph graph**. Older `create_react_agent` / prebuilt sprawl deprecated in favor of this. Pass model + tools → production-shaped ReAct with observability; still customizable when you outgrow the helper.

Pedagogical point: knowing history means `create_agent` is not magic—you already implemented the bones. That foundation also underlies deeper agents covered later in the course.

---

### Test Yourself — Section 13

1. Why can't pure LCEL express a reflection or ReAct retry loop cleanly?
2. What is the node I/O contract with graph state?
3. How does `ToolNode` decide which tools to run?
4. What does the path map on `add_conditional_edges` buy you beyond runtime routing?
5. Contrast `AgentExecutor` vs LangGraph on custom state and observability.
6. Where does LangGraph sit on the autonomy spectrum relative to routers and AutoGPT-like agents?

<details>
<summary>Answers</summary>

1. LCEL graphs are acyclic; cycles need an external while-loop (AgentExecutor) or LangGraph.
2. Nodes receive state and return a partial update dict (e.g. `{"messages": [...]}`).
3. Inspects the last AI message's `tool_calls` and executes matching bound tools (can parallelize).
4. Documents legal destinations for visualization and explicit mapping when the router returns labels ≠ node names.
5. AgentExecutor: opaque loop, weak custom state; LangGraph: explicit topology, typed state, checkpoints/LangSmith.
6. Between LLM routers (acyclic) and fully autonomous agents—scoped freedom with developer-owned flow.

</details>

---

## 14. Reflection Agent

### What are we building? A Reflection Agent

Reflection agents improve quality by prompting an LLM to **critique past outputs**, then revise—iterate until acceptable.

**Project:** revise a tweet (Eden's LangChain tool-calling announcement). Flow: generate → critique → revise → critique → … until a stopping rule. Under 100 lines because LangGraph owns the loop.

```
                    START
                      │
                      ▼
              ┌──────────────┐
         ┌───►│   generate   │── len(messages)>6? ──yes──► END
         │    │ (tweet LLM)  │
         │    └──────┬───────┘
         │           │ no
         │           ▼
         │    ┌──────────────┐
         │    │   reflect    │  (critique; stored as HumanMessage)
         │    └──────┬───────┘
         │           │
         └───────────┘
```

Branch: `project/reflection-agent`.

💡 **Extended Notes:** Classic "generator–critic" pattern. Stopping on message count is a teaching heuristic; production often uses LLM-as-judge, score thresholds, or max wall-clock/cost budgets (foreshadowing Agentic RAG).

---

### Project Setup

Lightweight compared to Reflexion/Agentic RAG: Poetry init, `.env` with OpenAI + LangSmith tracing, `chains.py` for prompts/LCEL, `main.py` for graph wiring. Prefer matching `project/reflection-agent` commits lesson-by-lesson (`git log --oneline` on that branch). No Tavily required for pure tweet reflection.

---

### Creating the Reflector Chain and the Tweet Reviosr Chain

Two LCEL chains in `chains.py`:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI()
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
```

**How it works:** `MessagesPlaceholder("messages")` injects full history each turn so the generator sees prior drafts + critiques, and the reflector sees the latest tweet in context. Generation revises when critique is present; reflection only critiques.

---

### Defining our LangGraph Graph

State schema with **`add_messages` reducer** so updates append instead of replace:

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from chains import generate_chain, reflect_chain

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}  # critique as "human"

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)
graph = builder.compile()
```

**How it works:**

- First `generate`: only the user tweet request → draft.
- `reflect`: critique; **cast AI critique to `HumanMessage`** so the next generation treats feedback like user instructions (models are trained on human feedback).
- Conditional after generate: stop after enough messages (~2 full reflect cycles with threshold `> 6`), else reflect.
- Deterministic edge reflect → generate.

**Visualization gotcha:** after `compile()`, `get_graph().draw_mermaid()` / Excalidraw may **omit** the conditional edge from generate→reflect and generate→END. Runtime still works. Cause: LangGraph does not infer destinations from `should_continue` alone. Fix—pass a path map as the third argument to `add_conditional_edges`, e.g. map `END→END` and `REFLECT→REFLECT` (same pattern as ReAct). Re-draw and the dashed conditional appears. `print_ascii()` is also available for terminal diagrams.

💡 **Extended Notes — reducers:** `add_messages` is one reducer; any function matching the reducer interface can define merge semantics (replace, append, union sets, …). That flexibility is a major LangGraph advantage over stuffing ad-hoc kwargs into AgentExecutor.

---

### LangSmith Tracing

Invoke the compiled graph with input like "Make this tweet better:" plus Eden's original LangChain tool-calling tweet (single interface for OpenAI / Gemini / Claude function calling—big deal when only OpenAI functions existed).

Expect ~**20 seconds**: many sequential LLM calls. In LangSmith (project name from `.env`), open the running trace. The **final** LLM prompt is the best teaching artifact—it already contains the full history: system "twitter techie influencer…", user request, draft, human-tagged critique, revised draft, next critique, … until the stop rule.

On the left, LangSmith lists graph objects: `should_continue`, reflection nodes, generate nodes—first-class observability for LangGraph, not just raw chat completions. You implemented a slim critiquing algorithm; you *could* have done it with raw LangChain loops, but the graph form stays tiny and inspectable.

---

### Test Yourself — Section 14

1. Why cast reflection output to `HumanMessage`?
2. What does `Annotated[..., add_messages]` change vs plain assignment?
3. Is `should_continue` a node? What must it return?
4. Reflection vs a single "write a better tweet" prompt—what's the architectural win?

<details>
<summary>Answers</summary>

1. So the generator treats critique as user feedback, aligning with chat training.
2. Updates append to the message list instead of replacing it.
3. No—it's a conditional-edge router; return a node name or `END`.
4. Explicit iterative critique with inspectable state/traces; you can swap the stop rule for an LLM judge later.

</details>

---

## 15. Reflexion Agent

### What are we building? A Reflexion Agent

Extends reflection with **tools** (Tavily web search) and **structured self-critique** so revisions are grounded in external evidence + citations. Harder problem: not just producing critique, but making the model *use* critique across iterations.

Inspired by the **Reflexion** paper (Northeastern / MIT / Princeton) and LangChain's blog implementation (refactored by Eden for teachability). Branch name in repo: `project/reflexion-agent` (videos sometimes say "reflection agent"—same project family, different from Section 14).

**Build goal:** ~250-word researched article with citations on a topic (demo: AI-powered / autonomous SOC startups and funding).

```
 START
   │
   ▼
┌──────────┐   answer + Reflection{missing,superfluous}
│  draft   │   + search_queries  (tool_choice=AnswerQuestion)
└────┬─────┘
     ▼
┌──────────────┐
│ execute_tools│  Tavily batch (concurrent)
└────┬─────────┘
     ▼
┌──────────┐   revised answer + new critique + queries + references
│  revise  │   (tool_choice=ReviseAnswer)
└────┬─────┘
     │
     ├─ iterations left? ──yes──► execute_tools ──► revise ──► …
     │
     └─ no ──► END
```

Stack: strong model (course: GPT-4 Turbo; repo tip often `o4-mini`), function calling for schemas, Tavily, LangSmith.

💡 **Extended Notes — Reflection vs Reflexion:**

| | Reflection (§14) | Reflexion (§15) |
|--|------------------|-----------------|
| External tools | No | Yes (search) |
| Output shape | Free text | Pydantic via tool calling |
| Critique structure | Prose | `missing` / `superfluous` fields |
| Grounding | Parametric LLM knowledge | Web results + citations |
| Stop condition | Message count | Iteration / tool-message count |

---

### Project Setup

Poetry project; deps: dotenv, black, isort, langchain, langchain-openai, langgraph, langchain-tavily. `.env`: OpenAI, Tavily, LangSmith (`LANGCHAIN_PROJECT` e.g. reflexion). `main.py` hello + `load_dotenv`.

---

### Section Resources

*[No transcript — links / paper / blog in Udemy resources.]*

---

### Actor Agent V2

`chains.py` + `schemas.py`: first responder produces structured `AnswerQuestion`.

**Schemas:**

```python
from typing import List
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """Answer the question."""
    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""
    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
```

**Actor prompt** (shared by draft + revise via `first_instruction` partial):

```python
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
) | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
```

**How it works:**

- Shared `actor_prompt_template` keeps draft and revise aligned; only `{first_instruction}` changes.
- `.partial(time=lambda: ...)` injects "current time" at invoke so research answers feel time-aware.
- Field descriptions on Pydantic models are **prompting through schema**—the model fills `missing` / `superfluous` because those fields exist and are described.
- `tool_choice="AnswerQuestion"` **forces** the tool every turn (structured output via function calling, not optional tool use).
- `PydanticToolsParser(tools=[AnswerQuestion])` turns the tool payload into a typed object for debugging; the live graph often keeps the raw AIMessage with `tool_calls` in `MessagesState`.

**Demo query:** "Write about AI-Powered SOC / autonomous SOC problem domain, list startups that raised capital." First draft may list Darktrace / Vectra / etc. from parametric memory; reflection asks for funding figures; `search_queries` might be `AI-powered SOC startup funding`, `Darktrace funding history`, …. That gap between "sounds right" and "cited" is exactly why Reflexion adds Tavily.

**Non-determinism:** Eden hit a validation error when `search_queries` was omitted once—rerun worked. Harden with stronger "MUST provide search_queries" wording or split query generation into its own call for production.

State choice: `MessagesState` (list of messages)—same idea you hand-rolled as `MessageGraph` + `add_messages` in Section 14.

---

### Revisor Agent

Revision instructions plugged into the same actor template:

```python
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
```

**How it works:** `ReviseAnswer` extends `AnswerQuestion` with `references`. Same reflect + search-query discipline, plus citation discipline after Tavily results land in message history.

---

### ToolNode - Executing Tools

Trick: one search function, **two tool names** matching schema class names so draft-phase vs revise-phase searches are distinguishable in traces:

```python
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)

def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
```

**How it works:** When the model emits a tool call named `AnswerQuestion` or `ReviseAnswer`, `ToolNode` runs `run_queries` with the model's args (including `search_queries`). `batch` runs searches concurrently. `**kwargs` absorbs extra fields so schema leftovers don't crash the tool.

---

### Building Our LangGraph Graph

```python
from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2

def draft_node(state: MessagesState):
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState):
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return "execute_tools"

builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile()
```

**How it works:** Always draft → search → revise; then either another search/revise cycle or END. The repo counts **`ToolMessage`s** (search results), not AI `tool_calls`—slightly different from an early verbal explanation in the video. Eden also notes off-by-one behavior: with `MAX_ITERATIONS = 2` you may observe **three** revision cycles depending on when state updates vs when the conditional runs. Prefer LLM-as-judge in production (next section).

Extract final prose from the last AIMessage's tool call args (`answer` field).

---

### Tracing Our Graph

Open the LangSmith trace (~**50s**, on the order of **~35K tokens** in Eden's demo) and collapse to node level so it matches the architecture diagram:

1. **draft / responder** — `AnswerQuestion` tool call with `answer`, `reflection.missing`, `reflection.superfluous`, `search_queries` (e.g. funding / market / platform comparisons).
2. **execute_tools** — three Tavily queries with the **same start timestamp** → `ToolNode` ran them concurrently.
3. **revise** — history now includes tool results; emits `ReviseAnswer` (citations + new critique + new queries).
4. **event_loop** — routes to another `execute_tools` if under budget.
5. Second search wave uses **different** queries (ROI case studies, market size, adoption)—evidence the revisor's new `search_queries` matter.
6. Another revise → event_loop → eventually END.

**Iteration-counting gotcha (called out in lecture):** depending on when state updates relative to the conditional, `MAX_ITERATIONS = 2` may yield **three** revision cycles. Eden apologizes on video—the magic number is a teaching heuristic. Next section replaces it with LLM-as-judge (Agentic RAG graders).

---

### Test Yourself — Section 15

1. Why force `tool_choice` to `AnswerQuestion` / `ReviseAnswer`?
2. Why two `StructuredTool`s with different names wrapping the same function?
3. What bug class does counting iterations with magic numbers introduce?
4. How does Reflexion improve on Section 14's tweet reflector?

<details>
<summary>Answers</summary>

1. Guarantees structured fields (answer, reflection, queries, references) every turn.
2. Traceability: distinguish initial research tool calls from revision-phase searches.
3. Off-by-one / state-timing mismatches; Eden saw 3 iterations when intending 2.
4. Adds web grounding, citations, and structured missing/superfluous critique.

</details>

---

## 16. Agentic RAG

### What are Building In this Section- Agentic RAG Architecture

Advanced RAG workflow inspired by LangChain + Mistral cookbook, **refactored for production shape** (packages, tests, incremental commits)—not a notebook dump. Branch: `project/agentic-rag`.

Three research threads composed into one graph:

| Paper | Idea in this project |
|-------|----------------------|
| **Corrective RAG** | Grade retrieved docs; if weak → filter + web search |
| **Self-RAG** | Grade generation for grounding + answer relevance; regenerate or search |
| **Adaptive RAG** | Route question to vectorstore vs websearch at entry |

Lots of **reflection** on documents and answers, plus **routing** to the right datasource.

---

### Improving RAG Quality with the Corrective RAG Flow

**Corrective RAG (CRAG)** (course pronunciation varies—"Chirag" / "corrective RAG") is an advanced retrieval technique from the Corrective RAG research paper. Basic concept:

1. Take the user query; run vector / semantic search; retrieve candidate documents.
2. **Self-reflect / critique** each document: is it relevant to the original query?
3. **Happy path:** all docs relevant → augment the prompt → generate (classic RAG).
4. **Unhappy path:** some docs irrelevant → **filter them out** *and* run an **external web search** for fresher/better context → augment with remaining docs + web results → generate.

```
Query → vector retrieve → grade each doc
         │
         ├─ all relevant ──────────────► augment + generate
         │
         └─ any irrelevant ► filter them
                              + web search
                              ► augment + generate
```

Quality win: you stop treating "top-k cosine neighbors" as gospel. Retrieval errors become a first-class branch, not silent context pollution.

---

### Boilerplate Setup for an Agentic RAG Agent with LangGraph

Create project directory → `poetry init` → add packages:

- `beautifulsoup4` (HTML parsing for web loaders)
- `langchain`, `langgraph`, `langchain-hub`, `langchain-community`
- Tavily / search SDK, **Chroma**, `python-dotenv`, `black`, `isort`
- **`pytest`** — Eden insists tests matter for GenAI apps

Configure PyCharm/VS Code interpreter to the Poetry env. `.env`: `OPENAI_API_KEY`, LangSmith (`LANGCHAIN_TRACING_V2`, project name e.g. `CRAG`), `TAVILY_API_KEY`, `PYTHONPATH` pointing at repo root (so `graph.*` and `ingestion` imports resolve). `main.py` loads dotenv and prints `Hello Advanced RAG`. Course commits live on incremental branches (`1-start-here`, …); the combined tip is `project/agentic-rag`.

💡 **Extended Notes:** Pytest in a GenAI course is deliberate. LLM tests are flaky (non-idempotent answers), depend on third-party availability/rate limits, and cost tokens. Mitigations: cheaper grader models, golden-set evals offline, VCR-style fixtures for deterministic CI. Thin live tests still catch "forgot `load_dotenv`" and schema regressions—better than nothing.

---

### Code Structure

Architecture-mirroring layout:

```
ingestion.py
main.py
graph/
  graph.py          # wire nodes + edges
  state.py          # GraphState
  consts.py         # node name constants
  nodes/            # one file per node
    retrieve.py
    grade_documents.py
    web_search.py
    generate.py
  chains/           # one file per chain (≈ node logic)
    retrieval_grader.py
    generation.py
    hallucination_grader.py
    answer_grader.py
    router.py
    tests/
      test_chains.py
```

Eden's rule: **repo structure should reflect graph architecture**. Not the only valid layout—but it scales when you add nodes.

---

### LangChain Vector Store Ingestion Pipeline (WebLoader, ChromaDB)

Ingest Lilian Weng posts (agents, prompt engineering, adversarial attacks)—focus is retrieval logic, not ingestion Olympics:

```python
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Chroma.from_documents(... persist_directory="./.chroma")  # run once
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()
```

**How it works:** WebBaseLoader → flatten nested lists → tiktoken chunks (250, no overlap) → Chroma + OpenAI embeddings on disk → `as_retriever()` for similarity search. Comment out re-indexing after first persist.

---

### Managing Information Flow in LangGraph: The GraphState

```python
from typing import List, TypedDict

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: bool
    documents: List[str]
```

(Runtime often carries Document objects in `documents` despite the annotation—typed loosely for teaching.)

**How it works:** Every node reads/writes this shared bag: question for grading/search, documents as evolving context, `web_search` flag for CRAG routing, `generation` for Self-RAG graders.

---

### Fetching Context for LLMs: The LangGraph Retrieve Node

```python
def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
```

---

### Building a Relevance Filter for RAG using LangChain's Structured Output

**Chain** — binary relevance via structured output:

```python
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)
retrieval_grader = grade_prompt | structured_llm_grader
```

**Node** — filter docs; set `web_search=True` if any doc fails:

```python
def grade_documents(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        if score.binary_score.lower() == "yes":
            filtered_docs.append(d)
        else:
            web_search = True
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }
```

**How it works:** Heuristic—"any irrelevant chunk ⇒ also search the web." Tests: relevant "agent memory" doc → `yes`; same doc vs "how to make pizza" → `no`. Lecture discusses LLM test pain (non-idempotent, third-party outages, cost) and why thin tests still matter.

---

### Implementing a Web Search Node in LangGraph using Tavily API

```python
web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state["documents"] if "documents" in state else None
    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined = "\n".join([r["content"] for r in tavily_results])
    web_results = Document(page_content=joined)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}
```

**How it works:** Join top contents into one `Document`, append to surviving vector docs (or replace if Adaptive RAG jumped straight to web). Guard for missing `documents` when routing skips retrieve.

---

### Creating the LLM Generation Chain and Node for LangGraph

Hub prompt `rlm/rag-prompt` + `StrOutputParser`:

```python
prompt = hub.pull("rlm/rag-prompt")
generation_chain = prompt | llm | StrOutputParser()

def generate(state: GraphState) -> Dict[str, Any]:
    generation = generation_chain.invoke(
        {"context": state["documents"], "question": state["question"]}
    )
    return {
        "documents": state["documents"],
        "question": state["question"],
        "generation": generation,
    }
```

---

### Building and Running the Complete LangGraph Agent

**Corrective RAG wiring** (intermediate milestone):

```
START → retrieve → grade_documents ─┬─ web_search=true ─► websearch → generate → END
                                    └─ web_search=false ► generate → END
```

```python
def decide_to_generate(state):
    if state["web_search"]:
        return WEBSEARCH
    return GENERATE

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)
# entry retrieve → grade → conditional → (websearch → generate) | generate → END
```

`main.py`: `app.invoke({"question": "agent memory?"})`. LangSmith shows grade → optional Tavily → generate.

---

### Self RAG- Intro

**Self-RAG** (from the Self-RAG paper) means reflecting on the **answer the model already generated**, not only on retrieved docs (that was CRAG).

1. Compare generation to documents: did the model **hallucinate**, or is the answer grounded?
2. If grounded → second reflection: does the answer actually **answer the user's question**?
   - Yes → return to the user.
   - No → likely the vector store lacked coverage → **web search**, then continue.
3. If **not** grounded → **regenerate** (do not return garbage; try again against the same docs).

You will implement grader chains, tests, and conditional branches end-to-end in the next lecture.

```
                 generate
                    │
         hallucination grader
              /           \
         grounded        not grounded
            │                 │
      answer grader      regenerate (→ generate)
       /        \
   useful    not useful
     │            │
    END      websearch → generate
```

---

### Self RAG- Implementation

**Hallucination grader** (`binary_score: bool`) and **answer grader** (same idea vs question). Conditional edge function (not separate nodes—because the *decision* is the routing):

```python
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    score = hallucination_grader.invoke(
        {"documents": state["documents"], "generation": state["generation"]}
    )
    if score.binary_score:
        score = answer_grader.invoke(
            {"question": state["question"], "generation": state["generation"]}
        )
        if score.binary_score:
            return "useful"
        return "not useful"
    return "not supported"

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
```

**How it works:** Path map labels (`useful` / `not useful` / `not supported`) become readable edge names in diagrams while mapping to real nodes. Tests assert grounded generations vs pizza-dough nonsense.

---

### Adaptive RAG

Adaptive RAG ≈ **question router** at the entry: vectorstore vs websearch based on whether the index can answer (prompt lists indexed topics: agents, prompt engineering, adversarial attacks).

```python
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

question_router = route_prompt | structured_llm_router

def route_question(state: GraphState) -> str:
    source = question_router.invoke({"question": state["question"]})
    if source.datasource == WEBSEARCH:
        return WEBSEARCH
    return RETRIEVE

workflow.set_conditional_entry_point(
    route_question,
    {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE},
)
```

```
                    START
                      │
              route_question
               /          \
         retrieve        websearch
            │               │
      grade_documents       │
         /      \           │
   generate   websearch ◄───┘
       │         │
       └────► generate
                │
         self-RAG graders
         (useful / retry / search)
```

Final `graph/graph.py` in the branch composes **Adaptive + Corrective + Self-RAG**. Demo: `"agent memory"` → route to vectorstore → retrieve → grade → (maybe web) → generate → self-RAG judges → END. Demo: `"how to make pizza"` → route straight to websearch → generate → judges.

Full orchestrator from `project/agentic-rag` (trimmed comments only):

```python
# graph/graph.py — Adaptive + Corrective + Self-RAG
workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE},
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,  # hallucinated → regenerate
        "useful": END,              # grounded + answers question
        "not useful": WEBSEARCH,    # grounded but incomplete → search
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
# Note: tip-of-tree may also contain a legacy `add_edge(GENERATE, END)` from the
# Corrective-only milestone; Self-RAG's path map is the intended post-generate policy.

app = workflow.compile()
```

ASCII — full agentic RAG:

```
                         START
                           │
                    route_question
                    /            \
                   ▼              ▼
               retrieve        websearch
                   │              │
            grade_documents       │
              /          \        │
             ▼            ▼       │
         generate      websearch──┘
             │            │
             └─────► generate ◄────┐
                       │           │
            self-RAG conditional   │
         /         |         \     │
      useful  not useful  not supported
        │         │            │
       END    websearch    (loop generate)
```

💡 **Extended Notes:** Naive RAG = retrieve + generate. Agentic RAG = retrieval is one node in a **policy graph** that can refuse bad context, fetch more, and refuse bad answers. Every grader multiplies cost and latency—use small/fast models for binary judges, cap regenerate loops, and log route decisions for offline evaluation. Prefer the Self-RAG path map over any leftover `GENERATE → END` edge when reading tip-of-tree code.

---

### Complete Agentic RAG — Chains Reference

**Hallucination grader** (`graph/chains/hallucination_grader.py`):

```python
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_grader = hallucination_prompt | structured_llm_grader
```

**Answer grader** (`graph/chains/answer_grader.py`): same pattern with `GradeAnswer.binary_score` and prompt "does the answer resolve the question?"

**Router** (`graph/chains/router.py`): `RouteQuery.datasource: Literal["vectorstore", "websearch"]` with system prompt listing indexed topics (agents, prompt engineering, adversarial attacks)—everything else → web search.

**How the three papers stack in one compile:**

| Layer | When it fires | Mechanism |
|-------|---------------|-----------|
| Adaptive | Graph entry | `set_conditional_entry_point(route_question)` |
| Corrective | After retrieve | `grade_documents` + `decide_to_generate` |
| Self-RAG | After generate | `grade_generation_grounded_in_documents_and_question` |

`main.py` stays trivial:

```python
from dotenv import load_dotenv
load_dotenv()
from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "agent memory?"}))
```

---

### Test Yourself — Section 16

1. CRAG vs naive RAG when one of four chunks is irrelevant?
2. Why put Self-RAG checks on a conditional edge instead of a node?
3. What does Adaptive RAG add that CRAG+Self-RAG alone lack?
4. Name three reasons LLM unit tests are awkward—and why write them anyway?
5. Map each paper to a graph feature in this project.

<details>
<summary>Answers</summary>

1. Filter the bad chunk and set web_search to enrich context before generate.
2. The check's purpose is choosing the next step (END / generate / websearch)—routing is the natural abstraction.
3. Entry routing: skip vector retrieval when the index can't help.
4. Non-determinism, third-party failures/rate limits, token cost; still catch regressions and wiring errors.
5. Corrective → doc grader + web fallback; Self → hallucination/answer graders after generate; Adaptive → conditional entry router.

</details>

---

## Cross-Section Comparison (Senior Cheat Sheet)

| Pattern | Loop over | External tools | Structured critique | Typical stop |
|---------|-----------|----------------|---------------------|--------------|
| ReAct graph | Tool use | Yes | Tool schemas | No tool_calls |
| Reflection | Prose quality | No | Free-text critique | Message budget |
| Reflexion | Research quality | Search | Pydantic reflection | Iteration budget |
| Agentic RAG | Retrieval + answer quality | Search + vector | Graders + router | LLM judges + END |

**LangChain AgentExecutor → LangGraph:** from hidden while-loop to inspectable state machine. **Reflection → Reflexion:** from self-talk to tool-grounded revision. **Naive RAG → Agentic RAG:** from "always retrieve" to "retrieve, verify, correct, route."

---

*End of Sections 13–16. Next in the outline: Introduction to MCP (~transcript line 19838).*
