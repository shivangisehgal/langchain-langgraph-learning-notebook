# LangChain & LangGraph — Complete Self-Contained Learning Notebook
### (Plain-Language Edition)

> This is your full replacement for watching Eden Marco's Udemy course
> **LangChain — Agentic AI Engineering with LangChain & LangGraph** (3rd edition, LangChain v1.0+).
>
> **What's different about this edition:** every idea from the original notebook is still here — nothing has been cut. The difference is *how* it's explained. Dense shorthand has been unpacked into full sentences, jargon is defined the first time it appears, and each concept gets a "why does this matter" framing before the technical detail. If a sentence in the original made you re-read it three times, it's been rewritten here.

---

## How to use this notebook

1. **Read the sections in order.** Sections 1 → 28 follow the course exactly. Each section assumes you understood the ones before it, because the ideas genuinely stack: you need LCEL before agents, agents before the ReAct deep-dive, and all of that before LangGraph, MCP, and Deep Agents make sense.
2. **Actually run the code.** Every snippet comes from the real course repositories. Local copies live in the `course-code/` folder next to this file. When you want something that runs end-to-end, it's better to check out the matching GitHub branch than to copy-paste from here.
3. **Look for the 💡 Extended Notes boxes.** These are additions beyond what the lectures said — clarifications, production judgment calls, and fill-ins for lectures that had no transcript. Everything *outside* these boxes tracks the course content directly.
4. **"How it works" walkthroughs.** Any non-trivial code snippet is followed by a line-by-line explanation of what that code actually does and why each piece is there.
5. **"Test Yourself" questions.** These appear at the end of every major section. Answers are hidden in collapsible `<details>` blocks, so you can attempt them first.
6. **ASCII diagrams.** The course's visual slides (ReAct loops, RAG pipelines, LangGraph topologies, MCP architecture, Deep Agents) are reproduced as text diagrams you can read in any editor.

### Where the code comes from

| Repository / branch | What it's used for |
|---|---|
| [emarco177/langchain-course](https://github.com/emarco177/langchain-course) → `project/hello-world` | The first, simplest LCEL chain |
| same repo → `project/search-agent` | The modern `create_agent` search agent |
| same repo → `project/agents-under-the-hood` | The three "under the hood" ReAct layers |
| same repo → `project/rag-gist` | Medium Analyzer — the first RAG build |
| [emarco177/documentation-helper](https://github.com/emarco177/documentation-helper) | The full docs assistant (crawl → RAG agent → Streamlit UI) |
| [emarco177/langgraph-course](https://github.com/emarco177/langgraph-course) → `project/ReAct-agent`, `reflection-agent`, `reflexion-agent`, `agentic-rag` | All the LangGraph projects |
| Local `course-code/` folder | An offline snapshot of everything above, so you can read code without cloning |

### What you need before starting

- **Python 3.10 or newer**, **Git**, and a way to make virtual environments. The course uses `uv` (a fast, Rust-based replacement for pip) rather than conda.
- **Comfort reading and debugging code.** You do *not* need any machine learning background — that's a deliberate design choice of the course.
- **Access to an LLM.** Any of OpenAI, Anthropic, or Google Gemini works, or you can run models locally with Ollama.
- **A few optional API keys**, which you only need when you reach the sections that use them: Tavily (web search), Pinecone (vector database), and LangSmith (tracing).

---

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. The GIST of LangChain — Hello World Chain](#2-the-gist-of-langchain--hello-world-chain)
- [3. The GIST of AI Agents](#3-the-gist-of-ai-agents)
- [4. Agents Under The Hood (1/4) — Core Architecture](#4-agents-under-the-hood-14--core-architecture)
- [5. [Layer 1] The ReAct Loop](#5-layer-1-the-react-loop)
- [6. [Layer 2] Raw Function Calling](#6-layer-2-raw-function-calling)
- [7. [Layer 3] The ReAct Prompt — The Foundation of Function Calling](#7-layer-3-the-react-prompt--the-foundation-of-function-calling)
- [8. Function Calling (Theory)](#8-function-calling-theory)
- [9. The GIST of RAG — Embeddings, Vector Databases & Retrieval](#9-the-gist-of-rag--embeddings-vector-databases--retrieval)
- [10. Building a Documentation Assistant](#10-building-a-documentation-assistant)
- [11. Prompt Engineering Theory](#11-prompt-engineering-theory)
- [12. LLM Applications In Production](#12-llm-applications-in-production)
- [13. Introduction To LangGraph](#13-introduction-to-langgraph)
- [14. Reflection Agent](#14-reflection-agent)
- [15. Reflexion Agent](#15-reflexion-agent)
- [16. Agentic RAG](#16-agentic-rag)
- [17. Introduction to Model Context Protocol (MCP)](#17-introduction-to-model-context-protocol-mcp)
- [18. Using a Pre-built Server (mcpdoc) with AI Clients](#18-using-a-pre-built-server-mcpdoc-with-ai-clients)
- [19. Building MCP Servers and Clients with LangChain](#19-building-mcp-servers-and-clients-with-langchain)
- [20. Useful Tools When Developing LLM Applications](#20-useful-tools-when-developing-llm-applications)
- [21. Deep Agents](#21-deep-agents)
- [22. Deep Agents Skills](#22-deep-agents-skills)
- [23. LangChain Glossary](#23-langchain-glossary)
- [24. Industry Insights: Assaf Elovic](#24-industry-insights-assaf-elovic)
- [25. Industry Insights: Roy Miara](#25-industry-insights-roy-miara)
- [26. Agent Security](#26-agent-security)
- [27. The Dark Side of "Vibe Coding"](#27-the-dark-side-of-vibe-coding)
- [28. Bonus](#28-bonus)

---

## 1. Introduction

### Course Introduction

Eden Marco is a backend engineer who moved into AI engineering, and he's an official LangChain ambassador. This is the **third edition** of his course — he re-recorded the whole thing rather than patching the old videos, because LangChain changed too much for patching to work. It's filmed against **LangChain 1.0** (specifically around version 1.0.2, and intended to stay compatible up to roughly 1.11.x).

He rebuilt the course based on student feedback from earlier editions. The changes: significantly more content on production concerns and security, reorganized lecture ordering, and all code updated to match how modern LangChain is actually written today (with emphasis on composability and robustness).

**Why his background matters for how the course is taught:** Eden spent years doing backend engineering at cybersecurity companies. He had **zero machine learning experience before 2023** — he jumped into generative AI only once LLMs made it accessible without a research background. That's the entire pitch behind LangChain as he frames it: LangChain **commoditized** LLM application development, meaning engineers without PhDs can now build real agents and RAG systems. The course is taught from that perspective throughout — as an engineer explaining to other engineers, not as a researcher explaining models.

### Course Objectives

By the time you finish the course, you should be able to do five things:

1. **Build LLM-powered applications with LangChain** — specifically the two dominant application shapes in the industry: **agents** and **RAG**.
2. **Understand how those systems work internally.** The course's repeated promise is "no magic" — you will read LangChain's actual source code, manually implement ReAct loops, and see exactly what function calling does under the hood.
3. **Use the surrounding ecosystem**: **LangSmith** for tracing and debugging, and **LangGraph** for workflow orchestration and durable agents.
4. **Apply prompt engineering** — both the history of the field and the concrete techniques (zero-shot, few-shot, chain-of-thought, ReAct).
5. **Think about production**: testing, logging, monitoring, alerting, and security. These aren't bolted on as a final module — they're woven through the entire course as recurring discussions.

**The core mental model — two kinds of LLM application:**

```
┌─────────────────────────────────────────────────────────┐
│                  LLM Applications                       │
├──────────────────────┬──────────────────────────────────┤
│       AGENTS         │              RAG                 │
│  The LLM acts as a   │  Retrieve private or external    │
│  reasoner that picks │  documents first, then generate  │
│  tools and steps     │  an answer grounded in them      │
│  dynamically         │                                  │
└──────────────────────┴──────────────────────────────────┘
```

Almost everything else in the course is a variation, combination, or deepening of these two shapes.

**Who this is for:** software engineers and data scientists who are comfortable writing and debugging code. **No machine learning background is required.** Eden mentions that some genuinely surprising people have completed it — lawyers, doctors — but be clear-eyed: the course is technical and code-heavy.

**What's assumed but not taught:**

- **Solid Python** — you should be able to write functions and classes, and run a program without help.
- **Basic Git** — cloning repositories and making commits.
- **Virtual environments and environment variables** — how to isolate dependencies and store secrets outside your code.

This is deliberately **not** a Python 101 course. Skipping the basics is what lets the course get deep into LangChain quickly. (Course logistics, in case you're taking it on Udemy rather than reading this notebook standalone: the 30-day refund policy applies, and Eden mentions he'll personally handle refunds after that window if someone needs it.)

### Course Structure + How to Get the Best Out of It

Practical advice from this lecture, with the Udemy-platform-specific bits stripped out:

- **Watch sections in order** (1 → 2 → 3 …). This isn't a suggestion — each section builds directly on abstractions introduced earlier, and skipping ahead means you'll hit terms with no grounding.
- **Check each video's Resources tab.** That's where the Discord invite, code snippets, documentation links, and — importantly — the *specific commit link* matching what was recorded live.
- **Use the Troubleshooting section** when something breaks, and the **Theory section** whenever a term like *chain-of-thought* or *the ReAct paper* comes up and you want the background.
- **Where to ask questions:** Discord is preferred for course questions; LinkedIn and Udemy DMs also work. Community Q&A frequently already contains the answer to whatever bug you just hit, so search before posting.

### Course's Community

Discord is the main support channel — for configuration problems, error messages, and conceptual questions alike. Eden is most available on weekends and checks in roughly twice a day on weekdays.

His strongest advice here: **ask early, and don't self-censor.** There are no stupid questions, and in his experience, when one person asks something, many other students had the exact same question but stayed quiet. A quick answer from him or a fellow student can save you hours of frustrating debugging.

One practical gotcha: if the Discord invite link fails, **log into Discord first**, *then* click the invite. That resolves most connection issues.

### Course Resources

*(This lecture had no transcript available.)*

What you should expect to find here: the GitHub repository with per-video commit links, the Discord invite, documentation bookmarks, and shared public LangSmith traces.

**One practical tip that matters a lot:** when following along, always check out the **exact commit** linked in the lecture rather than the latest `main` branch. Repository tips-of-tree drift as the instructor keeps updating things, and following along against a moved target is a common source of confusion.

### Test Yourself

**Q1.** What are the two primary application categories Eden frames for LLM apps?
**Q2.** What is *not* required to take this course — machine learning, or software skills?
**Q3.** Why remake the entire course for LangChain 1.0 instead of just patching the old videos?

<details>
<summary>Answers</summary>

1. **Agents** and **RAG** applications.
2. Prior **machine learning** knowledge is not required; **software/Python comfort** is.
3. LangChain's APIs and best practices moved too fast for patching. Re-recording keeps all code current with 1.0+, incorporates accumulated student feedback, and removes the constant friction of hitting deprecated APIs mid-lecture.

</details>

---

## 2. The GIST of LangChain — Hello World Chain

### What is LangChain? (Under ~6 Minutes)

**The definition:** LangChain is an **open-source framework** that makes it easier to build applications powered by LLMs, by giving you a set of tools and abstractions. It's been widely adopted in the industry, mostly by developers who want to *use* models as a **black box** rather than train or understand them.

**Why it exists — the "stitching" problem.** Suppose you want to build a real app on top of a powerful model like Claude Sonnet. Very quickly you'll need:

- **Private data the model was never trained on** — your PDFs, your emails, your Notion database.
- **Dynamically constructed prompts** that change based on user input.
- **Conversation history** stored between the user and the AI.
- **The ability to switch vendors easily** — today Claude, tomorrow Mistral, next month a local Llama model.
- **Tool use** — connecting the model to Google Search, or letting it make arbitrary API calls based on what the user asked.

You could build all of that yourself, but it's a lot of moving parts to stitch together and keep in sync. LangChain does the heavy lifting through a set of modules:

| Module | What it handles |
|--------|-----------------|
| **Chat models** | One consistent interface across every LLM vendor — Eden's phrase is that you can "switch models like you switch your socks" |
| **Prompts / prompt templates** | Managing, optimizing, and serializing prompts; injecting variables into them at runtime |
| **Document loaders** | Loading Notion pages, PDFs, emails, and thousands of other sources into a single unified `Document` format |
| **Agents / tools / LangGraph** | Letting the model reason and then invoke real capabilities — what Eden calls giving the LLM "superpowers" |
| **Tracing (LangSmith)** | Debugging and monitoring LLM apps once they're running |

```
User input ──► PromptTemplate ──► ChatModel ──► (optional parse/tool/chain)
                    │                  │
                    └──── LCEL pipe ───┘
```

Over this course you'll implement both agents and RAG end-to-end, including reading the actual implementations rather than treating them as a black box.

💡 **Extended Notes — Why vendor lock-in matters more than it sounds**

The chat model abstraction isn't just developer-experience sugar. Production teams routinely A/B test different models against each other, need fallbacks when a vendor has an outage, or have to move to locally-hosted models for privacy or compliance reasons. Having a single interface (`invoke`, a shared message format, a shared tool-binding API) turns all of those from a rewrite into a **config change**.

You'll appreciate this much more in Sections 4–7, once you've seen how much per-vendor schema and format wrangling LangChain was quietly doing on your behalf.

### What Are We Building? LangChain Hello World Chain

**The goal is to learn by doing.** You'll build a tiny chain that:

1. Takes a block of text about **Elon Musk** (copied from Wikipedia),
2. Formats it through a **prompt template**,
3. Sends it to an LLM,
4. Returns a **short summary plus two interesting facts**.

Along the way you'll meet: prompt templates, chat models, chains (LCEL), debugging and tracing, **OpenAI GPT-5** (or any first-tier model — Gemini or Claude work identically), and **Ollama** for running open-weights models locally (the demo uses **Gemma 3**).

### Project Setup

This is the bootstrap pattern used for every project in the course:

1. **Clone the course GitHub repo** (linked in the video resources and the intro).
2. **Create a fresh working branch**, for example `project/hello-world`, using `git checkout --orphan`. An orphan branch has no history from the parent, so you can optionally wipe all files for a genuinely clean slate. (Note: if you clone the repo later, that branch name may already exist — just pick a new name.)
3. **Use `uv`** — a Rust-based, very fast alternative to pip. Poetry or Pipenv work equally well; the course only ever needs "install packages, make a virtual environment, run a file."
4. **Run `uv init`** — this creates `main.py` and `pyproject.toml`.
5. **Install the packages:**

```bash
uv add langchain
uv add langchain-openai          # the OpenAI integration package
uv add python-dotenv             # loads variables from a .env file
uv add black isort               # code formatting (optional, just hygiene)
# later on: langchain-ollama, for running local models
```

**Why are provider integrations separate packages?** Historically every provider was bundled inside core LangChain. They were split apart for two reasons: so you don't download a hundred providers' dependencies just to use one, and so each vendor can maintain and version their own package independently. The same pattern applies to `langchain-tavily`, `langchain-ollama`, and every other integration you'll install later.

6. **Add a standard Python `.gitignore`.** Never commit `.venv` or any secrets.
7. **Create a `.env` file:**

```bash
OPENAI_API_KEY=sk-...
# Optional, needed later:
# GOOGLE_API_KEY=...          # for Gemini, via Google AI Studio + langchain-google-genai
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=...
# LANGSMITH_PROJECT=hello-world
```

**A serious security note:** API keys are passwords. There are malicious scrapers actively hunting for leaked keys on GitHub. Two protections: set a **budget limit** on your OpenAI dashboard so a leaked key can't drain your account, and never commit the `.env` file. Also note that without billing set up or credits on your account, you'll typically hit **429-style rate-limit failures** rather than a clear "you have no money" error.

**The variable name `OPENAI_API_KEY` must be exact** — LangChain looks up that specific name in your environment to find credentials. Same principle applies to `TAVILY_API_KEY` and others later.

Verify it loaded correctly:

```python
from dotenv import load_dotenv
import os

load_dotenv()
print(os.environ.get("OPENAI_API_KEY"))  # should print your key locally — delete this line before committing
```

Then commit the boilerplate with a message like `environment setup`. The course links a specific commit per video so you can match state exactly.

### LangChain Fundamentals: Prompt Templates, ChatModels, and Chains

**What a prompt is:** simply the text input you give the model, which it processes and responds to. (Increasingly prompts can be multimodal too — images, audio.) The formal breakdown of what a prompt is *composed of* comes later, in the Prompt Engineering Theory section.

**What a PromptTemplate is:** a wrapper class that parameterizes a prompt so you can reuse it with different inputs.

- Concretely: one template, run with product = "cat food" one time and product = "piano" another time, gives you two different outputs from the same reusable structure.
- Its core job is to **format variables into the string** that eventually gets sent to the LLM. On top of that it gives you validation, reusability, and first-class tracing support.

**What chat models are:** classes like `ChatOpenAI` and `ChatOllama` — your primary interface to any LLM. There's an important historical shift here. **Older LLMs** were literally string in → string out. **Modern chat models** are built for conversation: they take a **list of structured messages** (system instructions, human messages, AI messages, tool messages) and return an **AI message**. Beyond generating text they also handle tool calling and expose token metadata. (The Glossary section covers the message types in full.)

**A habit worth building:** Cmd/Ctrl-click into the LangChain source code whenever you use a class. This course deliberately opens implementations over and over — it's the main reason it can promise "no magic."

**What a chain is:** a workflow that connects components so that the **output of step N becomes the input of step N+1**. A step can be an LLM call, a prompt template, a data transformation, a tool call, or even another chain nested inside.

This composability is genuinely why LangChain took off. It let you write "format the input → call the LLM → parse the output → hit an external API → feed that into another LLM" as a clean pipeline rather than as spaghetti code.

### Building a LangChain Chain to Summarize Text

Here's the actual course code (`project-hello-world/main.py`):

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

1. **`load_dotenv()`** — reads your `.env` file and loads `OPENAI_API_KEY` (and later the LangSmith variables) into the running process's environment.

2. **`information`** — this is raw context the model hasn't memorized for this particular call. You're injecting it through the template. This is exactly the same pattern you'll use for private documents when you get to RAG — the only difference there is that the text gets *retrieved* rather than pasted in by hand.

3. **`summary_template`** — the instruction text, with an `{information}` placeholder. Important: those curly braces are **template variables**, not Python f-string evaluation. Nothing gets substituted at the moment you define the string; substitution happens at runtime when you invoke.

4. **`PromptTemplate(input_variables=[...], template=...)`** — declares which keys the template expects. If you misspell `information` when invoking, you get a clean, immediate error instead of a silently broken prompt being sent off to the API and billed for.

5. **Why not just use an f-string?** Four reasons: templates *enforce* that you supply the required variables; they're reusable across multiple chains; they're first-class objects for **logging and tracing** (they show up as their own step in LangSmith); and they interact better with output parsers and injection-aware formatting later on.

6. **`ChatOpenAI(temperature=0, model="gpt-5")`** — the chat model wrapper around OpenAI's SDK. It picks up credentials from `OPENAI_API_KEY` automatically.

7. **`temperature`** — the randomness dial on the model's output:
   - **0 to 0.3**: more deterministic and repeatable. Use this for summaries, code generation, and tests.
   - **0.8 and above**: more creative and varied. Use this for poetry, brainstorming, and idea generation.

8. **`chain = summary_prompt_template | llm`** — this is **LCEL** (the LangChain Expression Language). The `|` pipe operator builds a **Runnable** pipeline: whatever comes out of the left side is fed into the right side.

9. **`chain.invoke({"information": information})`** — runs the whole pipeline:
   - The prompt template is invoked first, producing a `PromptValue` (essentially a formatted string with some extra structure).
   - That gets piped into `llm.invoke(...)`.
   - The result is an **`AIMessage`** object; the actual text is in **`.content`**.

10. **`print(response.content)`** — prints just the generated text.

```
┌──────────────────┐     PromptValue      ┌─────────────┐     AIMessage
│ PromptTemplate   │ ───────────────────► │  ChatModel  │ ──────────────► content
│ {information}    │                      │  (OpenAI)   │
└──────────────────┘                      └─────────────┘
         ▲
         │ invoke({"information": bio})
```

💡 **Extended Notes — LCEL is the hard part, and that's expected**

Eden says outright that LCEL and the `|` operator are often **harder to internalize than agents are**. Don't be discouraged if it feels slippery at first.

The mental model to hold for now: it's **left-to-right dataflow**, not magic. Every component is a `Runnable`, and every `Runnable` supports `.invoke()`, `.stream()`, and `.batch()`. You do *not* need to understand the full LCEL composition theory to finish Hello World — you just need "the template formats the text, then the model generates from it."

### Debugging and Tracing Our LangChain Chain

If you set a breakpoint and inspect the result in a debugger:

- **`response`** is an **`AIMessage`** object, not a plain string.
- The generated text lives in **`response.content`**.
- Other fields worth looking at: `type`, `response_metadata` (which contains the model name and the finish reason), and token usage counts. Those last two become genuinely important later when you're monitoring agent costs.

The full set of message types and their richer fields is covered in the Glossary section of the course.

### Using Local Open-Weights Models with LangChain and Ollama

**This demonstrates LangChain's core strength:** to swap models, you change the chat model client and *nothing else*. The rest of the chain is untouched.

**Setup:**

1. Install [Ollama](https://ollama.com) for your operating system.
2. Pull a model: `ollama pull gemma3:270m` (or something larger if you have the disk and RAM).
3. Verify it works from the CLI: `ollama list` to see what you have, `ollama run <model>` to chat with it directly.
4. Then in your code:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(temperature=0, model="gemma3:270m")
# Comment out the ChatOpenAI line; the chain stays exactly the same: template | llm
```

**The tradeoff Eden observed in the lecture:** tiny local models are **fast and free**, but they often **don't follow instructions as reliably**. In the actual demo, the small Gemma model produced a summary but failed to give a clean, separate "two interesting facts" section — it didn't fully do what was asked.

That's the general pattern with lightweight open-weights models: you gain speed and cost, you lose answer quality and instruction-following. For the **agentic work later in the course**, Eden specifically recommends using something with stronger reasoning *and* **function calling support** — he suggests **GPT-OSS** if you want to stay local and have the disk space for it, since it supports function calling and is suited for agentic workloads.

Cloud-hosted open-weights providers (Groq, for example) follow the identical pattern: install the provider's package, swap the chat model class, keep the same chain.

### Integrating LangSmith for LangChain Application Tracing

**LangSmith** is LangChain's platform for tracing and debugging LLM applications. The free tier is more than enough for everything in this course.

**Setup steps:**

1. Sign up at LangSmith, then click **Set up tracing** in the dashboard.
2. Generate an API key.
3. Set these environment variables in your `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=hello-world
# If you're OUTSIDE the US: you must also set LANGSMITH_ENDPOINT to the EU endpoint,
# otherwise you'll get authentication errors when LangChain tries to send traces.
```

The `LANGSMITH_PROJECT` value is just a string you choose — it's the bucket that all your traces get grouped under in the LangSmith UI.

4. **No code changes are needed** for basic tracing. Keep using normal LangChain objects; runs automatically appear in your project.

**What a Hello World trace actually shows you:**

- A **RunnableSequence** at the top level (the prompt → chat model pipeline).
- Inputs shown as Human/formatted messages, outputs as an AIMessage.
- Latency, time-to-first-token (which matters a lot for streaming UIs), success/failure status, and total token counts.
- The nested steps broken out individually: first the PromptTemplate formatting the string, then ChatOpenAI (or ChatOllama) generating.

You can also add **tags** to filter runs later, and the aggregate view gives you stats like error rates and median/P50/P90 token counts. Traces can be made **public and shareable** via a link — that's how Eden shares his own runs in the video resources.

Commit the working chain with a message like `hello World Chain`.

### Semantic Versioning in LangChain

The course was filmed against **LangChain ~1.0.2**. Check your `uv.lock` file to see the exact version that got resolved for your project.

- **Patch and minor updates within 1.0.x** are usually fully compatible — you shouldn't need to change anything.
- **Breaking major changes** get handled by Eden updating the course. If you notice the videos diverging from current APIs, ping Discord.

💡 **Extended Notes — Managing versions in a fast-moving ecosystem**

Always pin your versions via the lockfile for reproducibility. When you do upgrade across a major version, re-run smoke tests on your chains and agents — and pay particular attention to **tool calling** and the **`create_agent` APIs**, which changed substantially between the classic LangChain agent style and the 1.0 approach.

### Test Yourself

**Q1.** What does `summary_prompt_template | llm` create, and what method runs it?
**Q2.** Name two benefits of `PromptTemplate` over an f-string.
**Q3.** Which environment variable must exist for `ChatOpenAI` to find credentials by default?
**Q4.** What's the main quality tradeoff when switching Hello World to a tiny Ollama model?

<details>
<summary>Answers</summary>

1. An LCEL **Runnable** chain; you run it with **`.invoke(...)`**.
2. Any two of: variable validation with clean errors, reusability across chains, first-class tracing/logging, and safer structured formatting for output parsers.
3. **`OPENAI_API_KEY`**.
4. Speed and cost improve significantly, but instruction-following and answer quality typically drop.

</details>

---

## 3. The GIST of AI Agents

### What Are AI Agents? A High-Level Overview

Ask ten people what an "AI agent" is and you'll get ten different answers. But they share a common core, and this is the definition aligned with the LangChain ecosystem:

> **An agent is a software system that uses an LLM as a reasoning engine to decide what actions to take, and then executes those actions.**

**The critical distinction — agents versus chains:**

| | Chain / Runnable | Agent |
|--|------------------|-------|
| **Who controls the flow** | The **developer hardcodes** the sequence | The **LLM decides** what comes next |
| **What the LLM does** | Usually handles one step (summarize, extract) | Acts as both **reasoner and actor** over a set of tools |
| **Flexibility** | Predictable, fixed pipeline | Dynamic tool use for open-ended tasks |

The single sentence to remember: in a chain, you might use an LLM somewhere in the middle, but **you** defined the entire control flow. In an agent, **the LLM** is deciding what to do next.

**What tools are:** capabilities you write ahead of time — an API call, a database query, code execution, a web search. The essential point is that **the LLM doesn't have internet access or database access by itself**. Those are things you explicitly grant it. This is what Eden calls giving the LLM "superpowers," and why he describes the idea as mind-blowing: an extremely capable reasoning system, connected to the ability to actually do things, can automate work that was previously impossible to automate.

**ReAct agents** (from **Rea**soning + **Act**ing): these combine chain-of-thought-style reasoning with real tool actions in an **iterative loop** that continues until the task is complete. Both LangChain and LangGraph ship prebuilt ReAct-style agents that handle state, tools, and customization for you.

**What this section covers:** just the **`create_agent` interface** — how to equip an LLM with tools and get an agent running quickly. The next major block of the course (Sections 4–7) peels back every layer of that abstraction until you've rebuilt it from scratch.

### What Are We Building? AI Job Search Agent

**The demo that motivates it:** ChatGPT with Web Search enabled. The query is "search for three Bay Area AI engineer / LangChain job postings on LinkedIn and list their details." ChatGPT goes and searches, then returns the postings — each one **with its source URL**.

**Why those source URLs matter so much:** LLMs hallucinate. They can generate plausible-sounding garbage. When you show the user a source URL, they can click through, verify the claim, cross-check it, and decide for themselves whether to trust it. That grounding is what creates trust between the user and the system. Without sources, the user has no way to evaluate whether an answer is trustworthy.

(This is also a preview of **generative UI** — where the application visually reflects what the agent is doing step by step, like showing LinkedIn icons as it searches. That's covered later in the course.)

**The core insight:** models are text-in/text-out (or multimodal), but they're **static in time**. They're trained on a corpus that stops at some point and they have no real-time information and no internet access. If you want to give them access to current information or external capabilities, you must **provide them with tools**. That's what you'll build here, using LangChain plus Tavily.

A historical aside: when Eden first taught this material around 2022, chat interfaces had no built-in search at all. Now every major chat app has it. The underlying agent pattern is exactly the same — it just became table stakes.

### The Evolution of LangChain ReAct Agents

Here's how the agent APIs evolved. Remember the *shape* of this progression rather than the exact dates:

```
2022  ReAct prompting only (text-based Thought/Action/Observation, parsed with regex)
  │
  ▼
Models gain native function/tool calling
  │
  ▼
Tool-calling agents (structured calls replace fragile prompt parsing)
  │
  ▼
LangGraph-backed ReAct (durable execution, persistence, fine-grained control)
  │
  ▼
LangChain 1.0  create_agent(...)   ← clean high-level API built on the LangGraph ReAct agent
```

Reading that in words:

- **The original ReAct agent** (November 2022) relied purely on ReAct *prompting* — the model reasoned about actions and observations in a text format that your code had to parse.
- **As models gained native function calling**, the architecture shifted to **tool-calling agents**, which used structured function calls instead of prompt-based tool selection. This made tool execution far more reliable and efficient.
- **The next major leap** was the LangGraph ReAct agent, which kept function calling but rebuilt the agent on top of LangGraph's low-level orchestration framework — adding durable execution, persistence, and the fine-grained control needed for production-grade applications.
- **LangChain 1.0** introduced **`create_agent`**, a clean high-level interface that's powered by the battle-tested LangGraph ReAct agent underneath.

**The course's teaching strategy:** start with the **newest, highest-level `create_agent`** (just the interface — how to use it), then walk **backward** through each layer until you've rebuilt an agent from raw ReAct prompts. The goal is to give you real intuition for building production agents, not to teach you to cargo-cult one API call.

### Setting Up the Environment for a LangChain Search Agent

Branch: `project/search-agent` (or the related ReAct search branch — follow the video resources, since Eden notes he committed some sections to specific branches for technical reasons).

```bash
uv init
uv add langchain langchain-openai langchain-tavily tavily-python python-dotenv black isort
```

**About Tavily:** it's a web search API built specifically for AI agents, and it's currently the most popular choice for giving agents search capability — it's featured in the official LangChain documentation as the default search service for agents. Eden's read on why: they have an excellent API, it scales well, it's easy to use, and they were essentially first to market on agent-focused search.

The free tier is about **1,000 API requests per month**, which is far more than this course needs. Beyond plain search, Tavily also offers **Crawl**, **Map**, and **Extract** APIs, which get used later in the RAG and documentation-assistant sections.

If you try their playground with a query like "what are the latest Anthropic models," you'll see the response includes both the **source URL** and the **content snippet** — exactly the two things an agent needs.

**A conceptual point worth internalizing:** `langchain-tavily` is a LangChain *provider* package, but Tavily isn't an LLM vendor. A LangChain provider doesn't have to sell models — it can be any vendor exposing services through APIs. Tavily exposes search and scraping; the LangChain integration wires those into LangChain's tool objects so they drop straight into your agents.

Add to `.env`:

```bash
OPENAI_API_KEY=...
TAVILY_API_KEY=...          # again, the exact name matters — LangChain looks it up
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=search-agent
```

To get the Tavily key: log in, go to your environment page, click the plus button, name the key, optionally set a monthly usage limit, and create it. Put `load_dotenv()` at the top of `main.py` as before.

### Creating Your First LangChain Agent: Tools and LLMs

**An agent needs a minimum of two ingredients:** (1) **tools**, and (2) an **LLM** to act as the reasoning engine.

**What counts as a tool?** Any Python function the agent is allowed to execute — an API call, a database query, arbitrary code, a search. Since you write the implementation, the capabilities are effectively unbounded.

**How to create one:**

1. Write a normal Python function with **type hints** and a **docstring**.
2. Decorate it with `@tool` from LangChain.
3. Under the hood, the function's **name, arguments, and docstring** become metadata that gets sent to the LLM. The model uses that metadata (via **function calling**) to decide *whether* to call the tool and *how* to call it.

The first lecture deliberately uses a **stub** tool — a fake search that always returns "Tokyo weather is sunny" — to prove the loop works before wiring in real search.

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

llm = ChatOpenAI()  # the default model at filming time was gpt-3.5; set gpt-5 explicitly later
tools = [search]
agent = create_agent(model=llm, tools=tools)

result = agent.invoke(
    {"messages": HumanMessage(content="What's the weather in Tokyo?")}
)
# LangChain accepts either a single message or a list under the "messages" key
```

**A rule of thumb for docstrings that matters more than it looks:** be **explicit and unambiguous**. The model is choosing which tool to call purely from natural language descriptions. A vague docstring produces a confused agent.

### From Query to Answer: How a LangChain Agent Thinks

Run the stub agent and inspect `result["messages"]`. You'll find four messages, and understanding what each one is dissolves most of the "magic" around agents:

1. **HumanMessage** — the user's original query.
2. **AIMessage containing `tool_calls`** — the model chose `search(query="weather in Tokyo")`. Note carefully: **this is not the final answer, and it is not the tool being executed.** It's only the model *naming* which tool it wants called and with what arguments.
3. **ToolMessage** — the observation, i.e. the actual result of running the tool ("Tokyo weather is sunny…"). A ToolMessage is simply the data structure LangChain uses to represent a tool execution result.
4. **AIMessage** — the final natural-language answer, this time with no tool calls attached, because the model now has everything it needs.

If you open this in LangSmith, the run is labeled **LangGraph** — that's because `create_agent` runs on LangGraph internally. Don't worry about that yet; it's covered fully later.

Two details visible in the trace that are worth calling out:

- **The first LLM call includes the tool schemas** in the request. LangChain sends the model not just the question but also the full description of every tool it's allowed to use.
- **LangChain's runtime executes the function**, then makes a *second* LLM call containing the original input, the model's decision to call the tool, and the tool's result. With all that context, generating the final answer is easy for the model.

```
User ──► LLM(+tools) ──► tool_call? ──yes──► execute tool ──► ToolMessage ──┐
              ▲                              │                              │
              └──────────── append history ──┴──────────────────────────────┘
                         no tool_call → Final Answer
```

**The two conceptual pieces to separate in your head:**

1. **The reasoning engine** (the LLM) — decides which tool and which arguments.
2. **The agent runtime** (LangChain/LangGraph) — actually executes the tool, manages the message list, and runs the loop.

Keeping these separate is the single most useful mental model for everything that follows.

### Integrating Real-World Search with Tavily and LangChain Tools

First, the DIY approach — a custom tool wrapping the Tavily SDK directly:

```python
from tavily import TavilyClient

tavily = TavilyClient()  # reads TAVILY_API_KEY from the environment

@tool
def search(query: str) -> str:
    """Search the internet..."""
    return tavily.search(query=query)
```

You invoke the agent exactly the same way; the only difference is that LangSmith now shows real search payloads in the tool observations instead of your stub string.

Try the job query:

> search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?

Something interesting happens with capable models: they may emit **multiple parallel tool calls** — several different LinkedIn/search queries fired at once. The final LLM turn then consumes all of those ToolMessages together.

**Best practice — prefer the vendor's official LangChain tool:**

```python
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)
```

**Why this is better than your own wrapper:** Tavily's own team wrote the tool descriptions and argument schemas, and they know their API best. Their tool exposes parameters like `include_domains` and `search_depth=advanced` with proper descriptions, so the agent makes far more precise calls than a thin hand-rolled wrapper would allow.

The course commits show this evolution deliberately: custom `@tool` → official `TavilySearch` → structured output.

### Structured Output with LangChain Agents Using Pydantic

Most agent answers come back as free text. But production applications need **downstreamable structure** — JSON or Pydantic objects you can serialize, send from an API, and render in a UI.

You get this by passing `response_format=` into `create_agent` with a Pydantic schema:

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
    print(result)  # the result dict now includes structured_response: AgentResponse
```

#### How it works

- **Pydantic `BaseModel`** gives you a base class to inherit from when defining a structured data schema. It handles data parsing, serialization, and automatic type validation for you.

- **`Field(description=...)`** attaches metadata to each attribute, describing what belongs in that field. This description is read by the **model**, which is how it knows what to put there. (Amusing detail from the lecture: the description above has a typo — "Thr" instead of "The" — and Eden points out the LLM handles it fine. Clarity still helps, but small typos aren't fatal.)

- **`default_factory=list`** means that if the agent doesn't supply any sources, the `sources` attribute is populated with an empty list rather than erroring.

- **Nesting matters here:** `Source` is a separate Pydantic model used *inside* `AgentResponse`. This demonstrates that you can build arbitrarily nested structured schemas, not just flat ones.

- **`response_format=AgentResponse`** — with this one argument, the agent's result gains a **`structured_response`** key, typed as `AgentResponse`, containing `answer` and `sources[].url`.

- LangSmith shows the structured payload attached to the final turn, so you can verify it visually.

- **A hint about what's happening underneath:** this is implemented via function calling / structured-output strategies, which is exactly what the next lecture explains.

### [THEORY] Predictable Agent Responses with LangChain Structured Output

Structured output lets agents return data in a specific, predictable format — a Pydantic object, a JSON response, or a dataclass — rather than raw text, so you can use it programmatically downstream.

There are **two implementation strategies** underneath:

1. **Provider strategy** (LangChain's default when the model supports it) — uses the model vendor's own **native structured-output API**. Most top-tier models offer this natively. The schema can be a Pydantic model, a dataclass, a TypedDict, or a raw JSON Schema. The key consequence: responsibility for producing valid output sits with the **model provider**, not with LangChain. As Eden puts it half-jokingly — if you get a wrong answer, you take it up with the vendor, not LangChain.

2. **Tool strategy** — for models that lack native structured output but *do* support tool calling (which is essentially all top-tier models). LangChain exposes **exactly one tool** whose schema *is* the object you want, and instructs the model that it must always call that tool. By forcing that single tool call, the schema gets enforced on the output. This is historically how structured output evolved before native support existed.

LangChain automatically picks the provider strategy when it's available, unless you explicitly override it. Both strategies accept the same family of schema types.

### Test Yourself

**Q1.** What is the key control-flow difference between a chain and an agent?
**Q2.** In a tool-using run, what does the first AIMessage usually contain before the final answer?
**Q3.** Why prefer `TavilySearch` over a hand-rolled `@tool` wrapping the SDK?
**Q4.** Where does a Pydantic `response_format` show up in the agent's result dict?
**Q5.** Name the two structured-output strategies LangChain uses.

<details>
<summary>Answers</summary>

1. Chains hardcode the sequence of steps; agents let the **LLM choose** the next action or tool.
2. **`tool_calls`** — which tool to call and with what arguments. Not the final user-facing answer.
3. The vendor maintains better schemas, descriptions, and argument options, which leads to more precise tool selection by the agent.
4. Under the **`structured_response`** key.
5. **Provider strategy** and **tool strategy**.

</details>

---

## 4. Agents Under The Hood (1/4) — Core Architecture

### Introduction to The Core Architecture of AI Agents

This is Eden's favourite section of the course, and the promise is specific: by the end of it you will know exactly how agents work internally, with the deepest understanding you can have of them.

**The teaching approach is to peel the onion, one layer at a time:**

| Layer | What you build | What's abstracted away |
|-------|----------------|------------------------|
| **0** (already done) | `create_agent(model, tools)` | Absolutely everything |
| **1** | A manual agent **while-loop** using LangChain primitives (`@tool`, `bind_tools`, message objects) | Vendor JSON schemas, raw SDK details |
| **2** | The same loop using the **raw Ollama SDK** with hand-written JSON tool schemas | LangChain's chat and tool APIs |
| **3** | The same loop using a **ReAct prompt plus regex parsing** — no function calling at all | Native tool-calling APIs entirely |

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

**Why each layer teaches you something different:**

- **Layer 1** shows you what the agent loop actually *is* — a while loop leveraging function calling — while LangChain still saves you the boilerplate.
- **Layer 2** strips LangChain away entirely and makes you write all the JSON schemas by hand. This is where you genuinely appreciate what LangChain was doing for you: giving you one single interface, and the ability to switch models freely. That flexibility matters in production, because when a new model launches you want to switch to it with minimal code change.
- **Layer 3** goes below function calling itself, rebuilding the agent with just a ReAct prompt, regular expressions, and a scratchpad. This is how agents were originally implemented when they first appeared, and seeing it is what makes every layer of abstraction above it feel *necessary* rather than arbitrary.

**Eden's strong recommendation, repeated several times:** do this section **hands-on**. Write the code, run it, and read the traces. Don't just watch. The whole value is in seeing the magic dissolve as you type it.

**Models used:** Ollama running **Qwen3** (chosen because it's lightweight and supports function calling), with occasional OpenAI. Any function-calling-capable model works — it doesn't have to be open-weights.

### What Are We Building? An E-Commerce Agent

The running example is a hardware store bot. The catalog has **laptops, headphones, and keyboards**, and there are **bronze, silver, and gold** discount tiers.

**Two tools:**

1. `get_product_price(product)` → looks up the catalog price
2. `apply_discount(price, discount_tier)` → returns the discounted price

**The example query:** *"What is the price of a laptop after applying a gold discount?"*

Notice that answering this **requires two sequential tool calls** — you can't apply the discount until you know the price. That's what makes it a good test of the agent loop rather than a single-shot tool call.

With `create_agent` you'd just pass both tools and be done. Here you'll implement a **lean agent loop by hand**, so that the loop diagram becomes actual code you wrote.

**A deliberate design detail:** the discount percentages are intentionally strange — **5% / 12% / 23%** rather than round numbers. This is so the model **cannot guess or hallucinate the arithmetic**. It has no choice but to actually call `apply_discount`, which makes the tool-calling behaviour unambiguous when you inspect the trace.

### [Theory] The Gist of ReAct

**The agent loop, the ReAct loop, and the ReAct algorithm are all the same thing.** This is the engine behind Claude Code, Gemini CLI, Codex, and Devin-style agents. It originates from the paper *ReAct: Synergizing Reasoning and Acting in Language Models* (by researchers at Princeton and Google). Eden covers the paper itself in the Prompt Engineering Theory section.

**The loop, step by step:**

1. **A user query enters** the system.
2. **Thought** — the LLM sees the system prompt, the **tool descriptions**, and the conversation history so far, and decides either to make a tool call *or* to give a final answer.
3. **Action** — your application executes the chosen tool. Critical point: **the LLM only *names* the call; your code runs it.**
4. **Observation** — the result that comes back from the tool.
5. **Append to history** (the message list, sometimes called the **scratchpad**) and loop back to step 2.
6. **Exit** when the LLM returns no tool call — i.e. it produces a final answer instead.

**Walking through the e-commerce example concretely:**

```
Iter 1: Thought → Action get_product_price("laptop") → Observation 1299.99
Iter 2: Thought → Action apply_discount(1299.99, "gold") → Observation 1000.99
Iter 3: Thought → Final Answer (no tool call this time)
```

That's it. It is, quite literally, "just" a while loop. The simplicity is the point — and it's simultaneously one of the most powerful patterns in modern software.

### Setup

Branch: `project/agents-under-the-hood`.

```bash
uv init
# delete the auto-generated main.py if you don't want it
uv add langchain langchain-ollama langchain-openai python-dotenv black isort
```

You need `langchain-ollama` because you'll run open-weights models locally, and `langchain-openai` because you'll switch models later in the section to compare behaviour.

`.env`:

```bash
OPENAI_API_KEY=...          # optional; only for the model-switch demos
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=ReAct Under The Hood
LANGSMITH_TRACING=true
# Ollama needs no API key at all — everything runs locally on your machine
```

Pull and serve the model:

```bash
ollama pull qwen3:1.7b      # about 1.4GB
ollama run qwen3:1.7b       # quick smoke test — chat with it, then type /bye
ollama serve                # start the server so your Python code can reach it
```

**Why the 1.7 billion parameter version specifically:** Eden tried the 0.6b variant first and found it too weak to reliably do function calling. The 1.7b model is stronger while still being light on disk. You can substitute any function-calling-capable model — OpenAI, Anthropic, or Gemini all work identically here.

### Test Yourself

**Q1.** Put the four layers in order, from most abstract to most raw.
**Q2.** Who executes the tool — the LLM or your application?
**Q3.** Why does the demo catalog use non-obvious discount percentages?

<details>
<summary>Answers</summary>

1. Layer 0 (`create_agent`) → Layer 1 (LangChain loop) → Layer 2 (raw function calling) → Layer 3 (ReAct prompt).
2. **Your application** executes the tool. The LLM only selects the tool name and arguments — or declares a Final Answer.
3. So the model **cannot hallucinate the math**. With weird percentages it's forced to actually call `apply_discount`, which makes the tool-calling behaviour visible and unambiguous.

</details>

---

## 5. [Layer 1] The ReAct Loop

### Writing Tools

File: `1_agent_loop_langchain_tool_calling.py`.

**The imports, and why each one is there:**

- **`init_chat_model`** — a very convenient LangChain utility that creates a chat model from a plain **string**, like `"ollama:qwen3:1.7b"` or `"openai:gpt-5"`. All you need is the matching provider package installed (`langchain-ollama`, `langchain-openai`, etc.). LangChain even maps bare model names to providers automatically — `gpt-o1` and `o3` map to OpenAI, DeepSeek maps to DeepSeek, Claude maps to Anthropic. This is one of LangChain's real strengths: switching models becomes a one-string change.
- **`@tool`** — turns a plain Python function into a LangChain tool, automatically generating the JSON schema from your type hints and docstring.
- **`HumanMessage`, `SystemMessage`, `ToolMessage`** — portable message types that work identically across every provider. `SystemMessage` marks system instructions, `HumanMessage` marks user input, and `ToolMessage` carries a tool's execution result.
- **`MAX_ITERATIONS = 10`** — a safety cap so the loop can't run forever. Eden picked 10 for no particular reason; any number above two works. It's purely a heuristic guard rail.
- **`@traceable(name="LangChain Agent Loop")`** — wraps the whole run under one LangSmith span. You need this because a hand-written agent loop doesn't automatically nest its traces the way `create_agent` does.

**The two tools, from the actual course code:**

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

The `print` statements aren't decoration — they let you visually confirm in the terminal that the tool genuinely executed, and with which arguments.

**The point to internalize:** your **docstrings and type hints literally are the tool's API** as far as the model is concerned. `@tool` takes them and normalizes them into whatever schema format the specific vendor expects. Write them as if you're writing documentation for a colleague who will only ever read that one paragraph.

### Tool Binding and Defensive Prompting

```python
tools = [get_product_price, apply_discount]
tools_dict = {t.name: t for t in tools}

llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
llm_with_tools = llm.bind_tools(tools)
```

**What `bind_tools` does:** from this point on, every `invoke` call on `llm_with_tools` automatically attaches the tool schemas to the request. This requires a **function-calling-capable** model — if the model doesn't support it, this won't work. The exact same code path works for OpenAI or Anthropic; you only change the string passed to `init_chat_model`.

**What `tools_dict` is for:** the model returns a tool *name* as a string. You need a way to get from that string to the actual runnable tool object so you can execute it. This dictionary is that lookup table.

**The defensive system prompt** — this turns out to be critical, especially with smaller open-weights models. The four rules:

1. **NEVER guess prices** — you must call `get_product_price` first.
2. Only call `apply_discount` **after** you have a real price observation, and pass that exact number through.
3. **NEVER compute discounts with mental math** — always use the tool.
4. If the discount tier isn't specified, **ask the user** — don't assume one.

Eden added these rules only *after* watching Qwen hallucinate prices in testing. This is a genuinely useful lesson about working with smaller models: they will confidently invent numbers unless you explicitly forbid it.

**A worthwhile exercise:** delete these rules, re-run, and watch the failures happen. Seeing the model confidently make up a price teaches you more about defensive prompting than reading about it.

Seeding the message list:

```python
messages = [
    SystemMessage(content="You are a helpful shopping assistant. ... STRICT RULES ..."),
    HumanMessage(content=question),
]
```

### Understanding the ReAct Agent Loop in LangChain

Here's the complete loop from the course code:

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

        # Process only the FIRST tool call — deliberately forcing one tool per iteration
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

1. **The Thought step:** `llm_with_tools.invoke(messages)` sends the full conversation plus the tool schemas to the model, and returns an `AIMessage`.

2. **The exit condition:** if `tool_calls` is empty, the model has decided it has enough information to answer. Return `.content` as the final answer.

3. **The simplification:** the code processes only the *first* tool call. Models can return several at once (parallel function calling is a real feature), but the loop is kept serial here for clarity — one tool per iteration is much easier to follow when you're learning.

4. **The Action step:** `tools_dict[name].invoke(args)` looks up the tool by name and runs it, producing an observation.

5. **The memory step:** append *both* the `AIMessage` (the model's decision to call a tool) **and** the `ToolMessage` (the result). Both are needed so that the next Thought step sees the complete history — what it decided and what came back.

6. **`tool_call_id`** links the result back to the specific call that produced it. This matters for correctness when there are multiple calls, and it's what makes traces readable.

7. **`@tool` gives you automatic tracing** — tool executions appear as their own spans in LangSmith without extra work.

**Mapping the loop diagram directly onto the code:**

| Diagram node | The code that implements it |
|--------------|------------------------------|
| Thought | `llm_with_tools.invoke` |
| Action | `tool_to_use.invoke` |
| Observation | `ToolMessage` |
| Final Answer | `if not tool_calls` |

This is a lean, stripped-down version of exactly what `create_agent` and the LangGraph ReAct agent do for you behind the scenes.

There's a related, simpler teaching variant in `project-ReAct-Algo/main.py` — it uses a text-length tool and a callback printer, but follows the identical `bind_tools` loop pattern, just with `while True` instead of a bounded `for`.

### Model Switch

Change exactly one string:

```python
llm = init_chat_model("openai:gpt-5", temperature=0)
# instead of ollama:qwen3:1.7b
```

Everything else stays identical, and LangSmith will now show `ChatOpenAI` instead of Ollama in the trace.

**An important caveat, and a genuinely useful production lesson:** easy switching is *not* the same as production readiness. Eden observed that **GPT-5.2** actually **underperformed** on this specific agent compared to both GPT-5 and Qwen — it started asking clarifying questions rather than just calling the tools, for the same prompt.

The takeaway: **evaluate before you swap.** A model being newer or better-marketed doesn't mean it's better *for your particular agent loop*. This is exactly why the "evals" discipline discussed later in the course exists.

### Test Yourself

**Q1.** What does `bind_tools` attach to each LLM request?
**Q2.** Why do you append both the AIMessage and the ToolMessage each iteration?
**Q3.** What happens if `tool_calls` comes back empty?
**Q4.** Why is "never calculate discounts yourself" in the system prompt?

<details>
<summary>Answers</summary>

1. The tool **schemas and descriptions**, so the model can emit structured tool calls.
2. So the next Thought step has both the **decision and the observation** in its history — this is the agent's memory, sometimes called the scratchpad.
3. Treat `.content` as the **final answer** and return it, ending the loop.
4. Defensive prompting against **hallucinated math** — especially necessary on smaller models, which will confidently invent arithmetic results.

</details>

---

## 6. [Layer 2] Raw Function Calling

### Manual JSON Schemas vs LangChain Tool Abstraction

File: `2_agent_loop_raw_function_calling.py`. The approach: copy Layer 1, then strip out every LangChain chat, tool, and message abstraction, replacing them with `import ollama`.

Without `@tool`, you have two options, neither pleasant:

- **Hand-write the vendor's JSON schemas yourself**, or
- Pass Python functions directly (an Ollama-specific convenience) using **Google-style docstrings** — a path that's poorly documented, and Eden actually has to dig into the Ollama source code to figure out how it works.

Here's what the manual schema looks like for Ollama:

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

Compare that to the two decorated functions from Layer 1. Same information, dramatically more typing — and every character of it is an opportunity for a typo that silently breaks tool selection.

**The tradeoff, laid out:**

| Concern | LangChain `@tool` | Writing raw schemas |
|---------|-------------------|---------------------|
| **Authoring the schema** | Generated automatically from type hints and docstrings | Manual, verbose, and error-prone |
| **Supporting multiple vendors** | One interface covers all | Anthropic, Ollama, and OpenAI all use different formats |
| **Documentation quality** | Reasonably consistent | Often incomplete — Ollama's formal schema is barely documented |
| **Switching models** | Change one string | Rewrite the SDK calls, the message roles, *and* the schemas |

That bottom-right cell is a big part of why LangChain won early adoption: it eliminated the **integration cost** of supporting many vendors.

Your tools stay as plain Python functions here, and you have to add `@traceable(run_type="tool")` manually, because you lost LangChain's automatic tool tracing along with everything else.

### Building a ReAct Agent Loop with the Raw Ollama SDK

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
        tool_name = tool_call.function.name          # attribute access, not dictionary access
        tool_args = tool_call.function.arguments

        observation = tools_dict[tool_name](**tool_args)  # direct function call, not .invoke

        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(observation)})
        # Note: the Ollama path may not carry the tool_call_id that a LangChain ToolMessage expects

    return None
```

#### How it works — and what LangChain was quietly doing for you

1. **Message roles become raw dictionaries.** You now write `{"role": "user", ...}` and `{"role": "assistant", ...}` instead of typed `HumanMessage` and `AIMessage` objects. Notice the naming difference too: LangChain normalizes across vendors that disagree about whether it's "human" or "user," "AI" or "assistant."

2. **Tracing is entirely manual.** You add `@traceable` to the chat function and to each tool. LangChain's chat models trace themselves out of the box.

3. **Tool dispatch changes shape** — `fn(**args)` here, versus `tool.invoke(args)` in Layer 1.

4. **The tool call object has a different shape** — `tool_call.function.name` (attribute access) here, versus `tool_call["name"]` (dictionary access) in LangChain.

5. **The ReAct loop structure itself is completely unchanged.** Only the wiring differs. That's the whole point of this layer: the *algorithm* is the same; LangChain was only ever handling the plumbing.

One practical detail: rename the outer trace span to **"Ollama Agent Loop"**, otherwise LangSmith will confusingly still say "LangChain Agent Loop."

### Recap (Layer 2)

Same shopping question, same three iterations, same final answer — all without LangChain.

What you paid for that independence: hand-written JSON schemas, SDK-specific message dictionaries, and manual tracing setup. And if you wanted to switch to Anthropic, you'd be writing a *different* schema dialect against a *different* client library. That cost, multiplied across every vendor you might want to support, is exactly the problem LangChain solves.

### Test Yourself

**Q1.** What does LangChain's `@tool` generate that Layer 2 makes you write by hand?
**Q2.** Why is multi-vendor support expensive without LangChain?
**Q3.** How do you invoke a tool in Layer 2 versus Layer 1?

<details>
<summary>Answers</summary>

1. The per-vendor **JSON tool schemas**, derived automatically from your function signatures and docstrings.
2. Because each vendor uses different **schema shapes, different message role names, and a different SDK** — so support is rewritten per vendor rather than shared.
3. Layer 2: `tools_dict[name](**args)` — a direct Python call. Layer 1: `tool.invoke(args)` — through LangChain's tool object.

</details>

---

## 7. [Layer 3] The ReAct Prompt — The Foundation of Function Calling

### What Are We Building? Function Calling (via ReAct Prompt)

The **ReAct prompt** is the prompt that launched LangChain agents. It lives on the LangChain Hub as `hwchase17/react` (authored by Harrison Chase, LangChain's CEO) and has millions of downloads.

Here's the historical context that makes this layer worth building: **before native function calling existed** (OpenAI introduced it around 2023), agents were **pure prompt engineering**. There was no `tools=` API. Instead, you wrote a prompt that instructed the model to emit text in a strict, predictable format — and then your own code parsed that text and executed the corresponding function.

In this layer you'll power the same e-commerce agent using only that technique — **no `tools=` API at all**.

**The canonical ReAct prompt format:**

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

Read that carefully — it's doing a lot of work. It describes the available tools, demonstrates the exact output format expected, explicitly says the cycle can repeat, and then ends mid-sentence with `Thought:` so the model has no choice but to continue in the demonstrated format.

**The agent scratchpad** is the accumulating history of Thought/Action/Observation that gets injected back into the prompt on each iteration. It's how the model "remembers" what it already tried.

**Two prompt-engineering techniques are quietly embedded here:** a few-shot-style demonstration of the output format, and chain-of-thought reasoning (that `Thought:` line forces the model to reason before acting). Both are covered in depth in the Prompt Engineering Theory section.

### Generating Dynamic Tool Descriptions in Python

File: `3_raw_react_prompt.py`. Delete the JSON schemas entirely. Instead, build plain-text descriptions using Python's `inspect` module:

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
        # __wrapped__ bypasses decorator wrappers (e.g. @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())
```

#### How it works

1. **`__wrapped__`** — this handles a subtle problem. If `@traceable` has wrapped your function, then `inspect.signature` would report the *wrapper's* signature, which includes tracer keyword arguments like `config=None`. Unwrapping gets you back to the real signature the model should see: `(product: str) -> float`.

2. **`inspect.signature` and `inspect.getdoc`** — these extract exactly the metadata the LLM needs to understand the tool. In Layer 1 this came free from `@tool`; in Layer 2 you hand-wrote it as JSON; here you're generating it as human-readable text.

3. Those generated descriptions get injected into the ReAct prompt f-string, alongside the same defensive STRICT RULES you wrote in Layer 1.

### Understanding the ReAct Prompt: Building Agents Without Function Calling

*(This lecture had no dedicated transcript.)*

The conceptual point: **the LLM is never told through any API that it's an agent.** There's no tool parameter, no special mode. Its "agency" comes entirely from two things you control — the prompt format, and your parser.

The mechanic is: the model continues generating text after `Thought:`, and you stop it before it gets far enough to invent its own `Observation`. That stopping is what the next section handles.

### Implementing Manual Tool Calling for LLMs

The critical piece here is an Ollama option: **`stop=["\nObservation"]`**.

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

**Why the stop token matters so much:** the model has been shown a format where `Observation:` follows `Action Input:`. If you don't stop it, it will happily keep generating and **hallucinate the Observation** — inventing a fake tool result and then reasoning from that fiction.

With the stop token in place, generation halts right after `Action Input:`. Then **you** execute the real tool and inject the real observation. That's the whole trick.

Note the structure: a **single user message** contains the instructions, the tool descriptions, the question, and the scratchpad, all concatenated. There's no separate system/user tool-calling protocol — it's all one blob of text.

### Agent Loop With ReAct Prompt

*(No transcript for this lecture; reconstructed from the course code.)*

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

        # Note: apply_discount has to cast price=float(price), because regex-parsed args are strings
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    return None
```

#### How it works

1. **The Final Answer branch:** a regex looks for `Final Answer:` in the output. If found, extract it and return — the loop is done.

2. **The tool branch:** two regexes pull out `Action:` (the tool name) and `Action Input:` (the arguments). The arguments then need messy parsing — splitting on commas, then stripping any `key=` prefix and any surrounding quotes. Finally, `tools[name](*args)` executes it.

3. **The scratchpad grows:** append the model's own output, then the *real* Observation, then a fresh `Thought:` — so that on the next iteration, the model picks up its chain of thought exactly where it left off, now with real data.

4. **The fragility is the lesson.** If the model drifts even slightly from the expected format — a missing colon, a different capitalization, a stray newline — the regex fails and the whole agent breaks. That fragility is precisely the historical motivation for native function calling.

Notice also the type problem in the comment: regex gives you strings, so `apply_discount` has to cast `price` to a float manually. With a JSON schema, the type would have been enforced for you.

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

### Cross-Layer Comparison (Same Agent, Three Implementations)

This table is the payoff of the whole "under the hood" block. Same agent, same question, same answer — three completely different mechanisms:

| Step | Layer 1 (LangChain) | Layer 2 (Raw FC) | Layer 3 (ReAct prompt) |
|------|---------------------|------------------|------------------------|
| **Reason** | Structured `tool_calls` | Structured `tool_calls` | Free text: Thought/Action |
| **Parse** | `ai_message.tool_calls[0]` | `.function.name` / `.arguments` | Regular expressions |
| **Execute** | `tool.invoke` | `fn(**args)` | `fn(*args)` |
| **Observe** | `ToolMessage` object | `{"role": "tool"}` dict | Appended scratchpad string |
| **Finish** | No tool_calls returned | No tool_calls returned | `Final Answer:` found in text |

### Test Yourself

**Q1.** What does the stop token `\nObservation` prevent?
**Q2.** What is the agent scratchpad in Layer 3?
**Q3.** Why is ReAct-prompt tool calling less reliable than native function calling?

<details>
<summary>Answers</summary>

1. It prevents the model from **hallucinating** its own Observation — inventing fake tool results and then reasoning from them.
2. A growing **string** containing the Thought/Action/Observation history, concatenated onto the prompt and re-sent on every iteration.
3. Because free-text output is **brittle to parse** — any format drift breaks the regex. Function calling returns schema-constrained structured calls instead, so there's nothing to break.

</details>

---

## 8. Function Calling (Theory)

*(This is the theory lecture that closes out the Layer 1–3 story. It also appears in the course outline as its own section.)*

### What Function Calling Actually Is

**Function calling** (also called **tool calling**) is a model's ability to emit a **structured function invocation** — a name plus arguments — instead of, or in addition to, free-form prose. Crucially, the call appears in a **dedicated slot in the model's response**, not buried in the `content` text where you'd have to scrape it out with a regex.

Some specifics worth pinning down:

- It's a **capability of certain models, not all of them**. These days it's effectively standard on any state-of-the-art model from OpenAI, Anthropic, or Google — so you can generally assume a flagship model supports it.
- It was **introduced by OpenAI in 2023**. Developers supply a list of **function definitions** (name, parameters, descriptions), and the model may respond with a JSON object specifying which function to call and with what arguments.
- Underneath, these models are **fine-tuned** specifically to detect when a function should be invoked based on the user's request, and to format their response as valid JSON adhering to the function's schema.
- Your application then parses that JSON, executes the real function, and feeds the result back into the conversation. That's precisely the agent loop you already built three times.

**The classic worked example.** The user asks: "What's the weather in Paris?" You've bound a `get_current_weather` tool. Instead of hallucinating a temperature, the model responds with something conceptually like:

```json
{
  "name": "get_current_weather",
  "arguments": { "location": "Paris", "unit": "celsius" }
}
```

Your app then runs the real weather API, appends the result as an observation, and lets the model produce the final natural-language answer.

### Why It Was Created

The **ReAct prompt** pioneered tool use, and it was genuinely clever — but it's **unreliable**. Models invent their own formats, drop closing tags, and change capitalization, and every one of those breaks your parser and crashes your program.

Function calling moves all that heavy lifting to the model vendor. The vendor does the reasoning; you get back machine-readable JSON that is far more deterministic and far easier to work with.

### The Two Capabilities It Unlocks

| Capability | What you get |
|---|---|
| **External tools** | Reliably connect LLMs to APIs, databases, search, calculators, anything |
| **Structured output** | Force the model's answer into JSON or Pydantic fields for downstream code — this is the exact same machinery behind `response_format` from Section 3 |

### Advantages

1. **Structured and reliable integration.** Because the model is fine-tuned to adhere strictly to your schema, you get far fewer formatting errors than with ReAct text. The output is machine-readable and much less prone to misinterpretation.
2. **Token efficient.** It skips all the verbose chain-of-thought narration in the visible output and returns primarily the call itself. Fewer tokens means lower cost and lower latency.
3. **It's the industry standard.** OpenAI (2023), Anthropic, Google, and every other major vendor ship function calling on their flagship models, and they've genuinely perfected the reliability. Almost nobody ships production agents on raw ReAct prompts anymore.

### The Tradeoff: Opaque Reasoning

This is the one real downside. When the model decides to call a tool, it typically does so **without exposing its chain of thought**. The reasoning stays internal to the LLM. You see the final function name and arguments, but not the justification for *why* it picked that function with those arguments.

That makes the decision effectively a **black box**, and it makes debugging and auditing harder — you can't easily answer "why did it choose that tool?"

In practice, the reliability win dominates, which is why the industry standardized on it anyway.

```
ReAct prompt era          Function calling era
─────────────────         ────────────────────
Text → regex → tool       Schema → JSON call → tool
Fragile, visible CoT      Reliable, opaque CoT
```

💡 **Extended Notes**

- **Function calling does *not* mean "the model runs code."** The model only *proposes* a call; your runtime executes it. Never treat model output as trusted code — this is a security principle as much as an architectural one, and it matters a lot in Section 26.
- **Structured output shares the same fine-tuning surface as tool calling.** Schema adherence is the underlying skill in both cases, which is why the "tool strategy" for structured output works at all.
- **Comparing the three approaches:**
  - **ReAct prompt:** full visibility into the model's thoughts; brittle parsing; high token cost.
  - **Native function calling:** reliable schema conformance; black-box decisions; lower token cost.
  - **Hybrid (LangGraph, covered later):** graph edges decide *when* tools run, while the model fills in arguments inside controlled nodes. You get back some of the visibility without giving up reliability.

### Test Yourself — Section 8

1. What does the model actually return when it wants to use a tool?
2. Name two problems function calling solves relative to a raw ReAct prompt.
3. What is the main *downside* of native function calling for debugging?
4. True/False: Function calling means the LLM executes your Python function inside the GPU.
5. Why is function calling considered more token-efficient than ReAct prompting?

**Answers**

1. A structured JSON (or vendor-native) tool call containing the function name plus arguments, delivered in a dedicated response field rather than in the prose content.
2. Reliable parseability and schema adherence; less brittle formatting; better determinism overall.
3. Opaque reasoning — you don't see the chain of thought that led to the call, making it hard to audit *why* a given tool and arguments were chosen.
4. **False.** The application (or agent runtime) executes the function after parsing the model's proposed call. The model never runs anything.
5. Because it skips the verbose thought/action narration and returns primarily just the call payload.

---

## Appendix A — Project Map for Sections 1–8

| Project folder | Which section it maps to |
|----------------|--------------------------|
| `course-code/project-hello-world/` | Section 2 — Hello World and LCEL |
| `course-code/project-search-agent/` | Section 3 — `create_agent` + Tavily + structured output |
| `course-code/project-ReAct-search-agent/` | Related search agent with `REACT_PROMPT` and schemas |
| `course-code/project-agents-under-the-hood/` | Sections 4–7 — Layers 1 through 3 |
| `course-code/project-ReAct-Algo/` | An extra `bind_tools` loop example with callbacks |

## Appendix B — Suggested Hands-On Checklist for Sections 1–8

- [ ] Run Hello World with OpenAI and then with Ollama; compare the two LangSmith traces side by side
- [ ] Build the stub search agent → swap in a Tavily custom tool → swap in official `TavilySearch` → add a Pydantic `response_format`
- [ ] Run all three `project-agents-under-the-hood` scripts against the same "laptop with gold discount" question
- [ ] Remove the defensive STRICT RULES in Layer 1 and observe exactly how it fails
- [ ] Remove `stop=["\nObservation"]` in Layer 3 and read the hallucinated observations it produces
- [ ] Sketch the ReAct diagram by hand and label Thought / Action / Observation / Final Answer in each of the three files

---

## 9. The GIST of RAG — Embeddings, Vector Databases & Retrieval

### Introduction to Retrieval Augmented Generation (RAG)

**The problem.** You have a large private document — Harry Potter, a 200-page financial contract, an internal company wiki. You want to ask questions whose answers live in *specific paragraphs*, like "how do you make this particular potion" or "what does this specific clause say."

The LLM was never trained on that private data, so it simply cannot answer from its weights alone. This isn't a model quality issue — the information genuinely isn't in there.

**The naive solution: stuff the whole document into the prompt.** Take the entire book, paste it into the prompt alongside the question, and send it. This works sometimes, but it fails for four distinct reasons:

1. **Hard token limits.** Even context windows of 1–2 million tokens are finite. A large enough corpus won't fit, full stop.
2. **The needle-in-a-haystack problem.** Research has shown that answer quality **degrades as the context grows**, even when everything technically fits. Relevant facts buried in the middle of a huge prompt get effectively lost. This is the counterintuitive one — bigger context windows do not mean you should fill them.
3. **Cost.** You pay per token, so a huge prompt on every single query is expensive.
4. **Latency.** Longer prompts simply take longer for the model to process.

**The RAG solution, at a high level.** Pre-process your corpus into **chunks**. At query time, **retrieve** only the most relevant chunks, **augment** the prompt with just those, and **generate** an answer grounded on that focused context.

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

This solves all four problems at once: you never approach the token limit, you avoid needle-in-a-haystack degradation because the context is small and targeted, you pay for far fewer tokens, and processing is faster. It also scales to very large documents and to *multiple* documents.

**The honest list of RAG's drawbacks:**

- **Chunking is genuinely hard.** Wrong boundaries destroy meaning — split a sentence in half and neither chunk makes sense.
- **Retrieval can miss the right chunk**, and then your answer is grounded in the wrong context.
- **Different content needs different strategies.** Chunking a prose document and chunking a code repository are not the same problem.
- **Dynamic user uploads need an online ingestion path** — you can't always pre-process everything in advance.

💡 **Extended Notes**

- **RAG = Retrieval + Augmentation + Generation.** Memorize the three letters as the three pipeline stages: retrieve the relevant chunks, augment the prompt with them, generate the answer.
- **"RAG is dead because of long context" is overstated.** Long context *complements* RAG rather than replacing it: you can retrieve richer snippets, you still save cost, you still get citable sources, and you still avoid positional bias inside the prompt. Eden's stance later in the course: long context windows *amplify* RAG; they do not replace it for production knowledge bases.

### Introduction to RAG Implementation

Here are the building blocks you'll actually assemble:

| Component | What it does |
|---|---|
| **Document loaders** | Give a uniform interface over PDF, Notion, Drive, WhatsApp, plain text, and thousands more — all producing `Document(page_content, metadata)` |
| **Text splitters** | Chunk long text while trying to preserve semantic coherence |
| **Embeddings** | Turn text into a dense vector, such that similar meaning produces nearby vectors |
| **Vector store (e.g. Pinecone)** | Persist those vectors and do fast nearest-neighbor search at query time |
| **Retriever** | A LangChain wrapper that turns a plain query string into the top-k matching `Document`s |

**Building intuition for embeddings.** An embedding model is best treated as a black box: text goes in, an array of floats comes out. A good model will place "I want an extra large coffee," "I'll have a tall coffee," and the Spanish "quiero pedir café extra grande" close together in that vector space — **even across languages** — because the underlying semantics align.

**Why you need a vector database.** It stores the embeddings and returns the closest neighbors to a query vector in milliseconds. Without one you'd have to embed everything yourself and scan linearly through every vector on every query, which is completely impractical at any real scale.

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

**The end-to-end mental model, using a book as the example:**

1. Split the book into thousands of chunks.
2. Embed each chunk into a vector.
3. Upsert those vectors into Pinecone (or Chroma, or any vector store).
4. When a question arrives, embed the question too.
5. Find the top-k most similar chunk vectors and pull back their original text.
6. Build a prompt containing the question plus those chunks, and send it to the LLM.

### Medium Analyzer — Boilerplate Project Setup

Project: `course-code/project-rag-gist/` (branch `project/rag-gist`).

**The setup flow:**

1. Clone the course repo and check out the starting commit for the RAG gist project.
2. Install dependencies via **uv** using `pyproject.toml` and `uv.lock`: run `uv lock`, then `uv sync`, which creates the `.venv` folder.
3. Point your IDE's Python interpreter at that `.venv` (run `which python3` after syncing to find the path).
4. Create `.env` with: `OPENAI_API_KEY`, the LangSmith variables (`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=RAG GIST`, `LANGSMITH_TRACING=true`), plus `INDEX_NAME` and `PINECONE_API_KEY`.
5. Create a Pinecone index — the example name is `medium-blogs-embeddings-index`. The settings that matter:
   - **Dense vectors** with **cosine similarity** (this is the default).
   - **Dimensions: 1536**, which is what `text-embedding-3-small` produces. Note this is deliberately higher than Pinecone's UI default of 512 — more capacity for semantic detail.
   - **Serverless capacity.** In production, choose your cloud provider and region carefully, since it affects latency, egress costs, and GDPR compliance.
6. The corpus is a single file, `mediumblog1.txt` — a Medium article about vector databases, pasted in as plain text.

💡 **Extended Notes — Pinecone index settings that bite people**

- **The dimension must exactly match your embedding model's output.** A mismatch causes either a silent failure or an outright rejection, and it's a very common first-time error.
- **Longer vectors hold more semantic detail but cost more** in both storage and query time. It's a real tradeoff, not a "bigger is better" setting.
- **Put the index in the same region as your application** to reduce both egress cost and latency.
- **The LangChain Pinecone integration expects the environment variable to be named exactly `PINECONE_API_KEY`** — same convention as `OPENAI_API_KEY` and `TAVILY_API_KEY`.

### Medium Analyzer — Class Review: TextLoader, TextSplitter, OpenAIEmbeddings, Pinecone

**Document loaders.** These are thin wrappers whose entire job is to normalize third-party data into a `Document` object. If you open the LangChain source (which the course encourages), you'll see `TextLoader` is genuinely simple: it opens a file path, sets `metadata["source"] = filepath`, and returns `[Document(...)]`. Something like `WhatsAppChatLoader` adds regex parsing to extract sender and date. But **every one of them exposes the same `.load()` interface** — PDF, Slack, YouTube, Notion, and thousands more. That uniformity is the entire point of the abstraction.

**CharacterTextSplitter.** Splits text on a separator (newlines by default) according to a `chunk_size` and a `chunk_overlap`.

- **`chunk_overlap`** is the important one to understand: it makes consecutive chunks share some text at their boundaries, so that meaning isn't destroyed by a cut landing mid-thought.
- **`length_function`** defaults to `len`, meaning it counts *characters*. You can swap in a token counter if you'd rather chunk by tokens.

**OpenAIEmbeddings.** A uniform LangChain interface over embedding providers — OpenAI, Cohere, Hugging Face, and others. Under the hood it's just an HTTP call to an `/embeddings` endpoint. A historical note worth knowing: `text-embedding-ada-002` was dramatically cheaper than OpenAI's earlier embedding models, and price genuinely matters when you're embedding an entire document warehouse.

**PineconeVectorStore.** Handles persistence, similarity search, and upserts. The free tier is sufficient for everything in this course.

### Medium Analyzer — Ingestion Implementation

The current course code (the updated path uses `UnstructuredLoader`; older videos used `TextLoader` — the pipeline is identical either way):

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

1. **Load** — produces a list of `Document` objects, each with `page_content` (the text) and `metadata.source` (where it came from).

2. **Split** — with `chunk_size=1000` and `overlap=0`, the Medium article becomes roughly 20 chunks. The exact count depends on where the separators fall.

3. **Embed and upsert** — `from_documents` does several things in one call: it iterates the chunks, calls the embedding API for each, and writes the resulting vectors *plus the original text plus the metadata* into Pinecone.

4. **Inspect it in the Pinecone UI.** Each row stores the **text**, the **source**, and the **vector**. This reveals something important: **embeddings are one-way**. You cannot reconstruct text from a vector. That's exactly why you must store the original text alongside it — otherwise retrieval would give you back meaningless numbers instead of readable context.

**The chunk-size heuristic:** small enough that several chunks comfortably fit in the model's context window, but large enough that a human reading a single chunk in isolation would still understand it. And remember: garbage in, garbage out still applies. Irrelevant context hurts both answer quality *and* cost, even if you're using a million-token model.

**An encoding gotcha:** some operating systems throw a `UnicodeDecodeError` here. The fix is to pass `encoding="utf-8"` or `autodetect_encoding=True` to the text loader.

**Why bother with LangChain's `from_documents` rather than doing it manually?** One interface across every vector store and embedding provider, plus built-in batching, async support, and rate-limit handling — all of which you'd otherwise have to write and maintain yourself.

### RECAP

Indexing is complete. The next half of the pipeline: embed the user's query, hit the vector DB for top-k matches, augment the prompt with them, and generate.

(Eden notes that the retrieval videos were re-filmed using Cursor and **uv**, while the ingestion code stayed as-is because it was already best practice.)

### Medium Analyzer — Naive Retrieval Implementation

First, the deliberately manual version — no LCEL, just plain Python:

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
    context = format_docs(docs)              # 2. stringify the results
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)          # 3. generate
    return response.content
```

Note `search_kwargs={"k": 3}` — that's asking the retriever for the 3 most similar chunks. And note the prompt says "based **only** on the following context," which is what grounds the answer and reduces hallucination.

**The demo query:** `"what is Pinecone in machine learning?"`

Eden runs this three ways to make the value of RAG concrete:

| Approach | What actually happened |
|---|---|
| **Raw LLM (GPT-3.5)** | **Hallucinated** — invented a "pinecone algorithm" for hyperparameter search that doesn't exist |
| **Raw LLM (newer GPT)** | Knew about Pinecone the vector database from its training data |
| **RAG + GPT-3.5** | **Correct** — a managed vector database for ML applications |

**The lesson:** RAG grounds weaker or older models *and* gives access to private data. Even strong models benefit whenever the corpus is proprietary or postdates their training cutoff.

**Limitations of writing it as a manual function:** no streaming, no async support, hard to compose with other pieces, and — importantly for debugging — **fragmented LangSmith traces**, because each component shows up as its own separate root run rather than as one connected pipeline.

### Medium Analyzer — 2 Step RAG (LCEL)

Now the same pipeline written declaratively with LCEL:

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

# invoke it
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

**How `assign` actually works** — this is the part worth slowing down on:

- **`RunnablePassthrough`** is the identity function on the input dictionary. It takes `{"question": ...}` and passes it through unchanged.
- **`.assign(context=subchain)`** *adds* a `context` key to that dictionary while **keeping** `question`. So the output is `{question, context}` — both keys, which is exactly what `prompt_template` needs, since the template has two placeholders.
- **The subchain** `itemgetter("question") | retriever | format_docs` pulls the question string out of the dict, retrieves matching documents, and formats them into a single string.
- **Plain Python functions like `format_docs` get auto-wrapped as `RunnableLambda`**, which is why you can drop a normal function straight into a pipe chain and it gains `.invoke`, `.stream`, and `.batch` for free.

Eden explicitly flags this LCEL block as the trickiest bit of syntactic sugar in the whole RAG section. It's worth re-reading until the `{question, context}` dictionary shape feels inevitable rather than magical.

**Naive versus LCEL, step by step:**

| Step | Manual function | LCEL chain |
|---|---|---|
| 1 Retrieve | `retriever.invoke(query)` | `itemgetter("question") \| retriever` |
| 2 Format | `format_docs(docs)` | `\| format_docs` |
| 3 Prompt | `prompt_template.format_messages(...)` | `\| prompt_template` |
| 4 LLM | `llm.invoke(messages)` | `\| llm` |
| 5 Parse | `response.content` | `\| StrOutputParser()` |
| **Tracing** | Scattered, separate root runs | **One** connected runnable sequence |

**Why LCEL wins for this particular pipeline:**

- **Declarative composition** with `|` — the code reads like the diagram.
- **Streaming, async, and batch support for free**, without writing any of it.
- **A single LangSmith trace** for the whole runnable sequence. Bottlenecks and intermediate inputs/outputs (assign → retriever → format → prompt → LLM) are all visible nested under one parent run, instead of scattered.

The answers themselves are identical either way. What differs is observability and maintainability. **Prefer LCEL whenever the pipeline is a fixed DAG of runnables.**

### LangChain RAG Documentation (a critique)

Eden's assessment of the official LangChain RAG documentation as of v1.0 — this is opinionated, and worth reading as production judgment rather than gospel:

1. **The semantic search tutorial** — solid on Documents, splitting, embedding, and storing. Light on building complete applications.

2. **"Build a RAG agent"** — this wraps retrieval as a *tool* on a ReAct agent. **Eden dislikes this as default production advice**, for four specific reasons:
   - The LLM may **skip retrieval** when it was actually needed, or call it unnecessarily when it wasn't.
   - The extra inference call to make that decision adds **latency and cost**.
   - It opens **jailbreak and off-topic risk** — a real concern for customer support use cases.
   - It's genuinely good in two situations: when you don't want greetings ("hi!") to hit the knowledge base, and when rewriting a query using chat history helps retrieval. But — and this is his point — **you can do both of those deterministically** without handing the LLM full agent freedom.

3. **The two-step chain in the docs** — this matches what the course built. Single inference call, less flexible. The docs' variant ("agent without tools plus middleware injection") is, in Eden's view, too opaque.

4. **Custom RAG with LangGraph** — this is the path Eden endorses, and it's what the course builds toward with corrective, self, and adaptive RAG later.

| Pattern | Control | Flexibility | Latency | Eden's production bias |
|---|---|---|---|---|
| **Always-retrieve 2-step LCEL** | High | Low | Lowest | Great for fixed Q&A |
| **Retrieval-as-tool agent** | Low | High | Higher (extra LLM call) | Risky as a default |
| **Hybrid LangGraph RAG** | Medium–High | Medium | Medium | Preferred in enterprises |

### Test Yourself — Section 9

1. List the four problems with stuffing an entire book into the prompt.
2. Why must vector DBs store the original text alongside the embeddings?
3. What does `k=3` mean on a retriever?
4. What does `RunnablePassthrough.assign(context=...)` add to the input dict?
5. When would Eden prefer a fixed 2-step RAG over a retrieval-tool agent?

**Answers**

1. Token limits, the needle-in-a-haystack quality drop, cost, and latency.
2. Because embeddings are **not invertible** — you cannot reconstruct text from a vector. You need the stored text to show to the LLM and to the user.
3. Return the **top 3** most similar chunks from the vector store.
4. It keeps the original keys (like `question`) and **adds** a computed `context` key, producing `{question, context}`.
5. For domain Q&A or support scenarios where retrieval should **always** run, and where giving the agent freedom to skip or misuse it is a liability rather than a benefit.

---

---

# ⏸ REWRITE IN PROGRESS — Sections 10–28 still to come

**What's complete in this file:** Sections 1 through 9, plus Appendices A and B, fully rewritten in plain language with every detail from the original preserved.

**What's still pending from the original notebook** (original lines 1936–5245):

| Section | Topic |
|---|---|
| 10 | Building a Documentation Assistant (Tavily crawl, chunking, batch indexing, retrieval agent, Streamlit UI, production concerns, RAG architecture comparison) |
| 11 | Prompt Engineering Theory (GIST of LLMs, prompt anatomy, zero/few-shot, chain-of-thought, ReAct prompting, quick tips, context engineering, system prompts) |
| 12 | LLM Applications In Production (landscape, privacy & data retention, generative UI/CopilotKit, open vs managed LLMs) |
| 13 | Introduction To LangGraph (why LangGraph, graphs, flow engineering, core components, hands-on ReAct AgentExecutor) |
| 14 | Reflection Agent |
| 15 | Reflexion Agent (Actor/Revisor, ToolNode, graph construction, tracing) |
| 16 | Agentic RAG (GraphState, retrieve node, relevance grading, web search node, generation, Corrective / Self / Adaptive RAG) |
| 17 | Introduction to Model Context Protocol (MCP) |
| 18 | Using a Pre-built Server (mcpdoc) with AI Clients |
| 19 | Building MCP Servers and Clients with LangChain |
| 20 | Useful Tools (LangChain Hub, TextSplitting Playground, LangChain vs LlamaIndex) |
| 21 | Deep Agents (taxonomy, to-do lists, sub-agents, context flow, file systems) |
| 22 | Deep Agents Skills (3 layers, progressive disclosure, skills.py internals) |
| 23 | LangChain Glossary (ChatModels, Messages, splitters, Document, token strategies, memory) |
| 24 | Industry Insights — Assaf Elovic |
| 25 | Industry Insights — Roy Miara |
| 26 | Agent Security |
| 27 | The Dark Side of "Vibe Coding" |
| 28 | Bonus + Appendices |

Say **"continue the rewrite"** and I'll pick up at Section 10 and keep going in the same style.

---
