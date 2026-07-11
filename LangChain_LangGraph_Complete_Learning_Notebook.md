# LangChain & LangGraph — Complete Self-Contained Learning Notebook

> Your full substitute for watching Eden Marco's Udemy course  
> **LangChain — Agentic AI Engineering with LangChain & LangGraph** (3rd edition, LangChain v1.0+)

---

## How to use this notebook

1. **Read in order** — Sections 1 → 28 mirror the course progression. Later sections assume earlier mental models (LCEL → agents → ReAct layers → RAG → LangGraph → MCP → Deep Agents).
2. **Run the code** — Snippets are from the real course repos. Extracted copies live under `course-code/` in this folder. Prefer checking out the matching GitHub branch when you want a runnable project.
3. **💡 Extended Notes** — Material I added beyond the transcript (clarifications, production judgment, missing-lecture fill-ins). Everything else tracks the course.
4. **How it works** — Under non-trivial snippets: line-by-line breakdown of what the code does.
5. **Test Yourself** — End of each major section. Answers are in `<details>` blocks or directly below the questions.
6. **Diagrams** — ASCII reproductions of course visuals (ReAct loops, RAG pipelines, LangGraph topologies, MCP, Deep Agents).

### Source repositories

| Repo / branch | Used for |
|---|---|
| [emarco177/langchain-course](https://github.com/emarco177/langchain-course) `project/hello-world` | Hello World LCEL chain |
| same → `project/search-agent` | Modern `create_agent` search agent |
| same → `project/agents-under-the-hood` | ReAct layers 1–3 |
| same → `project/rag-gist` | Medium Analyzer / RAG gist |
| [emarco177/documentation-helper](https://github.com/emarco177/documentation-helper) | Docs assistant (crawl → RAG agent → Streamlit) |
| [emarco177/langgraph-course](https://github.com/emarco177/langgraph-course) `project/ReAct-agent`, `reflection-agent`, `reflexion-agent`, `agentic-rag` | LangGraph projects |
| Local `course-code/` | Snapshot of the above for offline reading |

### Prerequisites (from the course)

- Python 3.10+, Git, virtual envs (`uv` preferred over conda)
- Comfort debugging code; **no ML background required**
- An LLM: OpenAI / Anthropic / Gemini **or** local via Ollama
- Optional APIs as you hit those sections: Tavily, Pinecone, LangSmith

---

## Table of Contents

- [1. Introduction](#1-introduction)
  - [Course Introduction](#course-introduction)
  - [Course Objectives](#course-objectives)
  - [Course Structure + How to Get the Best Out of It](#course-structure-how-to-get-the-best-out-of-it)
  - [Course's Community](#courses-community)
  - [Course Resources](#course-resources)
- [2. The GIST of LangChain — Hello World Chain](#2-the-gist-of-langchain-hello-world-chain)
  - [What is LangChain? (Under ~6 Minutes)](#what-is-langchain-under-6-minutes)
  - [What Are We Building? LangChain Hello World Chain](#what-are-we-building-langchain-hello-world-chain)
  - [Project Setup](#project-setup)
  - [LangChain Fundamentals: Prompt Templates, ChatModels, and Chains](#langchain-fundamentals-prompt-templates-chatmodels-and-chains)
  - [Building a LangChain Chain to Summarize Text](#building-a-langchain-chain-to-summarize-text)
  - [Debugging and Tracing Our LangChain Chain](#debugging-and-tracing-our-langchain-chain)
  - [Using Local Open-Weights Models with LangChain and Ollama](#using-local-open-weights-models-with-langchain-and-ollama)
  - [Integrating LangSmith for LangChain Application Tracing](#integrating-langsmith-for-langchain-application-tracing)
  - [Semantic Versioning in LangChain](#semantic-versioning-in-langchain)
- [3. THE GIST of AI Agents](#3-the-gist-of-ai-agents)
  - [What Are AI Agents? A High-Level Overview](#what-are-ai-agents-a-high-level-overview)
  - [What Are We Building? AI Job Search Agent](#what-are-we-building-ai-job-search-agent)
  - [The Evolution of LangChain ReAct Agents](#the-evolution-of-langchain-react-agents)
  - [Setting Up the Environment for a LangChain Search Agent](#setting-up-the-environment-for-a-langchain-search-agent)
  - [Creating Your First LangChain Agent: Tools and LLMs](#creating-your-first-langchain-agent-tools-and-llms)
  - [From Query to Answer: How a LangChain Agent Thinks](#from-query-to-answer-how-a-langchain-agent-thinks)
  - [Integrating Real-World Search with Tavily and LangChain Tools](#integrating-real-world-search-with-tavily-and-langchain-tools)
  - [Structured Output with LangChain Agents Using Pydantic](#structured-output-with-langchain-agents-using-pydantic)
  - [[THEORY] Predictable Agent Responses with LangChain Structured Output](#theory-predictable-agent-responses-with-langchain-structured-output)
- [4. Agents Under The Hood (1/4) — Core Architecture](#4-agents-under-the-hood-14-core-architecture)
  - [Introduction to The Core Architecture of AI Agents](#introduction-to-the-core-architecture-of-ai-agents)
  - [What Are We Building? An E-Commerce Agent](#what-are-we-building-an-e-commerce-agent)
  - [[Theory] The Gist of ReAct](#theory-the-gist-of-react)
  - [Setup](#setup)
- [5. [Layer 1] The ReAct Loop](#5-layer-1-the-react-loop)
  - [Writing Tools](#writing-tools)
  - [Tool Binding and Defensive Prompting](#tool-binding-and-defensive-prompting)
  - [Understanding the ReAct Agent Loop in LangChain](#understanding-the-react-agent-loop-in-langchain)
  - [Model Switch](#model-switch)
- [6. [Layer 2] Raw Function Calling](#6-layer-2-raw-function-calling)
  - [Manual JSON Schemas vs LangChain Tool Abstraction](#manual-json-schemas-vs-langchain-tool-abstraction)
  - [Building a ReAct Agent Loop with the Raw Ollama SDK](#building-a-react-agent-loop-with-the-raw-ollama-sdk)
  - [Recap (Layer 2)](#recap-layer-2)
- [7. [Layer 3] The ReAct Prompt — Foundation of Function Calling](#7-layer-3-the-react-prompt-foundation-of-function-calling)
  - [What Are We Building? Function Calling (via ReAct Prompt)](#what-are-we-building-function-calling-via-react-prompt)
  - [Generating Dynamic Tool Descriptions in Python](#generating-dynamic-tool-descriptions-in-python)
  - [Understanding the ReAct Prompt: Building Agents Without Function Calling](#understanding-the-react-prompt-building-agents-without-function-calling)
  - [Implementing Manual Tool Calling for LLMs](#implementing-manual-tool-calling-for-llms)
  - [Agent Loop With ReAct Prompt](#agent-loop-with-react-prompt)
  - [Cross-Layer Comparison (Same Agent)](#cross-layer-comparison-same-agent)
- [Bridge: [Theory] Understanding Function Calling for LLMs](#bridge-theory-understanding-function-calling-for-llms)
- [8. Function Calling](#8-function-calling)
  - [Intro](#intro)
  - [\[Theory\] Understanding Function Calling for LLMs](#theory-understanding-function-calling-for-llms)
- [9. The GIST of RAG — Embeddings, Vector Databases & Retrieval](#9-the-gist-of-rag-embeddings-vector-databases-retrieval)
  - [Introduction to Retrieval Augmented Generation (RAG)](#introduction-to-retrieval-augmented-generation-rag)
  - [Introduction to RAG Implementation](#introduction-to-rag-implementation)
  - [Medium Analyzer — Boilerplate Project Setup](#medium-analyzer-boilerplate-project-setup)
  - [Medium Analyzer — Class Review: TextLoader, TextSplitter, OpenAIEmbeddings, Pinecone](#medium-analyzer-class-review-textloader-textsplitter-openaiembeddings-pinecone)
  - [Medium Analyzer — Ingestion Implementation](#medium-analyzer-ingestion-implementation)
  - [RECAP](#recap)
  - [Medium Analyzer — Naive Retrieval Implementation](#medium-analyzer-naive-retrieval-implementation)
  - [Medium Analyzer — 2 Step RAG (LCEL)](#medium-analyzer-2-step-rag-lcel)
  - [LangChain RAG Documentation (critique)](#langchain-rag-documentation-critique)
- [10. Building a Documentation Assistant](#10-building-a-documentation-assistant)
  - [What are we building? A lightweight Cursor-like docs feature (RAG)](#what-are-we-building-a-lightweight-cursor-like-docs-feature-rag)
  - [Quick Note: Pipenv vs uv](#quick-note-pipenv-vs-uv)
  - [Environment Setup](#environment-setup)
  - [Ingestion Pipeline Intro](#ingestion-pipeline-intro)
  - [Imports (ingestion)](#imports-ingestion)
  - [Tavily Crawling](#tavily-crawling)
  - [\[Optional\] TavilyMap + TavilyExtract (high customizability)](#optional-tavilymap-tavilyextract-high-customizability)
  - [\[Optional\] Crawling Deep Dive](#optional-crawling-deep-dive)
  - [Recap (ingestion)](#recap-ingestion)
  - [Chunking (Text Splitting)](#chunking-text-splitting)
  - [Batch Indexing](#batch-indexing)
  - [Retrieval Agent Implementation](#retrieval-agent-implementation)
  - [Run, Debug, Trace RAG Agent](#run-debug-trace-rag-agent)
  - [Frontend with Streamlit (UI)](#frontend-with-streamlit-ui)
  - [Documentation Helper In Production](#documentation-helper-in-production)
  - [RAG Architecture](#rag-architecture)
  - [Docs Helper Architecture (ASCII)](#docs-helper-architecture-ascii)
- [11. Prompt Engineering Theory](#11-prompt-engineering-theory)
  - [The GIST of LLMs](#the-gist-of-llms)
  - [What is a Prompt? Composition of a Formal Prompt](#what-is-a-prompt-composition-of-a-formal-prompt)
  - [Zero Shot Prompting](#zero-shot-prompting)
  - [Few Shot Prompting](#few-shot-prompting)
  - [Chain of Thought Prompting](#chain-of-thought-prompting)
  - [ReAct Prompting](#react-prompting)
  - [Prompt Engineering Quick Tips](#prompt-engineering-quick-tips)
  - [Context Engineering](#context-engineering)
  - [Context Engineering a System Prompt](#context-engineering-a-system-prompt)
- [12. LLM Applications In Production](#12-llm-applications-in-production)
  - [LLM Applications in Production](#llm-applications-in-production)
  - [LLM Application Development Landscape](#llm-application-development-landscape)
  - [Privacy & Data Retention](#privacy-data-retention)
  - [Generative UI/UX featuring CopilotKit](#generative-uiux-featuring-copilotkit)
  - [Official LangChain Academy Courses](#official-langchain-academy-courses)
  - [Open Source LLMs vs Managed Providers (Deepseek)](#open-source-llms-vs-managed-providers-deepseek)
  - [Confidence in AI Results (Assaf Elovic & Harrison Chase)](#confidence-in-ai-results-assaf-elovic-harrison-chase)
  - [\[NEW\] AI FOMO is the New Normal](#new-ai-fomo-is-the-new-normal)
  - [Finished course? What’s next!](#finished-course-whats-next)
- [13. Introduction To LangGraph](#13-introduction-to-langgraph)
  - [What is LangGraph?](#what-is-langgraph)
  - [Why LangGraph? LangGraph VS LangChain](#why-langgraph-langgraph-vs-langchain)
  - [What are Graphs?](#what-are-graphs)
  - [LangGraph & Flow Engineering](#langgraph-flow-engineering)
  - [LangGraph Core Components](#langgraph-core-components)
  - [[Hands On] Implementing ReAct AgentExecutor with LangGraph](#hands-on-implementing-react-agentexecutor-with-langgraph)
  - [Quick Note: poetry vs uv](#quick-note-poetry-vs-uv)
  - [[Hands On] Get Started: Setting Up Your ReAct Agent Project Environment](#hands-on-get-started-setting-up-your-react-agent-project-environment)
  - [[Hands On] Coding the Agent's Brain: Implementing the ReAct Runnable](#hands-on-coding-the-agents-brain-implementing-the-react-runnable)
  - [[Hands On] Building Blocks: Defining Your Agent's Nodes in LangGraph](#hands-on-building-blocks-defining-your-agents-nodes-in-langgraph)
  - [[Hands On] Bringing Your ReAct Agent to Life: Connecting Nodes into a Graph](#hands-on-bringing-your-react-agent-to-life-connecting-nodes-into-a-graph)
  - [[Hands On] Running Our LangGraph React Agent with Function Calling](#hands-on-running-our-langgraph-react-agent-with-function-calling)
  - [[IMPORTANT] Building Modern LLM Agents: From History to LangGraph v1.0](#important-building-modern-llm-agents-from-history-to-langgraph-v10)
- [14. Reflection Agent](#14-reflection-agent)
  - [What are we building? A Reflection Agent](#what-are-we-building-a-reflection-agent)
  - [Project Setup](#project-setup)
  - [Creating the Reflector Chain and the Tweet Reviosr Chain](#creating-the-reflector-chain-and-the-tweet-reviosr-chain)
  - [Defining our LangGraph Graph](#defining-our-langgraph-graph)
  - [LangSmith Tracing](#langsmith-tracing)
- [15. Reflexion Agent](#15-reflexion-agent)
  - [What are we building? A Reflexion Agent](#what-are-we-building-a-reflexion-agent)
  - [Project Setup](#project-setup)
  - [Section Resources](#section-resources)
  - [Actor Agent V2](#actor-agent-v2)
  - [Revisor Agent](#revisor-agent)
  - [ToolNode - Executing Tools](#toolnode---executing-tools)
  - [Building Our LangGraph Graph](#building-our-langgraph-graph)
  - [Tracing Our Graph](#tracing-our-graph)
- [16. Agentic RAG](#16-agentic-rag)
  - [What are Building In this Section- Agentic RAG Architecture](#what-are-building-in-this-section--agentic-rag-architecture)
  - [Improving RAG Quality with the Corrective RAG Flow](#improving-rag-quality-with-the-corrective-rag-flow)
  - [Boilerplate Setup for an Agentic RAG Agent with LangGraph](#boilerplate-setup-for-an-agentic-rag-agent-with-langgraph)
  - [Code Structure](#code-structure)
  - [LangChain Vector Store Ingestion Pipeline (WebLoader, ChromaDB)](#langchain-vector-store-ingestion-pipeline-webloader-chromadb)
  - [Managing Information Flow in LangGraph: The GraphState](#managing-information-flow-in-langgraph-the-graphstate)
  - [Fetching Context for LLMs: The LangGraph Retrieve Node](#fetching-context-for-llms-the-langgraph-retrieve-node)
  - [Building a Relevance Filter for RAG using LangChain's Structured Output](#building-a-relevance-filter-for-rag-using-langchains-structured-output)
  - [Implementing a Web Search Node in LangGraph using Tavily API](#implementing-a-web-search-node-in-langgraph-using-tavily-api)
  - [Creating the LLM Generation Chain and Node for LangGraph](#creating-the-llm-generation-chain-and-node-for-langgraph)
  - [Building and Running the Complete LangGraph Agent](#building-and-running-the-complete-langgraph-agent)
  - [Self RAG- Intro](#self-rag--intro)
  - [Self RAG- Implementation](#self-rag--implementation)
  - [Adaptive RAG](#adaptive-rag)
  - [Complete Agentic RAG — Chains Reference](#complete-agentic-rag-chains-reference)
- [Cross-Section Comparison (Senior Cheat Sheet)](#cross-section-comparison-senior-cheat-sheet)
- [17. Introduction to Model Context Protocol (MCP)](#17-introduction-to-model-context-protocol-mcp)
  - [Why MCP (Model Context Protocol)](#why-mcp-model-context-protocol)
  - [How LLMs Really Use Tools: Understanding Tool Calling](#how-llms-really-use-tools-understanding-tool-calling)
  - [MCP Architecture](#mcp-architecture)
  - [The GIST of the Protocol with Tool Calling](#the-gist-of-the-protocol-with-tool-calling)
  - [MCP Servers](#mcp-servers)
- [18. Using a Pre-built Server (mcpdoc) with AI Clients (Cursor & Claude)](#18-using-a-pre-built-server-mcpdoc-with-ai-clients-cursor-claude)
  - [What are we building? MCP Doc](#what-are-we-building-mcp-doc)
  - [MCP Inspector](#mcp-inspector)
  - [LLM.txt](#llmtxt)
  - [mcpdoc](#mcpdoc)
- [19. Building MCP Servers and Clients with LangChain](#19-building-mcp-servers-and-clients-with-langchain)
  - [Intro](#intro)
  - [Boilerplate](#boilerplate)
  - [Servers](#servers)
  - [What are we MCBuilding?](#what-are-we-mcbuilding)
  - [Simple MCP Server / Client Scaffold](#simple-mcp-server-client-scaffold)
  - [Bridging the Gap: The LangChain MCP Adapter Explained](#bridging-the-gap-the-langchain-mcp-adapter-explained)
  - [Imports / Client / Tracing](#imports-client-tracing)
- [20. Useful tools when developing LLM Applications](#20-useful-tools-when-developing-llm-applications)
  - [Stop Writing Deprecated Code: LangChain's Official MCP Server](#stop-writing-deprecated-code-langchains-official-mcp-server)
  - [LangChain Hub — Downloads prompt from the community](#langchain-hub-downloads-prompt-from-the-community)
  - [TextSplitting Playground](#textsplitting-playground)
  - [LangChain vs LlamaIndex](#langchain-vs-llamaindex)
- [21. Deep Agents](#21-deep-agents)
  - [Introduction to Deep Agents Section](#introduction-to-deep-agents-section)
  - [Taxonomy: Shallow Agents, Deep Agents, Coding Agents](#taxonomy-shallow-agents-deep-agents-coding-agents)
  - [Dynamic To-Do Lists](#dynamic-to-do-lists)
  - [Sub Agents and Hierarchical Delegation](#sub-agents-and-hierarchical-delegation)
  - [Subagents Context Flow](#subagents-context-flow)
  - [Deep Agents File Systems](#deep-agents-file-systems)
- [22. Deep Agents Skills](#22-deep-agents-skills)
  - [The 3 Layers of AI Agent Skills: From Usage to Source Code](#the-3-layers-of-ai-agent-skills-from-usage-to-source-code)
  - [Level 1: Using Agent Skills in the Deep Agents CLI](#level-1-using-agent-skills-in-the-deep-agents-cli)
  - [Layer 2: Tracing AI Agent Skills with LangSmith](#layer-2-tracing-ai-agent-skills-with-langsmith)
  - [RECAP — Skill Middleware](#recap-skill-middleware)
  - [Layer 3: Inside skills.py — Mechanics of Progressive Disclosure](#layer-3-inside-skillspy-mechanics-of-progressive-disclosure)
- [23. LangChain Glossary](#23-langchain-glossary)
  - [ChatModels](#chatmodels)
  - [Messages](#messages)
  - [RecursiveCharacterTextSplitter](#recursivecharactertextsplitter)
  - [Document](#document)
  - [Token Limitation Strategies](#token-limitation-strategies)
  - [Memory Intro — Coreference Resolution](#memory-intro-coreference-resolution)
  - [Memory Deepdive (LangGraph)](#memory-deepdive-langgraph)
- [24. Industry Insights: Building Production Agents with Assaf Elovic](#24-industry-insights-building-production-agents-with-assaf-elovic)
  - [The Core Architecture of Production-Grade AI](#the-core-architecture-of-production-grade-ai)
  - [How to Make Users Trust Your AI Agents](#how-to-make-users-trust-your-ai-agents)
  - [Tutorial: Building a Lean AI Feedback Loop](#tutorial-building-a-lean-ai-feedback-loop)
- [25. Industry Insights: Building Production Agents with Roy Miara](#25-industry-insights-building-production-agents-with-roy-miara)
  - [Intro](#intro)
  - [AI Agents in Cybersecurity CTF Competitions](#ai-agents-in-cybersecurity-ctf-competitions)
  - [Harness Engineering](#harness-engineering)
  - [Managing Variance and Hallucinations in Production Agents](#managing-variance-and-hallucinations-in-production-agents)
- [26. Agent Security](#26-agent-security)
  - [What is LLM App Sec?](#what-is-llm-app-sec)
- [27. The Dark Side of "Vibe Coding": Vulnerabilities in AI-Generated Apps](#27-the-dark-side-of-vibe-coding-vulnerabilities-in-ai-generated-apps)
  - [Introduction](#introduction)
  - [AI Coding Rarely Writes SQL Injections or XSS Bugs](#ai-coding-rarely-writes-sql-injections-or-xss-bugs)
  - [AI Agents Struggle with Role-Based Access Control](#ai-agents-struggle-with-role-based-access-control)
  - [AI Coding Agents Struggle with Business Logic and SSRF](#ai-coding-agents-struggle-with-business-logic-and-ssrf)
  - [AI Coding Agents Struggle with Rate Limiting and CSRF](#ai-coding-agents-struggle-with-rate-limiting-and-csrf)
  - [Prompt Engineering Won't Fix Insecure AI Code](#prompt-engineering-wont-fix-insecure-ai-code)
- [28. Bonus](#28-bonus)
  - [One-week practice plan](#one-week-practice-plan)
  - [Architectural rhymes worth remembering](#architectural-rhymes-worth-remembering)
  - [MCP (recall)](#mcp-recall)
  - [Deep agent hierarchy (recall)](#deep-agent-hierarchy-recall)
  - [Skills progressive disclosure (recall)](#skills-progressive-disclosure-recall)

---

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

---

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

---


## Master Appendix — Hands-On Project Checklist

Work through these in order as you finish each section of the notebook:

- [ ] **Hello World** — `course-code/project-hello-world/` with OpenAI and Ollama; open LangSmith traces
- [ ] **Search Agent** — `course-code/project-search-agent/` with Tavily + Pydantic `response_format`
- [ ] **Agents Under The Hood** — run all three scripts in `project-agents-under-the-hood/` on the same question
- [ ] **RAG Gist** — ingest `mediumblog1.txt`, compare raw LLM vs naive retrieval vs LCEL 2-step
- [ ] **Docs Helper** — crawl → chunk → index → `create_agent` retrieve tool → Streamlit UI
- [ ] **LangGraph ReAct** — `langgraph-project-ReAct-agent` nodes + conditional edges
- [ ] **Reflection** — generate ↔ reflect tweet loop
- [ ] **Reflexion** — draft → tools → revise with structured critique
- [ ] **Agentic RAG** — corrective + self + adaptive graders on Chroma
- [ ] **MCP** — inspect a prebuilt server, then wire a tiny FastMCP server into a LangChain agent
- [ ] **Deep Agents** — run CLI with a skill folder; inspect progressive disclosure in LangSmith

---

*Notebook assembled from course transcripts + [emarco177/langchain-course](https://github.com/emarco177/langchain-course) and related repos. 💡 Extended Notes are supplementary teaching additions, not Eden's voice.*
