# LangChain & LangGraph — Course Learning Pack

Personal self-contained notes for Eden Marco's *LangChain — Agentic AI Engineering with LangChain & LangGraph* (Udemy, LangChain v1.0+).

## Start here

Open **[LangChain_LangGraph_Complete_Learning_Notebook.md](./LangChain_LangGraph_Complete_Learning_Notebook.md)** — full substitute for watching the course (sections 1–28).

## Contents

| Path | Purpose |
|------|---------|
| `LangChain_LangGraph_Complete_Learning_Notebook.md` | Master notebook |
| `notebook-parts/` | Same content split into 4 smaller files |
| `LangChain_Udemy_Course_Outline.md` | Course outline |
| `LangChain_Transcripts.txt` | Lecture transcripts used to build the notebook |
| `course-code/` | Extracted Python from course GitHub project branches |

## Upstream course repos

- [emarco177/langchain-course](https://github.com/emarco177/langchain-course)
- [emarco177/langgraph-course](https://github.com/emarco177/langgraph-course)
- [emarco177/documentation-helper](https://github.com/emarco177/documentation-helper)

## Quick run examples

```bash
# Hello World
cd course-code/project-hello-world
uv sync
# set OPENAI_API_KEY (or switch to ChatOllama in main.py)
uv run python main.py

# Agents under the hood (needs Ollama + model)
cd ../project-agents-under-the-hood
uv sync
uv run python 1_agent_loop_langchain_tool_calling.py
```

## Note

This repo is for personal study. Course videos and instructor materials remain the property of the course author / Udemy.
