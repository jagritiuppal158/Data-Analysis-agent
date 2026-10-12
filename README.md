# Data Analysis Agent
using LangChain + persistent memory
An agent that takes **any tabular file** (CSV / Excel / JSON ) and answers natural-language questions
by calling analysis tools on the real data. Includes a persistent, inspectable **memory** that changes later answers.

> Every number in an answer comes from pandas / scipy / scikit-learn running on your dataframe, never from the LLM.
> The UI shows the full tool-call trace (tool, arguments, raw result) for every answer.

## start

```bash
pip install -r requirements.txt
```

Choose ONE free LLM backend:

| Option | Setup | Notes |
|---|---|---|
| **Groq** (recommended) | free key from https://console.groq.com/keys → `export GROQ_API_KEY=...` | fast, good tool calling (default model `llama-3.3-70b-versatile`) |
| **Gemini** | free key from https://aistudio.google.com/apikey → `export GOOGLE_API_KEY=...` | default `gemini-2.5-flash` |
| **Ollama** (fully local, no key) | install https://ollama.com, then `ollama pull llama3.1:8b` | slower; small models make more tool mistakes |

Model names change over time; override with `LLM_MODEL=...` or the sidebar box.

```bash
streamlit run app.py                 # UI with tool trace + memory panel
python cli.py data/orders.csv        # terminal version
```

## Architecture

```
question ─► retrieve relevant memories ─► prompt = rules + dataset schema + memories
        ▲                                         │
        │                                         ▼
   final answer ◄── LLM decides: answer, or call a tool ──► tool runs on the real DataFrame
                              ▲                                      │
                              └──────── result (or ERROR) ◄──────────┘   (max 8 steps)
```

* `agent/tools.py` – the tools (below). Column names are **arguments**, never hard-coded, so new datasets just work.
* `agent/agent.py` – explicit tool-calling loop. LangChain provides the chat-model wrappers (Groq/Gemini/Ollama),
  `@tool` schemas and `bind_tools`; I wrote the loop myself so errors are fed back to the model and every call is traced.
* `agent/memory.py` – SQLite memory store. `agent/data.py` – loader (delimiter sniffing, date parsing).

## The tools (7 analysis + 3 memory)

| Tool | Kind of analysis | Under the hood |
|---|---|---|
| `describe_dataset` | schema & data-quality profile | dtypes, missing %, uniques, summary stats, duplicates |
| `aggregate_data` | slice / group / rank | filter → groupby → sum/mean/count/… → sort → top-N |
| `correlation_analysis` | numeric relationships | Pearson / Spearman with p-values, target ranking |
| `compare_groups` | hypothesis testing | Welch t + Mann-Whitney + Cohen's d; ANOVA + Kruskal + η²; chi-square + Cramér's V (auto-chosen) |
| `detect_outliers` | anomalies | IQR, z-score, or multivariate Isolation Forest |
| `time_trend` | time series | resample, rolling mean, growth %, linear-trend significance, naive forecast |
| `plot_chart` | visual | bar / line / scatter / hist / box → PNG |
| `save_memory`, `list_memories`, `forget_memory` | memory | see below |

**Safety:** filters are pandas `query` strings screened against a blocklist (`__`, `import`, `eval`, …); no arbitrary code
execution. This is a pragmatic guard, not a full sandbox – don't expose it to untrusted users as-is.

## Memory design

Goal: remember *useful distilled facts*, not chat logs or data.

| Kind | Example | How it changes future behaviour |
|---|---|---|
| `filter` | "active customers = status ≠ cancelled" → `status != 'cancelled'` | validated against the data on save; applied via `filter_name` in any tool |
| `definition` | "revenue is net of refunds" | injected into the prompt when relevant |
| `correction` | "`amt` is in cents, not dollars" | injected into the prompt when relevant |
| `preference` | `time_granularity = weekly`, `default_top_n = 5` | **applied in code**: `time_trend` and `aggregate_data` use them when the user doesn't specify |

* **What is worth keeping?** The LLM decides (prompt rules), but a code gatekeeper rejects anything >300 chars, number dumps,
  unknown kinds, and filters that don't run on the dataset.
* **Conflicts / stale facts:** saving an existing key (fuzzy-matched, so `active_customer` ≈ `active_customers`) *supersedes* the old
  row, which is kept as history; the tool tells the user what was replaced.
* **Scoping:** preferences are global; everything else is scoped to a dataset *fingerprint* (hash of column names), so a
  definition for one dataset is never applied to another – but a new export with the same columns reuses it.
* **Retrieval:** BM25-style scoring over key + content; preferences always included, others only when relevant to the question.
* **User control:** sidebar panel lists memories with delete / clear-all; CLI has `/memory` and `/forget <id>`; or just ask the agent.
* **Manual proof:** teach a filter (e.g. 'active customers means status is not cancelled'),
  restart the app, ask 'total revenue from active customers?', and the tool trace shows
  filter_name being applied.

## What I built vs. what I used

* **Used:** pandas, numpy, scipy, scikit-learn, matplotlib (the analysis); LangChain (`langchain-core`, `langchain-groq`,
  `-google-genai`, `-ollama`) for model wrappers and tool schemas; Streamlit for the UI; SQLite; a free hosted/local LLM.
* **My own work:** the tool design and argument schemas, the filter/safety layer, the agent loop with error feedback and tracing,
  the entire memory system (gatekeeper, conflict handling, scoping, retrieval, preference-driven tool behaviour), and the UI.
* **AI assistance:** for tools guidance and general support to optimise code and debug errors..
## Demo script 

1. Load `data/orders.csv` → "Give me an overview of this dataset."
2. "Which region has the highest total revenue?" → show trace → `aggregate_data`
3. "Is revenue different across regions?" → `compare_groups` (ANOVA)
4. "Any unusual revenue values?" → `detect_outliers`
5. "How is revenue trending?" → `time_trend`; "Plot monthly revenue" → `plot_chart`
6. **Memory:** "Active customers means status is not cancelled. Remember that." → restart the app →
   "What's the total revenue from active customers?" → trace shows `filter_name="active_customers"`.
7. "I prefer weekly summaries." → restart → "How is revenue trending?" → trace shows `freq: W (from saved preference)`.
8. "Actually active customers should only be status == active." → memory panel shows the replaced definition.
9. Switch to `data/students.csv` (different schema, spaces in names) and ask "what predicts exam score?".

## Limitations / honest notes

* Free-tier LLMs rate-limit and small local models sometimes pick the wrong tool; the loop retries and feeds errors back, but
  quality depends on the model.
* Forecasts are naive linear extrapolation; correlation and group comparisons are exploratory, not causal.
* Memory retrieval is lexical (no embeddings) to stay dependency-free; swapping in embeddings is a small change in `retrieve()`.
