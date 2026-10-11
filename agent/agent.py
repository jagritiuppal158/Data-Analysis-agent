from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from .data import clean, column_kind
from .memory import MemoryStore
from .tools import Context, ToolError, make_tools
from dotenv import load_dotenv

load_dotenv() 

SYSTEM_PROMPT = """You are a careful data-analysis agent. You answer questions about ONE tabular dataset by calling tools.

HARD RULES
1. Every number in your answer must come from a tool result. Never estimate, recall, or compute figures yourself.
2. If you do not know the schema well enough, call describe_dataset first. Use column names EXACTLY as listed.
   In filters, wrap names containing spaces in backticks, e.g. `Unit Price` > 5.
3. Pick the tool that matches the question: breakdowns/totals/top-N -> aggregate_data; relationships ->
   correlation_analysis; "is A different from B" -> compare_groups; weird values -> detect_outliers;
   anything over time -> time_trend; "show/plot" -> plot_chart.
4. If a tool returns an ERROR, read it, fix your arguments and try again (do not give up after one error).
5. In the final answer: lead with the direct answer, give the key numbers, state any filters/assumptions you
   applied, and note statistical caveats (e.g. correlation != causation, small samples). Be concise.
   If the data cannot answer the question, say so instead of guessing.

MEMORY (persistent across sessions)
- "Relevant memories" below are facts the user taught you earlier. APPLY them: use a saved filter via the
  `filter_name` argument, respect definitions/corrections, and follow preferences. Mention briefly when you used one.
- When the user defines a term, names a reusable filter, states a preference ("I prefer weekly summaries"),
  or corrects you, call save_memory with a short distilled fact. For filters you MUST give a pandas `expression`.
  Saving under an existing key replaces the old fact (that is how corrections/updates work).
- Do NOT save raw data, analysis results, or one-off questions. If asked what you remember, call list_memories;
  if asked to forget something, call forget_memory.
- After saving, confirm in one short sentence what you remembered (and what it replaced, if anything).

DATASET SCHEMA
{schema}

RELEVANT MEMORIES
{memories}
"""


def get_llm(provider: str | None = None, model: str | None = None, temperature: float = 0.0,
            api_key: str | None = None):

    provider = (provider or os.getenv("LLM_PROVIDER") or "").lower()
    if not provider:
        provider = ("groq" if (api_key or os.getenv("GROQ_API_KEY")) else
                    "gemini" if os.getenv("GOOGLE_API_KEY") else "ollama")
    model = model or os.getenv("LLM_MODEL")
    if provider == "groq":
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("No Groq API key found. Paste it in the sidebar, put GROQ_API_KEY=... in a .env file, "
                             "or export it in the SAME terminal you run streamlit from. Free key: https://console.groq.com/keys")
        from langchain_groq import ChatGroq
        return ChatGroq(model=model or "openai/gpt-oss-120b", temperature=temperature, api_key=key)
    if provider == "gemini":
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("No Google API key found. Paste it in the sidebar or set GOOGLE_API_KEY. "
                             "Free key: https://aistudio.google.com/apikey")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-flash", temperature=temperature, google_api_key=key)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model or "llama3.1:8b", temperature=temperature)
    raise ValueError(f"Unknown provider '{provider}' (use groq, gemini or ollama)")


@dataclass
class AgentResult:
    answer: str
    trace: list = field(default_factory=list)
    figures: list = field(default_factory=list)
    memories_used: list = field(default_factory=list)


class DataAgent:
    def __init__(self, df, memory: MemoryStore | None = None, llm=None, outdir="outputs", max_steps: int = 8):
        self.memory = memory or MemoryStore("memory.db")
        self.ctx = Context(df=df, memory=self.memory, outdir=outdir)
        self.tools = make_tools(self.ctx)
        self.tool_map = {t.name: t for t in self.tools}
        self.llm = (llm or get_llm()).bind_tools(self.tools)
        self.max_steps = max_steps
        self.history: list = []  # only (user, final answer) text pairs, not tool traffic

    def _schema_text(self) -> str:
        df = self.ctx.df
        lines = [f"{len(df)} rows x {df.shape[1]} columns"]
        for c in df.columns:
            lines.append(f"- {c} ({df[c].dtype}, {column_kind(df[c])})")
        return "\n".join(lines)

    def _invoke(self, messages):
        for attempt in range(3):
            try:
                return self.llm.invoke(messages)
            except Exception as e:  # noqa: BLE001  (free tiers rate-limit; back off and retry)
                if attempt < 2 and any(s in str(e).lower() for s in ("rate", "429", "overloaded", "timeout")):
                    time.sleep(4 * (attempt + 1))
                    continue
                raise

    def ask(self, question: str) -> AgentResult:
        mems = self.memory.retrieve(question, self.ctx.scope)
        system = SYSTEM_PROMPT.format(schema=self._schema_text(),
                                      memories=MemoryStore.format_for_prompt(mems))
        messages = [SystemMessage(system), *self.history, HumanMessage(question)]
        trace, fig_start = [], len(self.ctx.figures)

        for _ in range(self.max_steps):
            ai: AIMessage = self._invoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                answer = ai.content if isinstance(ai.content, str) else "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in ai.content)
                self.history += [HumanMessage(question), AIMessage(answer)]
                self.history = self.history[-12:]
                return AgentResult(answer, trace, self.ctx.figures[fig_start:], mems)

            for call in ai.tool_calls:
                name, args = call["name"], call.get("args", {})
                t0 = time.time()
                error = False
                try:
                    if name not in self.tool_map:
                        raise ToolError(f"Unknown tool '{name}'. Available: {list(self.tool_map)}")
                    result = self.tool_map[name].invoke(args)
                    payload = json.dumps(clean(result), default=str)
                except ToolError as e:
                    error, payload = True, f"ERROR: {e}"
                except Exception as e:  # noqa: BLE001
                    error, payload = True, f"ERROR ({type(e).__name__}): {e}"
                if len(payload) > 7000:
                    payload = payload[:7000] + '... [truncated]'
                trace.append({"tool": name, "args": args, "result": payload, "error": error,
                              "seconds": round(time.time() - t0, 3)})
                messages.append(ToolMessage(content=payload, tool_call_id=call["id"], name=name))

        return AgentResult("I could not finish within the step limit. Try a more specific question.",
                           trace, self.ctx.figures[fig_start:], mems)
