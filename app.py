import json
from pathlib import Path
import streamlit as st
from agent.agent import DataAgent, get_llm
from agent.data import load_table
from agent.memory import MemoryStore

st.set_page_config(page_title="Data Analysis Agent", layout="wide")
DB = "memory.db"
memory = MemoryStore(DB)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Dataset")
    up = st.file_uploader("Upload CSV / Excel / JSON", type=["csv", "tsv", "txt", "xlsx", "json"])
    samples = sorted(p.name for p in Path("data").glob("*.csv"))
    sample = st.selectbox("Use a sample", ["-"] + samples)

    st.header("Model")
    provider = st.selectbox("Provider", ["auto", "groq", "gemini", "ollama"],
                            help="auto = Groq if GROQ_API_KEY set, else Gemini if GOOGLE_API_KEY set, else local Ollama")
    api_key = st.text_input("API key (optional)", "", type="password",
                            help="Paste your free Groq or Gemini key here. Used for this session only, never saved.")
    model = st.text_input("Model (optional)", "", help="e.g. llama-3.3-70b-versatile · gemini-2.5-flash · llama3.1:8b")

df, name = None, None
if up is not None:
    df, name = load_table(up, up.name), up.name
elif sample != "-":
    df, name = load_table(Path("data") / sample), sample

st.title("Data Analysis Agent")
if df is None:
    st.info("Pick or upload a dataset begin.")
    st.stop()

key = (name, provider, model, api_key)
if st.session_state.get("agent_key") != key:
    try:
        llm = get_llm(None if provider == "auto" else provider, model or None, api_key=api_key or None)
        st.session_state.agent = DataAgent(df, memory, llm)
        st.session_state.agent_key, st.session_state.chat = key, []
    except Exception as e:  # missing key / package / ollama not running
        st.error(f"Could not start the LLM: {e}")
        st.stop()
agent: DataAgent = st.session_state.agent

with st.expander(f"Preview: {name} ({len(df)} rows, {df.shape[1]} columns)"):
    st.dataframe(df.head(50), width="stretch")

# ------------------------------------------------------- memory panel (control)
with st.sidebar:
    st.header("Memory")
    mems = memory.list(agent.ctx.scope)
    if not mems:
        st.caption('Nothing saved yet. For example: "active customers means status is not cancelled".')
    for m in mems:
        c1, c2 = st.columns([8, 1])
        c1.markdown(f"**{m['kind']}**: `{m['key']}`  \n{m['content']}" + (f"  \n`{m['expression']}`" if m["expression"] else ""))
        if c2.button("Delete", key=f"del{m['id']}", help="Delete this memory"):
            memory.forget(m["id"])
            st.rerun()
    if mems and st.button("Forget everything for this dataset"):
        memory.clear(agent.ctx.scope)
        st.rerun()

def show_trace(trace):
    if not trace:
        return
    with st.expander(f"Tool calls ({len(trace)})"):
        for i, t in enumerate(trace, 1):
            status = " Error" if t["error"] else ""
            st.markdown(f"Tool: `{t['tool']}` ({t['seconds']}s){status}")
            st.code(json.dumps(t["args"], indent=2, default=str), language="json")
            st.code(t["result"][:1500], language="json")

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.caption("You" if m["role"] == "user" else "Agent")
        st.markdown(m["text"])
        for f in m.get("figures", []):
            st.image(f)
        show_trace(m.get("trace"))

if q := st.chat_input("Ask about your data…"):
    st.session_state.chat.append({"role": "user", "text": q})
    with st.chat_message("user"):
        st.caption("You")
        st.markdown(q)
    with st.chat_message("assistant"):
        st.caption("Agent")
        with st.spinner("Analyzing"):
            try:
                res = agent.ask(q)
            except Exception as e:  # noqa: BLE001
                st.error(f"LLM call failed: {e}")
                st.stop()
        st.markdown(res.answer)
        for f in res.figures:
            st.image(f)
        if res.memories_used:
            st.caption("Memories used: " + ", ".join(f"`{m['key']}`" for m in res.memories_used))
        show_trace(res.trace)
    st.session_state.chat.append({"role": "assistant", "text": res.answer, "figures": res.figures, "trace": res.trace})
    if any(t["tool"] in ("save_memory", "forget_memory") for t in res.trace):
        st.rerun()
