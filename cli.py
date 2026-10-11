import argparse
import json
from agent.agent import DataAgent, get_llm
from agent.data import load_table
from agent.memory import MemoryStore

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--provider", choices=["groq", "gemini", "ollama"])
ap.add_argument("--model")
ap.add_argument("--api-key")
ap.add_argument("--db", default="memory.db")
a = ap.parse_args()

mem = MemoryStore(a.db)
agent = DataAgent(load_table(a.file), mem, get_llm(a.provider, a.model, api_key=a.api_key))
print(f"Loaded {a.file}: {agent.ctx.df.shape}. Ask questions. /memory  /forget <id>  /quit")
while True:
    try:
        q = input("\nyou> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        continue
    if q == "/quit":
        break
    if q == "/memory":
        for m in mem.list(agent.ctx.scope):
            print(f"  #{m['id']} [{m['kind']}] {m['key']}: {m['content']} {m['expression'] or ''}")
        continue
    if q.startswith("/forget"):
        print("forgotten" if mem.forget(int(q.split()[1])) else "no such id")
        continue
    res = agent.ask(q)
    for t in res.trace:
        print(f" {t['tool']}({json.dumps(t['args'], default=str)[:150]}) {'ERROR' if t['error'] else 'ok'}")
    print("\nagent>", res.answer)
    for f in res.figures:
        print(" chart saved:", f)
