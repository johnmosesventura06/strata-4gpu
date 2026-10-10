#!/bin/bash
# Profile play: routing trace from a typical corpus -> make_profile.py -> A/B vs shipped.
set -e
cd /home/ai-host/strata-4gpu/strata4
RES=/home/ai-host/strata-profile.txt
LOG=gen-trace.log
: > "$RES"

# stop the live server cleanly
for p in $(ps -eo pid,cmd | grep "[s]erve/server.py" | awk "{print \$1}"); do kill "$p" 2>/dev/null || true; done
for p in $(ps -eo pid,cmd | grep "[b]uild/strata --serve" | awk "{print \$1}"); do kill "$p" 2>/dev/null || true; done
for i in $(seq 1 40); do
  BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>200" | wc -l)
  [ "$BUSY" = "0" ] && break
  sleep 5
done

# 1. tokenize a typical corpus -> token ids
python3 << 'PYEOF'
import json, sys
sys.path.insert(0, "tools")
import strata_tokenizer as ST
from pathlib import Path
tp = Path("../Strata-data/packs/ud-q4_k_xl/tokenizer")
vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
merges = (tp / "merges.txt").read_text(encoding="utf-8").split("\n")
types = json.loads((tp / "token_type.json").read_text())
tk = ST.Tokenizer(tokens, merges, types)
corpus = []
for f in ["serve/server.py", "src/program/generate.cpp", "src/core/expert_source.cpp",
          "/home/ai-host/notes", "/home/ai-host/strata-4gpu/PROJECT.md"]:
    p = Path(f)
    try:
        if p.is_dir():
            for q in sorted(p.glob("*.md"))[:4]:
                corpus.append(q.read_text(errors="ignore"))
        elif p.exists():
            corpus.append(p.read_text(errors="ignore"))
    except OSError:
        pass
corpus.append("Gusto ko i-config yung dual 5060 ti natin, ilagay mo sa notes. Check the VRAM and give me the tradeoff table, no hedging, plain language. Tapos i-bench natin. 'what's the tradeoff' comes before 'how'. Push back if you disagree with reasons.")
text = "\n\n".join(corpus)[:120000]
ids = tk.encode(text, parse_special=False)
Path("/tmp/trace-tokens.txt").write_text(" ".join(str(i) for i in ids))
print("tokens:", len(ids))
PYEOF

# 2. one-shot generate with --dump-routing (reuse the live config's args minus serve)
python3 - << 'PYEOF'
import json
c = json.load(open("../configs/nw_nway075.json"))
args = [a for a in c["args"]]
i = args.index("--tokens-file") if "--tokens-file" in args else -1
open("/tmp/trace-args.txt", "w").write(" ".join(args))
PYEOF
ARGS=$(cat /tmp/trace-args.txt)
./build/strata $ARGS --tokens-file /tmp/trace-tokens.txt --max-new 8 \
    --dump-routing /tmp/routing.trace > "$LOG" 2>&1 || { echo TRACE_RUN_FAILED >> "$RES"; tail -4 "$LOG" >> "$RES"; exit 1; }
ls -la /tmp/routing.trace | awk '{print "trace bytes:", $5}' >> "$RES"

# 3. rebuild the profile from the trace
python3 tools/make_profile.py /tmp/routing.trace --out ../Strata-data/profile-john.bin >> "$RES" 2>&1
ls -la ../Strata-data/profile-john.bin | awk '{print "profile bytes:", $5}' >> "$RES"

# 4. A/B boot on the new profile (everything else identical to nw_nway075)
python3 - << 'PYEOF'
import json
c = json.load(open("../configs/nw_nway075.json"))
a = c["args"]
a[a.index("--expert-profile") + 1] = "../Strata-data/profile-john.bin"
c["model_name"] = "prof-john"
json.dump(c, open("../configs/prof_john.json", "w"), indent=1)
PYEOF
setsid nohup python3 serve/server.py --engine strata --config ../configs/prof_john.json --port 8176 \
    > /tmp/prof-serve.log 2>&1 < /dev/null &
for i in $(seq 1 75); do
  curl -s --max-time 2 "http://127.0.0.1:8176/health" 2>/dev/null | grep -q '"loaded": true' && break
  sleep 4
done
curl -s --max-time 2 "http://127.0.0.1:8176/health" | head -c 60; echo >> "$RES"

# 5. bench: FIRST-PROMPT cold prefill (fresh cache, this is where the profile bites) + decode
python3 - << 'PYEOF'
import json, time, urllib.request
def chat(msg, mx, port=8176):
    body = json.dumps({"messages":[{"role":"user","content":msg}], "max_tokens":mx, "temperature":0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type":"application/json"})
    t0=time.time(); d=json.loads(urllib.request.urlopen(req, timeout=1800).read()); return time.time()-t0
# COLD first prompt: 24K tokens nobody has touched since boot
unit = open("serve/server.py").read()
cold = (unit * (24576*4//len(unit)+1))[:24576*4]
chat(cold + "\n\nOne line: what does this file serve?", 24)
chat("Write a 2400-token dense technical narrative about paged memory pools, pinning, eviction and why each matters. Continue deeply.", 2400)
PYEOF
echo "== profile-john boot ==" >> "$RES"
grep -E "prompt tokens.*generated" "$LOG" >> "$RES" 2>/dev/null || true
grep -E "prompt tokens.*generated" strata-our-udq4k.log | tail -2 >> "$RES"
grep -iE "expert cache .* hit" /tmp/prof-serve.log | head -2 >> "$RES"
echo PROFILE_PLAY_DONE >> "$RES"
