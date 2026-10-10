#!/bin/bash
# Profile play v2: real corpus through serve -> counter trace -> --no-base profile -> A/B.
cd /home/ai-host/strata-4gpu/strata4 || exit 1
RES=/home/ai-host/strata-profile2.txt
LOG=strata-our-udq4k.log
TRACE=/tmp/routing.bin
: > "$RES"

for p in $(ps -eo pid,cmd | grep "[s]erve/server.py" | awk "{print \$1}"); do kill "$p" 2>/dev/null || true; done
for p in $(ps -eo pid,cmd | grep "[b]uild/strata --serve" | awk "{print \$1}"); do kill "$p" 2>/dev/null || true; done
for i in $(seq 1 40); do
  BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>200" | wc -l)
  [ "$BUSY" = "0" ] && break
  sleep 5
done

# 1. boot with dump-routing wired
python3 - << 'PYEOF'
import json
c = json.load(open("../configs/nw_nway075.json"))
c["args"] += ["--dump-routing", "/tmp/routing.bin"]
c["model_name"] = "prof-trace"
json.dump(c, open("../configs/prof_trace.json", "w"), indent=1)
PYEOF
setsid nohup python3 serve/server.py --engine strata --config ../configs/prof_trace.json --port 8178 \
    > /tmp/prof-trace-serve.log 2>&1 < /dev/null &
SP=$!
for i in $(seq 1 75); do
  curl -s --max-time 2 "http://127.0.0.1:8178/health" 2>/dev/null | grep -q '"loaded": true' && break
  sleep 4
done
curl -s --max-time 2 "http://127.0.0.1:8178/health" >/dev/null || { echo "TRACE_BOOT_FAILED" >> "$RES"; exit 1; }
echo "trace server up" >> "$RES"

# 2. corpus through the real HTTP path (chat template included)
python3 << 'PYEOF'
import json, urllib.request, pathlib
msgs = []
for f in ["serve/server.py", "src/program/generate.cpp", "src/core/expert_source.cpp", "src/core/verify.cpp",
          "src/program/generate.cpp"]:
    t = pathlib.Path(f).read_text(errors="ignore")
    msgs.append(t[:14000] + "\n\nWhat are the three biggest functions in this file? One line each.")
msgs.append("Gusto ko i-tune yung serving lane natin. Check the VRAM numbers, i-compare sa bench, at sabihin mo yung tradeoff bago mo gumawa. No hedging, plain language, push back if I'm wrong.")
msgs.append("Explain why prefill stays flat when expert tiers grow, in dense technical prose, 400 tokens.")
msgs.append("Write a rate-limited job scheduler in Python with priorities and fair-share. Full code only.")
for m in msgs:
    body = json.dumps({"messages": [{"role": "user", "content": m}], "max_tokens": 120, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8178/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=1800).read()
    except Exception as e:
        print("req err", e)
print("corpus done")
PYEOF

# 3. graceful stop -> the engine's tail writes the trace
kill "$SP" 2>/dev/null
sleep 8
for p in $(ps -eo pid,cmd | grep "[b]uild/strata --serve" | awk "{print \$1}"); do kill "$p" 2>/dev/null || true; done
for i in $(seq 1 40); do
  BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>200" | wc -l)
  [ "$BUSY" = "0" ] && break
  sleep 5
done
ls -la "$TRACE" | awk "{print \"trace bytes:\", \$5}" >> "$RES"

# 4. rebuild the profile from our traffic only
python3 tools/make_profile.py "$TRACE" --no-base --out ../Strata-data/profile-john.bin >> "$RES" 2>&1

# 5. A/B boot on the new profile + same bench as nway075's boots
python3 - << 'PYEOF'
import json
c = json.load(open("../configs/nw_nway075.json"))
c["args"][c["args"].index("--expert-profile") + 1] = "../Strata-data/profile-john.bin"
c["model_name"] = "prof-john"
json.dump(c, open("../configs/prof_john.json", "w"), indent=1)
PYEOF
setsid nohup python3 serve/server.py --engine strata --config ../configs/prof_john.json --port 8179 \
    > /tmp/prof-john-serve.log 2>&1 < /dev/null &
for i in $(seq 1 75); do
  curl -s --max-time 2 "http://127.0.0.1:8179/health" 2>/dev/null | grep -q '"loaded": true' && break
  sleep 4
done
python3 - << 'PYEOF'
import json, time, urllib.request
def chat(msg, mx, port=8179):
    body = json.dumps({"messages":[{"role":"user","content":msg}], "max_tokens":mx, "temperature":0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type":"application/json"})
    urllib.request.urlopen(req, timeout=1800).read()
unit = open("/home/ai-host/strata-4gpu/strata/serve/server.py").read()
big = (unit * (16384*4//len(unit)+1))[:16384*4]
chat(big + "\n\nSummarize what this file does in two sentences.", 40)   # COLD first prompt
chat("Write a 2400-token dense technical narrative about paged memory pools, pinning, eviction and why each matters. Continue deeply.", 2400)
PYEOF
{ echo "== profile-john A/B =="; grep -E "prompt tokens.*generated" "$LOG" | tail -2; grep -iE "done:.*hit" /tmp/prof-john-serve.log | head -2; } >> "$RES"
kill $(ps -eo pid,cmd | grep "[s]erve/server.py.*8179" | awk "{print \$1}") 2>/dev/null
echo PROFILE2_DONE >> "$RES"
