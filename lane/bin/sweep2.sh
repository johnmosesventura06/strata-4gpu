#!/bin/bash
# Round 2: stack winners + deep-context probes on the 256K 3-tier base.
cd /home/ai-host/strata-4gpu/strata4 || exit 1
RES=/home/ai-host/strata-sweep2-results.txt
LOG=strata-our-udq4k.log
: > "$RES"
PORT_N=0

kill_engine() {
  for p in $(pgrep -f "serve/server.py.*813[0-9]"); do kill "$p" 2>/dev/null; done
  for p in $(pgrep -x strata); do kill "$p" 2>/dev/null; done
  sleep 10
}

run_one() {
  local name="$1"; shift
  python3 - "$name" "$@" << 'PYEOF'
import json, sys
name = sys.argv[1]; kv = sys.argv[2:]
c = json.load(open("/home/ai-host/strata-4gpu/configs/t3_256k.json"))
a = c["args"]
for i in range(0, len(kv), 2):
    flag, val = kv[i], kv[i+1]
    if flag in a:
        a[a.index(flag) + 1] = val
    else:
        a += [flag, val]
c["model_name"] = "sweep2-" + name
json.dump(c, open(f"/home/ai-host/strata-4gpu/configs/sweep2_{name}.json", "w"), indent=1)
PYEOF
  PORT_N=$((PORT_N + 1)); local port=$((8130 + PORT_N))
  nohup python3 serve/server.py --engine strata --config "../configs/sweep2_${name}.json" --port "$port" > "/tmp/sweep2-$name-serve.log" 2>&1 &
  local sp=$!
  for i in $(seq 1 36); do grep -q "ready:" "/tmp/sweep2-$name-serve.log" 2>/dev/null && break; sleep 5; done
  if ! grep -q "ready:" "/tmp/sweep2-$name-serve.log" 2>/dev/null; then
    echo "$name BOOT_FAILED" >> "$RES"; kill "$sp" 2>/dev/null; return 1
  fi
  python3 - "$port" "$name" << 'PYEOF'
import json, time, urllib.request, sys
port, name = sys.argv[1], sys.argv[2]
def chat(msg, mx):
    body = json.dumps({"messages":[{"role":"user","content":msg}], "max_tokens":mx, "temperature":0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type":"application/json"})
    t0=time.time(); json.loads(urllib.request.urlopen(req, timeout=1800).read()); return time.time()-t0
chat("warm", 8); chat("warm2", 8)
chat("Write a 2400-token dense technical narrative about paged memory pools, pinning, eviction and why each matters. Continue deeply.", 2400)
unit = open("/home/ai-host/strata-4gpu/strata/serve/server.py").read()
big = (unit * (16384*4//len(unit)+1))[:16384*4]
chat(big + "\n\nSummarize what this file does in two sentences.", 40)
# DEEP: ~220K tokens, then decode twice (2nd turn = warm conversation cache, deep position)
line = "Each spring the crew rotates turbine blades, logs vibration spectra, and calibrates the yaw drives. "
hay = (line * 45) * 235
fact = "SECRET BEACON KESTREL-9 LAUNCH CODE 4-7-1-5-9 STORED IN HANGAR ELEVEN. "
deep = hay + fact + (line * 45) * 10 + "\n\nWhat is the KESTREL-9 launch code? One line."
chat(deep, 300)
chat("Expand on the hangar detail in 400 tokens of dense prose.", 400)
PYEOF
  { echo "== $name =="; grep -E "prompt tokens.*generated" "$LOG" | tail -4; } >> "$RES"
  kill "$sp" 2>/dev/null
  kill_engine
}

for spec in "combo --head-split 0.40 --feed-max 80 --prefill 24576" \
            "combo_h60 --head-split 0.60 --feed-max 80 --prefill 24576" \
            "combo_w8k --head-split 0.40 --feed-max 80 --prefill 24576 --mtp-window 8192"; do
  run_one $spec
done
echo SWEEP2_DONE >> "$RES"
