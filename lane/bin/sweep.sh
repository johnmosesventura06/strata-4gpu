#!/bin/bash
# Decode/prefill knob sweep on the 256K 3-tier base config. One flag per boot.
cd /home/ai-host/strata-4gpu/strata4 || exit 1
RES=/home/ai-host/strata-sweep-results.txt
PORT_N=0
LOG=strata-our-udq4k.log
: > "$RES"

kill_engine() {
  for p in $(pgrep -f "serve/server.py.*812[0-9]"); do kill "$p" 2>/dev/null; done
  for p in $(pgrep -x strata); do kill "$p" 2>/dev/null; done
  sleep 10   # the pinned expert arena needs seconds to release
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
c["model_name"] = "sweep-" + name
json.dump(c, open(f"/home/ai-host/strata-4gpu/configs/sweep_{name}.json", "w"), indent=1)
PYEOF
  PORT_N=$((PORT_N + 1)); local port=$((8120 + PORT_N))
  nohup python3 serve/server.py --engine strata --config "../configs/sweep_${name}.json" --port "$port" > "/tmp/sweep-$name-serve.log" 2>&1 &
  local sp=$!
  # wait for ready (max 180s)
  for i in $(seq 1 36); do grep -q "ready:" "/tmp/sweep-$name-serve.log" 2>/dev/null && break; sleep 5; done
  if ! grep -q "ready:" "/tmp/sweep-$name-serve.log" 2>/dev/null; then
    echo "$name BOOT_FAILED" >> "$RES"; kill "$sp" 2>/dev/null; return 1
  fi
  python3 - "$port" << 'PYEOF'
import json, time, urllib.request, sys
port = sys.argv[1]
def chat(msg, mx):
    body = json.dumps({"messages":[{"role":"user","content":msg}], "max_tokens":mx, "temperature":0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type":"application/json"})
    t0=time.time(); d=json.loads(urllib.request.urlopen(req, timeout=1800).read()); return time.time()-t0
chat("warm", 8); chat("warm2", 8)
chat("Write a 2400-token dense technical narrative about paged memory pools, pinning, eviction and why each matters. Continue deeply.", 2400)
unit = open("/home/ai-host/strata-4gpu/strata/serve/server.py").read()
big = (unit * (16384*4//len(unit)+1))[:16384*4]
chat(big + "\n\nSummarize what this file does in two sentences.", 40)
PYEOF
  # harvest the last engine-measured lines
  { echo "== $name =="; grep -E "prompt tokens.*generated|generated in" "$LOG" | tail -2; } >> "$RES"
  kill "$sp" 2>/dev/null
  kill_engine
}

for spec in "base" "mtpw8192 --mtp-window 8192" "hsplit040 --head-split 0.40" \
            "minp070 --spec-min-p 0.70" "feed80 --feed-max 80" "prech24576 --prefill 24576" \
            "pcie012 --pcie-frac 0.12"; do
  run_one $spec
done
echo SWEEP_DONE >> "$RES"
