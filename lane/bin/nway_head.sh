#!/bin/bash
# N-way head-split A/B on the new binary: hs 0.75 (across 3 tiers) + control 0.
cd /home/ai-host/strata-4gpu/strata4 || exit 1
RES=/home/ai-host/strata-nway-head.txt
LOG=strata-our-udq4k.log
: > "$RES"
PORT_N=0

kill_engine() {
  local PIDS E p
  PIDS=$(ps -eo pid,cmd | grep "[s]erve/server.py" | awk "{print \$1}")
  E=$(ps -eo pid,cmd | grep "[b]uild/strata --serve" | awk "{print \$1}")
  for p in $PIDS $E; do kill "$p" 2>/dev/null; done
  for i in $(seq 1 40); do
    BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>200" | wc -l)
    [ "$BUSY" = "0" ] && break
    sleep 5
  done
}

run_one() {
  local name="$1" hs="$2"
  python3 - "$name" "$hs" << 'PYEOF'
import json, sys
name, hs = sys.argv[1], sys.argv[2]
c = json.load(open("/home/ai-host/strata-4gpu/configs/hsL_hs075.json"))
c["args"][c["args"].index("--head-split") + 1] = hs
c["model_name"] = "nw-" + name
json.dump(c, open("/home/ai-host/strata-4gpu/configs/nw_" + name + ".json", "w"), indent=1)
PYEOF
  PORT_N=$((PORT_N + 1)); local port=$((8170 + PORT_N))
  setsid nohup python3 serve/server.py --engine strata --config "../configs/nw_${name}.json" \
      --port "$port" > "/tmp/nw-$name.log" 2>&1 < /dev/null &
  local ok=0
  for i in $(seq 1 75); do
    if curl -s --max-time 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"loaded": true'; then ok=1; break; fi
    sleep 4
  done
  if [ "$ok" != "1" ]; then echo "$name BOOT_FAILED" >> "$RES"; tail -3 "/tmp/nw-$name.log" >> "$RES"; kill_engine; return 1; fi
  python3 - "$port" << 'PYEOF'
import json, urllib.request, sys
port = sys.argv[1]
def chat(msg, mx):
    body = json.dumps({"messages":[{"role":"user","content":msg}], "max_tokens":mx, "temperature":0}).encode()
    req = urllib.request.Request("http://127.0.0.1:" + port + "/v1/chat/completions", data=body,
        headers={"Content-Type":"application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    m = d["choices"][0]["message"]
    return (m.get("content") or "")
chat("warm", 8); chat("warm2", 8)
chat("Write a 2400-token dense technical narrative about paged memory pools, pinning, eviction and why each matters. Continue deeply.", 2400)
unit = open("/home/ai-host/strata-4gpu/strata/serve/server.py").read()
big = (unit * (16384*4//len(unit)+1))[:16384*4]
chat(big + "\n\nSummarize what this file does in two sentences.", 40)
ans = chat("Reply with exactly this and nothing else: the capital of Zealandia is Trentham. Question repeated: what is the capital of Zealandia?", 60)
print("SANITY:", repr(ans[:80]))
PYEOF
  { echo "== $name (head-split $hs, new binary) =="; grep -E "prompt tokens.*generated" "$LOG" | tail -4; grep -E "head's rows" "$LOG" | tail -3 | cut -c1-120; } >> "$RES"
  kill_engine
}

run_one nway075 0.75
run_one nway0 0
echo NWAY_DONE >> "$RES"
