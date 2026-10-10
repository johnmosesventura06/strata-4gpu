#!/bin/bash
# head-split ladder: 0 (control), 0.75, 1.0 — combo base, no-deep bench
cd /home/ai-host/strata-4gpu/strata4 || exit 1
RES=/home/ai-host/strata-hs-ladder.txt
LOG=strata-our-udq4k.log
: > "$RES"
PORT_N=0
kill_engine() {
  PIDS=$(ps -eo pid,cmd | grep "[s]erve/server.py" | awk "{print \$1}")
  E=$(ps -eo pid,cmd | grep "[b]uild/strata --serve" | awk "{print \$1}")
  for p in $PIDS $E; do kill "$p" 2>/dev/null; done
  # wait until every card is actually free (pinned arena release takes 10-40 s)
  for i in $(seq 1 40); do
    BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>200' | wc -l)
    [ "$BUSY" = "0" ] && break
    sleep 5
  done
}
run_one() {
  local name="$1" hs="$2"
  python3 - "$name" "$hs" << 'PYEOF'
import json, sys
name, hs = sys.argv[1], sys.argv[2]
c = json.load(open("/home/ai-host/strata-4gpu/configs/hs_real.json"))
c["args"][c["args"].index("--head-split") + 1] = hs
c["model_name"] = "hsL-" + name
json.dump(c, open(f"/home/ai-host/strata-4gpu/configs/hsL_{name}.json", "w"), indent=1)
PYEOF
  PORT_N=$((PORT_N + 1)); local port=$((8150 + PORT_N))
  setsid nohup python3 serve/server.py --engine strata --config "../configs/hsL_${name}.json" --port "$port" \
      > "/tmp/hsL-$name.log" 2>&1 < /dev/null &
  local sp=$!
  local ok=0
  for i in $(seq 1 75); do
    if curl -s --max-time 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"loaded": true'; then ok=1; break; fi
    sleep 4
  done
  [ "$ok" = "1" ] || { echo "$name BOOT_FAILED" >> "$RES"; kill_engine; return 1; }
  python3 - "$port" << 'PYEOF'
import json, time, urllib.request, sys
port = sys.argv[1]
def chat(msg, mx):
    body = json.dumps({"messages":[{"role":"user","content":msg}], "max_tokens":mx, "temperature":0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type":"application/json"})
    urllib.request.urlopen(req, timeout=1800).read()
chat("warm", 8); chat("warm2", 8)
chat("Write a 2400-token dense technical narrative about paged memory pools, pinning, eviction and why each matters. Continue deeply.", 2400)
unit = open("/home/ai-host/strata-4gpu/strata/serve/server.py").read()
big = (unit * (16384*4//len(unit)+1))[:16384*4]
chat(big + "\n\nSummarize what this file does in two sentences.", 40)
chat("Now write 1200 more tokens continuing that summary with worked examples.", 1200)
PYEOF
  { echo "== $name (head-split $hs) =="; grep -E "prompt tokens.*generated" "$LOG" | tail -3; grep -E "head's rows|head stays" "$LOG" | tail -1 | cut -c1-140; } >> "$RES"
  kill_engine
}
run_one hs0 0
run_one hs075 0.75
run_one hs100 1.0
echo LADDER_DONE >> "$RES"
