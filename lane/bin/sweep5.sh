#!/bin/bash
# v5: remaining cheap decode knobs on the hs075 base.
cd /home/ai-host/strata-4gpu/strata4 || exit 1
RES=/home/ai-host/strata-sweep5.txt
LOG=strata-our-udq4k.log
RT=/home/ai-host/strata-4gpu/Strata-data/mtp/rt/draft_vocab.bin
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
  local name="$1"; shift
  python3 - "$@" << PYEOF
import json
c = json.load(open("/home/ai-host/strata-4gpu/configs/hsL_hs075.json"))
extra = $@
a = c["args"]
for i in range(0, len(extra), 2):
    f, v = extra[i], extra[i+1]
    if f in a: a[a.index(f) + 1] = v
    else: a += [f, v]
c["model_name"] = "s5-" + "$name"
json.dump(c, open("/home/ai-host/strata-4gpu/configs/s5_$name.json", "w"), indent=1)
PYEOF
  local vocab_swap=0
  if [ "$name" = "envocab" ]; then
    cp /home/ai-host/strata-4gpu/strata4/data/draft_vocab_en.bin "$RT"; vocab_swap=1
  fi
  PORT_N=$((PORT_N + 1)); local port=$((8160 + PORT_N))
  setsid nohup python3 serve/server.py --engine strata --config "../configs/s5_${name}.json" \
      --port "$port" > "/tmp/s5-$name.log" 2>&1 < /dev/null &
  local ok=0
  for i in $(seq 1 75); do
    if curl -s --max-time 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"loaded": true'; then ok=1; break; fi
    sleep 4
  done
  if [ "$ok" != "1" ]; then echo "$name BOOT_FAILED" >> "$RES"; [ "$vocab_swap" = "1" ] && cp /home/ai-host/strata-4gpu/strata4/data/draft_vocab.bin "$RT"; kill_engine; return 1; fi
  python3 - "$port" << 'PYEOF'
import json, urllib.request, sys
port = sys.argv[1]
def chat(msg, mx):
    body = json.dumps({"messages":[{"role":"user","content":msg}], "max_tokens":mx, "temperature":0}).encode()
    req = urllib.request.Request("http://127.0.0.1:" + port + "/v1/chat/completions", data=body,
        headers={"Content-Type":"application/json"})
    urllib.request.urlopen(req, timeout=1800).read()
chat("warm", 8); chat("warm2", 8)
chat("Write a 2400-token dense technical narrative about paged memory pools, pinning, eviction and why each matters. Continue deeply.", 2400)
unit = open("/home/ai-host/strata-4gpu/strata/serve/server.py").read()
big = (unit * (16384*4//len(unit)+1))[:16384*4]
chat(big + "\n\nSummarize what this file does in two sentences.", 40)
chat("Now write 1200 more tokens continuing that summary with worked examples.", 1200)
PYEOF
  { echo "== $name =="; grep -E "prompt tokens.*generated" "$LOG" | tail -3; } >> "$RES"
  [ "$vocab_swap" = "1" ] && cp /home/ai-host/strata-4gpu/strata4/data/draft_vocab.bin "$RT"
  kill_engine
}

run_one envocab "[]"
run_one lookup24 "[\"--spec-lookup\",\"24\"]"
run_one minmb2 "[\"--second-gpu-min-mb\",\"2\"]"
run_one spec6 "[\"--spec\",\"6\"]"
echo SWEEP5_DONE >> "$RES"
