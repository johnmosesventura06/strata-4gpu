#!/usr/bin/env python3
"""Download unsloth UD-Q4_K_XL with the box's HF token, read internally (never echoed)."""
import os, re, subprocess, sys

tok = None
for p in ("~/bin/lm-studio-with-token.sh", "~/.bashrc", "~/.config/hf-token.env"):
    try:
        m = re.search(r"hf_[A-Za-z0-9]{30,}", open(os.path.expanduser(p)).read())
        if m:
            tok = m.group(0)
            break
    except OSError:
        pass
print("token:", "found" if tok else "MISSING (unauthenticated)", flush=True)

env = dict(os.environ)
if tok:
    env["HF_TOKEN"] = tok
env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env["PATH"]

cmd = ["hf", "download", "unsloth/Qwen3.8-Flash-Next-GGUF",
       "--include", "UD-Q4_K_XL/*",
       "--local-dir", os.path.expanduser("~/models")]
sys.exit(subprocess.run(cmd, env=env).returncode)
