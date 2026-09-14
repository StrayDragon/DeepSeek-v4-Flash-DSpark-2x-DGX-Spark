#!/usr/bin/env bash
# Host-side, container-free assertions for the NVFP4 lane, through the `hf` CLI.
# Everything goes through $HF_ENDPOINT, so it works behind the mirror.
#
#   ./__l8ng__/hf-assert.sh                                  # mirror by default
#   HF_ENDPOINT=https://huggingface.co ./__l8ng__/hf-assert.sh
#
# Touches only 4 small metadata files (config.json, hf_quant_config.json,
# generation_config.json, encoding/encoding_dsv4.py); the 163.49 GiB of shards
# is never downloaded. Exit 0 = every assertion held.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Precedence: shell HF_ENDPOINT -> .env.nvfp4(.example) HF_ENDPOINT -> hf-mirror.
ENVF="$DIR/.env.nvfp4"; [ -f "$ENVF" ] || ENVF="$DIR/.env.nvfp4.example"
if [ -z "${HF_ENDPOINT:-}" ] && [ -f "$ENVF" ]; then
  HF_ENDPOINT="$(awk -F= '/^HF_ENDPOINT=/{print $2; exit}' "$ENVF")"
fi
: "${HF_ENDPOINT:=https://hf-mirror.com}"
export HF_ENDPOINT

MODEL="nvidia/DeepSeek-V4-Flash-0731-NVFP4"
REV="f1caa71142bd0be02f728c79f75042ac1e461579"
META="${TMPDIR:-/tmp}/dspark-nvfp4-meta"

command -v hf >/dev/null 2>&1 || {
  echo "FAIL  'hf' CLI not on PATH (install: uv tool install huggingface_hub)" >&2
  exit 1
}

echo "== endpoint / cache =="
hf env 2>/dev/null | grep -E "^- (ENDPOINT|HF_HUB_CACHE|HF_HUB_OFFLINE):" || true

echo "== repo =="
hf models info "$MODEL" > "$META.info.json" 2>/dev/null

echo "== metadata files (4) =="
hf download "$MODEL" --revision "$REV" \
  config.json hf_quant_config.json generation_config.json encoding/encoding_dsv4.py \
  --local-dir "$META" >/dev/null

python3 - "$ENVF" "$MODEL" "$REV" "$META" "$META.info.json" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
model, rev = sys.argv[2], sys.argv[3]
meta, info_path = Path(sys.argv[4]), Path(sys.argv[5])

fails: list[str] = []


def chk(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


env = {}
for raw in env_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line and not line.startswith("#") and "=" in line:
        key, val = line.split("=", 1)
        env[key.strip()] = val.split("#", 1)[0].strip().strip('"').strip("'")
env.setdefault("HF_CACHE", str(Path.home()) + "/.cache/huggingface")
env["HF_CACHE"] = env["HF_CACHE"].replace("${HOME}", str(Path.home()))

info = json.loads(info_path.read_text(encoding="utf-8"))
cfg = json.loads((meta / "config.json").read_text(encoding="utf-8"))
qc = cfg["quantization_config"]
hq = json.loads((meta / "hf_quant_config.json").read_text(encoding="utf-8"))["quantization"]
enc_bytes = (meta / "encoding" / "encoding_dsv4.py").read_bytes()
enc = enc_bytes.decode(encoding="utf-8")
enc_md5 = hashlib.md5(enc_bytes).hexdigest()
shards = [s["rfilename"] for s in info.get("siblings", []) if s["rfilename"].endswith(".safetensors")]

print("== assertions ==")
# lane identity
chk(info["id"] == model, f"hub id = {info['id']}")
chk(info["sha"] == rev, f"hub tip sha = {info['sha']}")
chk(env["DSPARK_MODEL_OFFICIAL"] == model, f".env DSPARK_MODEL_OFFICIAL = {env['DSPARK_MODEL_OFFICIAL']}")
chk(env.get("DSPARK_REVISION") == info["sha"], ".env DSPARK_REVISION == hub tip sha")
chk(len(shards) == 48, f"{len(shards)} safetensors shards")
used = int(info.get("used_storage", 0))
chk(used > 170_000_000_000, f"used_storage = {used:,} B = {used / 1024**3:.2f} GiB")

# architecture facts the lane arithmetic depends on
chk(cfg["architectures"] == ["DeepseekV4ForCausalLM"], "architectures = DeepseekV4ForCausalLM")
chk(cfg["model_type"] == "deepseek_v4", f"model_type = {cfg['model_type']}")
chk(cfg["num_hidden_layers"] == 43, f"num_hidden_layers = {cfg['num_hidden_layers']}")
chk(cfg["hidden_size"] == 4096, f"hidden_size = {cfg['hidden_size']} (ablation direction dim)")
chk(cfg["num_nextn_predict_layers"] == 1, f"num_nextn_predict_layers = {cfg['num_nextn_predict_layers']}")
chk(cfg["dspark_block_size"] == 5, f"dspark_block_size = {cfg['dspark_block_size']}")
chk(cfg["max_position_embeddings"] == 1_048_576,
    f"max_position_embeddings = {cfg['max_position_embeddings']:,}")

# quantization shape: NVFP4 routed experts, attn/shared/head/mtp excluded
chk(qc["quant_algo"] == "MIXED_PRECISION", f"config.json quant_algo = {qc['quant_algo']}")
chk(hq["quant_algo"] == "MIXED_PRECISION", f"hf_quant_config.json quant_algo = {hq['quant_algo']}")
chk(qc.get("moe_quant_algo") == "NVFP4", f"moe_quant_algo = {qc.get('moe_quant_algo')}")
chk(qc.get("group_size") == 16 and hq.get("group_size") == 16, "group_size = 16 in both files")
chk(set(hq["exclude_modules"]) >= {"*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"},
    f"exclude_modules = {sorted(hq['exclude_modules'])}")
chk(len([k for k in hq["quantized_layers"] if k.endswith("ffn.experts")]) == 43,
    "43 layers carry NVFP4 ffn.experts")

# encoder: byte-identical to the FP8 0731, and no image placeholder
chk(enc_md5 == "cce69a92ded46ed2fd14bb23856e78ec", f"encoder md5 = {enc_md5}")
chk("IMAGE_PLACEHOLDER" not in enc, "encoder has no IMAGE_PLACEHOLDER -> Vision-Exp tower stays off")

# knobs that must agree with the live artifact
k = int(env["MTP_NUM_TOKENS"])
seqs = int(env["MAX_NUM_SEQS"])
chk(k >= cfg["dspark_block_size"], f"MTP_NUM_TOKENS={k} >= dspark_block_size")
chk(k % cfg["num_nextn_predict_layers"] == 0, f"MTP_NUM_TOKENS={k} divisible by n_predict")
chk(int(env["MAX_MODEL_LEN"]) <= cfg["max_position_embeddings"],
    f"MAX_MODEL_LEN={env['MAX_MODEL_LEN']} <= max_position_embeddings")
cap = seqs * (k + 1)
print(f"info  capture: {seqs}x{k + 1}={cap} -> padded {-(-cap // 8) * 8}")

# the host hub cache is what the container reads at /cache/huggingface
hub = ""
for line in subprocess.run(["hf", "env"], capture_output=True, text=True).stdout.splitlines():
    if line.startswith("- HF_HUB_CACHE:"):
        hub = line.split(":", 1)[1].strip()
chk(hub.endswith("/hub") and hub == env["HF_CACHE"].rstrip("/") + "/hub",
    f"hf HF_HUB_CACHE = {hub} == .env HF_CACHE/hub (same dir the container mounts)")

print()
print("ALL ASSERTIONS PASSED" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
PY
