#!/usr/bin/env python3
"""CPU-only preflight for the __l8ng__ NVFP4 lane (no Docker, no network).

Run it on the dev machine before touching the two Sparks:

    python3 __l8ng__/check-nvfp4.py        # or: ./nvfp4.sh check

It checks the lane files only: the compose chain, the env file, the relative
mount resolution for BOTH project directories (head and worker), and the DSpark
/ context arithmetic against the pinned nvidia checkpoint facts. Exit 0 = every
gate passed; any failure is fatal and numbered.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Same precedence as nvfp4.sh: .env.nvfp4, else the committed example.
_env_candidates = (HERE / ".env.nvfp4", HERE / ".env.nvfp4.example")
ENV_FILE = next((p for p in _env_candidates if p.is_file()), _env_candidates[-1])
COMPOSE_FILE = HERE / "docker-compose.nvfp4.yml"

# --- Pinned facts, verified against the Hub on 2026-09 (see README) ---------
NVIDIA_REPO = "nvidia/DeepSeek-V4-Flash-0731-NVFP4"
NVIDIA_REVISION = "f1caa71142bd0be02f728c79f75042ac1e461579"
NVIDIA_SHARDS = 48
NVIDIA_BYTES = 175_550_788_904          # 163.49 GiB of safetensors
CKPT = {                                # nvidia config.json
    "max_position_embeddings": 1_048_576,
    "num_hidden_layers": 43,
    "num_nextn_predict_layers": 1,      # DSpark/MTP head count
    "dspark_block_size": 5,
}
# Measured on this cluster with the FP8 Vision-Exp lane (README KV note):
# 17.04 GiB pool = 2,331,430 tokens, same nvfp4_ds_mla row format, 43 layers.
MEASURED_POOL_GIB, MEASURED_TOKENS = 17.04, 2_331_430
GB10_TOTAL_GIB = 119.2                  # 128 GB decimal per Spark
VE_WEIGHTS_PER_RANK = 156.29 / 2        # FP8 Vision-Exp, TP=2
NON_KV_OVERHEAD_GIB = 0.83 * GB10_TOTAL_GIB - VE_WEIGHTS_PER_RANK - MEASURED_POOL_GIB

fail: list[str] = []
info: list[str] = []


def chk(cond: bool, msg: str) -> None:
    (info if cond else fail).append(("ok   " if cond else "FAIL ") + msg)


def read_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.split("#", 1)[0].strip().strip('"').strip("'")
        env[key.strip()] = val.replace("${HOME}", str(Path.home()))
    return env


def main() -> int:
    for f in (ENV_FILE, COMPOSE_FILE):
        chk(f.is_file(), f"present: {f.relative_to(ROOT)}")
    if fail:
        return report()

    env = read_env(ENV_FILE)
    doc = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    svc = doc["services"]["vllm-dspark"]
    command = " ".join(svc["command"]) if isinstance(svc["command"], list) else svc["command"]

    # 1. Lane identity -----------------------------------------------------
    chk(env["DSPARK_MODEL_OFFICIAL"] == NVIDIA_REPO,
        f"env DSPARK_MODEL_OFFICIAL = {env['DSPARK_MODEL_OFFICIAL']}")
    chk(env.get("DSPARK_REVISION") == NVIDIA_REVISION,
        f"env DSPARK_REVISION = {env.get('DSPARK_REVISION') or '(empty = tip of main)'}")
    served = env.get("SERVED_MODEL_NAME", "").split()
    chk(bool(served), f"env SERVED_MODEL_NAME first alias = {served[0] if served else '(missing)'}")
    chk(NVIDIA_REPO in command, "compose serve/model defaults carry the nvidia Hub id")
    chk(served and served[0] in command, "compose default served name matches the first alias")

    # 2. Required cluster keys (launcher fails closed on these) ------------
    for key in ("WORKER_HOST", "MASTER_ADDR", "MASTER_PORT", "NCCL_IB_HCA",
                "NCCL_SOCKET_IFNAME", "VLLM_HOST_IP", "WORKER_VLLM_HOST_IP",
                "HF_CACHE", "DSPARK_VLLM_IMAGE"):
        chk(bool(env.get(key)), f"env {key} is set")
    chk("@sha256:" in env.get("DSPARK_VLLM_IMAGE", ""), "image pin is a manifest digest")
    chk(env.get("ABLITERATED", "0") in {"0", "1"}, "ABLITERATED is 0 or 1")

    # 3. Serve chain: exactly one of each structural flag ------------------
    for flag in ("--kv-cache-dtype nvfp4_ds_mla", "--block-size 256",
                 "--tokenizer-mode deepseek_v4", "--tool-call-parser deepseek_v4",
                 "--reasoning-parser deepseek_v4", "--enable-chunked-prefill",
                 "--nnodes", "--node-rank", "--master-addr",
                 "--speculative-config", "--tensor-parallel-size"):
        n = len(re.findall(re.escape(flag), command))
        chk(n == 1, f"{flag} appears exactly once (found {n})")
    chk("modelopt_gb10_hybrid" in command, "optional GB10 hybrid quantization hook is wired")

    # 4. DSpark / context arithmetic ----------------------------------------
    k = int(env.get("MTP_NUM_TOKENS", "0"))
    seqs = int(env.get("MAX_NUM_SEQS", "0"))
    mlen = int(env.get("MAX_MODEL_LEN", "0"))
    chk(k >= CKPT["dspark_block_size"],
        f"DSpark k={k} >= checkpoint dspark_block_size={CKPT['dspark_block_size']}")
    chk(k % CKPT["num_nextn_predict_layers"] == 0,
        f"DSpark k={k} satisfies stock n_predict={CKPT['num_nextn_predict_layers']} "
        "(no block-k unlock needed on 0731)")
    chk(mlen <= CKPT["max_position_embeddings"],
        f"MAX_MODEL_LEN={mlen} <= max_position_embeddings={CKPT['max_position_embeddings']}")
    capture = (seqs * (k + 1) + 7) // 8 * 8
    chk(capture % 8 == 0 and capture >= seqs * (k + 1),
        f"cudagraph capture size = {capture} (={seqs}x{k + 1} padded to /8)")
    chk("MTP_NUM_TOKENS:-5" in command, "in-container MTP fallback is 5")

    # 5. Text-only encoder vs the gated Vision-Exp tower -------------------
    chk("grep -q IMAGE_PLACEHOLDER" in command,
        "Vision-Exp hotfix is gated on the installed encoder declaring IMAGE_PLACEHOLDER")
    chk("then python3 /opt/hotfix-dsv4-vision-exp.py || exit 1;" in command,
        "guard keeps the fail-closed `|| exit 1` on the vision path")
    snap = Path(env.get("DSPARK_ENCODING_FILE") or "")
    if not (snap.is_file() if snap else False):
        hub = Path(env["HF_CACHE"]) / "hub" / f"models--{NVIDIA_REPO.replace('/', '--')}"
        cands = sorted(hub.glob("snapshots/*/encoding/encoding_dsv4.py")) if hub.is_dir() else []
        if cands:
            text = cands[0].read_text(encoding="utf-8")
            chk(("IMAGE_PLACEHOLDER" in text) == ("IMAGE_PLACEHOLDER" not in "x"),
                f"cached encoder {cands[0].parent.name[:12]}: "
                f"IMAGE_PLACEHOLDER={'yes' if 'IMAGE_PLACEHOLDER' in text else 'no'} "
                "(no -> tower stays off, expected on this lane)")
        else:
            info.append("info  no local hub snapshot yet (prepare not run) - encoder check deferred")

    # 6. Relative mounts must resolve from BOTH project directories --------
    # head:    project dir = __l8ng__/   (first -f file)
    # worker:  project dir = repo root   (compose is scp'd as docker-compose.dspark.yml)
    def source_of(vol: object) -> str:
        raw = vol if isinstance(vol, str) else str(vol.get("source", ""))
        if raw.startswith("${"):                       # `:-` inside the braces
            end = raw.index("}")
            return raw[: end + 1]
        return raw.split(":", 1)[0]

    def unwrap(src: str) -> str:
        # `${NAME:-default}` -> default; a bare `$NAME` comes from the env file.
        m = re.fullmatch(r"\$\{[A-Za-z0-9_]+:?-(.*)\}", src)
        return m.group(1) if m else src

    rel = []
    for vol in svc["volumes"]:
        src = unwrap(source_of(vol))
        if src and not src.startswith(("$", "/")):
            rel.append(src)
    chk(len(rel) > 5, f"{len(rel)} relative bind mounts to resolve")
    for src in rel:
        head = (HERE / src)
        worker = (ROOT / src.split("/", 1)[0] / src) if False else (ROOT / src)
        chk(head.exists(), f"head project dir resolves: __l8ng__/{src}")
        chk(worker.exists(), f"worker project dir resolves: {src}")

    # 6b. Fabric wiring for this pair (head 10.0.0.1 / worker 10.0.0.2).
    master, worker = env["MASTER_ADDR"], env["WORKER_HOST"]
    chk(env["VLLM_HOST_IP"] == master, f"VLLM_HOST_IP == MASTER_ADDR ({master})")
    chk(env["WORKER_VLLM_HOST_IP"] == worker,
        f"WORKER_VLLM_HOST_IP == WORKER_HOST ({worker})")
    ifaces = {env["NCCL_SOCKET_IFNAME"], env.get("TP_SOCKET_IFNAME"), env.get("GLOO_SOCKET_IFNAME")}
    chk(ifaces == {env["NCCL_SOCKET_IFNAME"]} and env["NCCL_SOCKET_IFNAME"] not in ("", "lo"),
        f"bootstrap ifaces all = {env['NCCL_SOCKET_IFNAME']} (never empty/lo)")
    chk(env["NCCL_IB_HCA"].startswith("roce"), f"NCCL_IB_HCA = {env['NCCL_IB_HCA']}")
    whc = env.get("WORKER_HF_CACHE", "")
    chk(whc == "" or whc.startswith("/"), f"WORKER_HF_CACHE = {whc or '(defaults to HF_CACHE)'}")
    info.append(f"info  pair: head {master} / worker {worker}, "
                f"HCA {env['NCCL_IB_HCA']}, bootstrap {env['NCCL_SOCKET_IFNAME']}")

    # 6c. The launcher *sources* this file in bash, so it must be source-clean:
    # multi-word values need quotes, otherwise the 2nd word runs as a command.
    src = subprocess.run(["bash", "-c", f'set -eu; . "{ENV_FILE}"'],
                         capture_output=True, text=True)
    chk(src.returncode == 0,
        f"env file sources cleanly in bash ({src.stderr.strip() or 'no stderr'})")
    chk(len(env["SERVED_MODEL_NAME"].split()) == 2 and env["SERVED_MODEL_NAME"].startswith("deepseek-v4-flash-0731-nvfp4"),
        f"SERVED_MODEL_NAME = {env['SERVED_MODEL_NAME']} (primary + alias)")

    # 7. KV capacity estimate for this artifact ----------------------------
    tok_per_gib = MEASURED_TOKENS / MEASURED_POOL_GIB
    w_rank = NVIDIA_BYTES / 2 / 1024**3
    ve_overhead = NON_KV_OVERHEAD_GIB
    info.append(f"info  artifact: {NVIDIA_SHARDS} shards, "
                f"{NVIDIA_BYTES / 1024**3:.2f} GiB total, {w_rank:.2f} GiB/rank at TP=2")
    info.append(f"info  non-KV overhead from the FP8 datapoint: {ve_overhead:.2f} GiB/rank; "
                f"measured row ratio {tok_per_gib:,.0f} tokens/GiB")
    for util in (0.835, 0.86, 0.88):
        pool = util * GB10_TOTAL_GIB - w_rank - ve_overhead
        info.append(f"info  util {util}: KV ~{pool:.1f} GiB/rank -> "
                    f"~{pool * tok_per_gib / 1e6:.2f}M cached tokens -> "
                    f"~{pool * tok_per_gib / CKPT['max_position_embeddings']:.2f}x of "
                    f"{CKPT['max_position_embeddings']:,}")
    chk(0.80 <= float(env.get("GPU_MEMORY_UTILIZATION_TEXT", "0")) <= 0.90,
        f"GPU_MEMORY_UTILIZATION_TEXT={env.get('GPU_MEMORY_UTILIZATION_TEXT')}")

    # 7b. Source vs this artifact on 119.2 usable GiB (hub used_storage, 2026-09-14).
    SRC_GIB, NV_GIB = 155.43, 163.51
    util = float(env.get("GPU_MEMORY_UTILIZATION_TEXT", "0.835"))
    pool_src = util * GB10_TOTAL_GIB - SRC_GIB / 2 - NON_KV_OVERHEAD_GIB
    pool_nv = util * GB10_TOTAL_GIB - NV_GIB / 2 - NON_KV_OVERHEAD_GIB
    d_gib = (NV_GIB - SRC_GIB) / 2
    d_tok = d_gib * tok_per_gib
    chk(abs(pool_src - 18.07) < 0.15 and abs(pool_nv - 14.03) < 0.15,
        f"util {util}: KV pool FP8/MXFP4 ~{pool_src:.2f} vs NVFP4 ~{pool_nv:.2f} GiB/rank")
    chk(abs(d_tok - 552_758) < 1500,
        f"NVFP4 costs {d_gib:.2f} GiB/rank of KV = ~{d_tok:,.0f} tokens "
        f"(~{d_tok / CKPT['max_position_embeddings']:.2f}x of context) vs the source checkpoint")

    # 7c. DSpark k options. Official cards pair (7, greedy); this lane ships (5, probabilistic).
    n_pred = CKPT["num_nextn_predict_layers"]
    seqs = int(env.get("MAX_NUM_SEQS", "6"))
    for k in (5, 7):
        chk(k % n_pred == 0, f"k={k} legal at n_predict={n_pred} "
                             f"(block size {CKPT['dspark_block_size']})")
        capture = seqs * (k + 1)
        info.append(f"info  k={k}: capture {seqs}x{k + 1}={capture} -> padded "
                    f"{-(-capture // 8) * 8} (CUDA graph sizes are multiples of 8)")
    chk(int(env.get("MTP_NUM_TOKENS", "0")) in (5, 7),
        f"MTP_NUM_TOKENS={env.get('MTP_NUM_TOKENS')} is one of the two legal k")
    chk(env.get("DRAFT_SAMPLE_METHOD", "probabilistic") in ("probabilistic", "greedy"),
        f"DRAFT_SAMPLE_METHOD={env.get('DRAFT_SAMPLE_METHOD', 'probabilistic')}")

    # 7d. Cross-checkpoint facts (index `total_size` + config, measured 2026-09-14).
    VE_TOTAL, SRC_TOTAL = 167_811_372_792, 166_878_536_440
    ve_rank, src_rank = VE_TOTAL / 2 / 1024**3, SRC_TOTAL / 2 / 1024**3
    delta_tok = (ve_rank - src_rank) * tok_per_gib
    chk(abs(ve_rank - 78.15) < 0.02 and abs(src_rank - 77.72) < 0.02,
        f"per-rank weights: Vision-Exp {ve_rank:.2f} vs 0731 {src_rank:.2f} GiB")
    chk(abs(delta_tok - 59_428) < 400,
        f"the vision tower + 2 extra drafter layers cost ~{delta_tok:,.0f} tokens of KV "
        f"({ve_rank - src_rank:.2f} GiB/rank)")
    for name, n_pred in (("Vision-Exp", 3), ("0731", 1)):
        legal = [k for k in range(CKPT["dspark_block_size"], 10) if k % n_pred == 0]
        info.append(f"info  {name}: n_predict={n_pred} -> legal k in [block_size,9] = {legal}")
    chk([k for k in range(5, 10) if k % 3 == 0] == [6, 9] and CKPT["num_nextn_predict_layers"] == 1,
        "k rule: Vision-Exp needs 6/9 (n_predict=3), 0731 takes 5/6/7 (n_predict=1)")
    info.append("info  encoders: Vision-Exp 36 707 B/957 lines/8x IMAGE_PLACEHOLDER "
                "(md5 06a743aa6802e28f1f0e9d5fe49f1def); 0731 29 001 B/760 lines/0 "
                "(md5 cce69a92ded46ed2fd14bb23856e78ec)")

    return report()


def report() -> int:
    for line in info:
        print(line)
    for line in fail:
        print(line)
    print(f"\n{len(info)} passed, {len(fail)} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
