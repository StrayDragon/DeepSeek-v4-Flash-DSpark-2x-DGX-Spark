from pathlib import Path
src = Path("docker-compose.dspark.yml").read_text()
orig = src

R = []
def rep(old, new, n=1):
    global src
    assert src.count(old) >= 1, f"MISS: {old[:70]}"
    src = src.replace(old, new, n)
    R.append((old[:60], new[:60]))

HDR = """# __l8ng__ lane: nvidia/DeepSeek-V4-Flash-0731-NVFP4 (text-only, ModelOpt
# MIXED_PRECISION: NVFP4 routed experts + FP8/MXFP4-preserving rest).
# Derived byte-for-byte from docker-compose.dspark.yml with FIVE deltas:
#   1. model / served-name defaults -> nvidia/DeepSeek-V4-Flash-0731-NVFP4
#   2. MTP_NUM_TOKENS fallback 6 -> 5 (0731: num_nextn_predict_layers=1,
#      dspark_block_size=5, so k=5 satisfies k %% n_predict without the
#      block-k unlock)
#   3. encoder fallback also probes models--nvidia--DeepSeek-V4-Flash-0731-NVFP4
#   4. the Vision-Exp hotfix is now gated on the checkpoint encoder declaring
#      IMAGE_PLACEHOLDER (0731/NVFP4 encoder has none -> skipped, no FATAL
#      "drift:no-image-placeholder"; Vision-Exp still gets it)
#   5. header comments
# Everything else is the upstream chain. See __l8ng__/README.md.
"""
src = HDR + src

rep('DSPARK_MODEL: "${DSPARK_MODEL:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"',
    'DSPARK_MODEL: "${DSPARK_MODEL:-nvidia/DeepSeek-V4-Flash-0731-NVFP4}"')
rep('MTP_NUM_TOKENS: "${MTP_NUM_TOKENS:-6}"',
    'MTP_NUM_TOKENS: "${MTP_NUM_TOKENS:-5}"')
rep('MODEL_HUB_DIR=$$(printf \'%s\' "$${DSPARK_MODEL:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}" | sed \'s|/|--|g\');',
    'MODEL_HUB_DIR=$$(printf \'%s\' "$${DSPARK_MODEL:-nvidia/DeepSeek-V4-Flash-0731-NVFP4}" | sed \'s|/|--|g\');')
rep('if [ -z "$${ENCODING_SOURCE}" ] && [ -f /models/deepseek-ai/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py ]; then ENCODING_SOURCE=/models/deepseek-ai/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py; fi;',
    'if [ -z "$${ENCODING_SOURCE}" ]; then for candidate in /cache/huggingface/hub/models--nvidia--DeepSeek-V4-Flash-0731-NVFP4/snapshots/*/encoding/encoding_dsv4.py; do if [ -f "$${candidate}" ]; then ENCODING_SOURCE="$${candidate}"; break; fi; done; fi;\n'
    '        if [ -z "$${ENCODING_SOURCE}" ] && [ -f /models/deepseek-ai/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py ]; then ENCODING_SOURCE=/models/deepseek-ai/DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py; fi;')
rep('python3 /opt/hotfix-dsv4-vision-exp.py || exit 1;',
    'ENCODER_INSTALLED=/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py; '
    'if [ -f "$${ENCODER_INSTALLED}" ] && grep -q IMAGE_PLACEHOLDER "$${ENCODER_INSTALLED}"; then python3 /opt/hotfix-dsv4-vision-exp.py || exit 1; '
    'else echo "[dspark] text-only checkpoint: Vision-Exp image tower not applicable"; fi;')
rep('SPECULATIVE_CONFIG="{\\"method\\":\\"dspark\\",\\"num_speculative_tokens\\":$${MTP_NUM_TOKENS:-6}',
    'SPECULATIVE_CONFIG="{\\"method\\":\\"dspark\\",\\"num_speculative_tokens\\":$${MTP_NUM_TOKENS:-5}')
rep('--max-cudagraph-capture-size $$(( ( ${MAX_NUM_SEQS:-6} * (${MTP_NUM_TOKENS:-6} + 1) + 7 ) / 8 * 8 ))',
    '--max-cudagraph-capture-size $$(( ( ${MAX_NUM_SEQS:-6} * (${MTP_NUM_TOKENS:-5} + 1) + 7 ) / 8 * 8 ))')
rep('exec /usr/local/bin/vllm serve ${DSPARK_MODEL:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}',
    'exec /usr/local/bin/vllm serve ${DSPARK_MODEL:-nvidia/DeepSeek-V4-Flash-0731-NVFP4}')
rep('--served-model-name ${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp}',
    '--served-model-name ${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-nvfp4}')

Path("__l8ng__/docker-compose.nvfp4.yml").write_text(src)
print("deltas applied:", len(R))
