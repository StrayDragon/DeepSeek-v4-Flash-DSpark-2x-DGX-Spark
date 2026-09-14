# DeepSeek-V4-Flash-0731 **NVFP4** lane（`nvidia/DeepSeek-V4-Flash-0731-NVFP4`，2× DGX Spark / TP=2）

本目录是仓库默认 lane（`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`）之外的一条**旁路 recipe**：
同一个启动簇、同一套 hotfix 链，只换 checkpoint 与随 checkpoint 变化的三个数值。
所有文件都在 `__l8ng__/`，不改动仓库原有文件。

> **现行入口**：仓库自带的 `./start-deepseek-v4-flash-dspark.sh` + `.env.dspark`
> （`cp .env.dspark.example .env.dspark`）；本目录只作为调研记录与 CPU 门禁。
>
> `.env.dspark.example`（tracked）**只做网络预填**：`WORKER_HOST=10.0.0.2`、`MASTER_ADDR=10.0.0.1`、
> `NCCL_IB_HCA=rocep1s0f1`、`NCCL_SOCKET_IFNAME`=`TP_SOCKET_IFNAME`=`GLOO_SOCKET_IFNAME`=`enp1s0f1np1`
> （两侧同名 ⇒ 无需 `WORKER_NCCL_*`）、`VLLM_HOST_IP=10.0.0.1` / `WORKER_VLLM_HOST_IP=10.0.0.2`。
> 其余保持 upstream 默认（`ABLITERATED=0`、`k=6`、`MAX_MODEL_LEN=1048576`、`MAX_NUM_SEQS=6`、util `0.835`），
> 所以 `scripts/test-runtime-ablation.py` 的 example 门禁（断言 `ABLITERATED=0`）无需改动。
>
> 本机工作副本 `.env.dspark`（ignored）在此之上另设三项：`ABLITERATED=1`、`HF_ENDPOINT=https://hf-mirror.com`、
> `HF_DOWNLOAD_WORKERS=8`。`ABLITERATED=1` 需要 shell 里有 `HF_TOKEN`（gated 工件）。
> 实测优先级：`prepare` 只做 `set -a; source` ⇒ **文件里的显式赋值胜出**，被注释的行由 shell export 补齐；
> `start` 对 `ABLITERATED` 做了 shell 快照 ⇒ 该键上 shell 胜出。
>
> `prepare-dspark-model-cache.sh` 已改为把**非空**的 `HF_ENDPOINT` 以 `-e` 转发给两个在线块（下载 / gated 工件）；
> 未设置则保持 huggingface_hub 自带默认，离线的 `verify_cache()` 块不需要它。
>
> upstream rebase 成本：`.env.dspark.example` +13/-6（网络段 + 2 行 endpoint 注释）、`.gitignore` +1、
> `prepare-dspark-model-cache.sh` +16（endpoint 转发）、`__l8ng__/` 纯新增；其余文件与 upstream 逐字节一致。

| 文件 | 作用 |
| --- | --- |
| `README.md` | 本教程（调研结论 + 步骤 + 容量数学 + 故障表） |
| `.env.nvfp4.example` | 该 lane 的 `--env-file` 模板（已跟踪）。开工前 `cp __l8ng__/.env.nvfp4.example __l8ng__/.env.nvfp4`（工作副本，已入 `.gitignore`）；`nvfp4.sh` 优先读 `.env.nvfp4`，没有就退回 `.example` |
| `docker-compose.nvfp4.yml` | 自包含 compose（由 `docker-compose.dspark.yml` 派生，5 处差异） |
| `nvfp4.sh` | 启动簇入口：`prepare/start/stop/status/logs/smoke/check/ci` |
| `check-nvfp4.py` | **纯 CPU** 静态门禁（无需 Docker daemon），129 项 |
| `hf-assert.sh` | 走 `hf` CLI + `$HF_ENDPOINT` 的**不依赖容器**实断言（只拉 4 个小元文件） |
| `gen-compose.py` | 从 `docker-compose.dspark.yml` 重生成 `docker-compose.nvfp4.yml`（9 处替换，上流 rebase 后跑一次即可） |
| `patches`, `vllm_patch_gb10` | 指回仓库根的 symlink（见「Project directory」） |

---

## 1. 调研结论（为什么是这几个值）

**调研方式**：本机无 GPU/无镜像，全部为 dry-run——Hub API + `raw/` 文件比对 + 仓库
commit/CHANGELOG 交叉核对，最后用 `docker compose config` 做**客户端渲染**校验（不需要 daemon）。

### 1.1 checkpoint 事实（`nvidia/DeepSeek-V4-Flash-0731-NVFP4`）

| 项 | 值 | 来源 |
| --- | --- | --- |
| 分片 | 48 × `model-*.safetensors`，合计 **175 550 788 904 B = 163.49 GiB** | Hub tree API |
| 当前 main 的 commit | `f1caa71142bd0be02f728c79f75042ac1e461579` | Hub API `sha` |
| `architectures` | `DeepseekV4ForCausalLM`（与 0731 同） | `config.json` |
| `max_position_embeddings` | `1048576` | `config.json` |
| `num_hidden_layers` | `43` | `config.json` |
| `num_nextn_predict_layers` | **`1`** | `config.json` |
| `dspark_block_size` | **`5`** | `config.json` |
| 量化 | `hf_quant_config.json` → `quant_algo: MIXED_PRECISION`，`layers.N.ffn.experts = NVFP4 (group_size 16)`，`exclude_modules: *.attn.* / *.ffn.shared_experts.* / head / mtp.*` | `hf_quant_config.json` |
| 附带文件 | `encoding/encoding_dsv4.py`、`inference/*`、`tokenizer.json` 全在 | Hub tree API |

三条关键判断：

1. **`config.json` 与 `deepseek-ai/DeepSeek-V4-Flash-0731` 逐键 diff，只有 `quantization_config` 一个键不同**（其余 60+ 键全等）。
   所以本仓库为 0731 准备的全部 hotfix（#21/#22/#26/#27/#43/#55/#117/#133/#144 …）**不用改一个字节**就能用。
2. **`encoding/encoding_dsv4.py` 与 0731 的完全一致**（同为 29 001 B，md5 `cce69a92ded46ed2fd14bb23856e78ec`），
   且 **不含 `IMAGE_PLACEHOLDER`**（与 Vision-Exp 编码器的唯一结构差异）。这条直接决定 §2.2 的 vision 门禁写法。
3. **权重字节没有变小**：源 checkpoint 的 routed experts 本来就是 MXFP4（E2M1 nibble + 每 32 元素 1 个 E8M0 scale）；
   NVFP4 复用同一 E2M1 nibble 网格，只是把 scale 换成「每 16 元素 1 个 E4M3」——**无损位转换，scale 字节翻倍**。
   所以 `163.49 GiB` 比 FP8 源（`155.43 GiB`）**大 5.2 %**，而不是小。收益只可能来自 **W4A4 激活内核**，不是显存。

### 1.2 与默认 Vision-Exp lane 的差异一览

| 项 | Vision-Exp（默认） | 本 lane（0731 NVFP4） | 依据 |
| --- | --- | --- | --- |
| `DSPARK_MODEL_OFFICIAL` | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | `nvidia/DeepSeek-V4-Flash-0731-NVFP4` | — |
| `DSPARK_REVISION` | `86f746b3…` | `f1caa711…` | Hub `sha` |
| `SERVED_MODEL_NAME` | `deepseek-v4-flash-vision-exp` | `deepseek-v4-flash-0731-nvfp4 deepseek-v4-flash-0731` | 首个别名被 probe/smoke 使用 |
| `MTP_NUM_TOKENS` (k) | `6`（`n_predict=3`，需 ≥5 且整除 3） | **`5`** | `n_predict=1`、`dspark_block_size=5` → 整除天然成立，**不需要 `DSPARK_ENABLE_DSPARK_BLOCK_K`** |
| cudagraph capture | 48（6×7 pad→48） | **40**（6×6=36 pad→40） | compose 公式 `seqs*(k+1)` 上取 8 的倍数 |
| 图像输入 | 原生 `image_url`（ViT+Aligner） | **纯文本** | 0731 编码器无 `IMAGE_PLACEHOLDER` |
| 权重 | 156.29 GiB | 163.49 GiB | tree API |
| `--kv-cache-dtype` | `nvfp4_ds_mla` | `nvfp4_ds_mla`（同） | compose |

历史对照：`docs/DEEPSEEK_V4_FLASH_0731.md` 与 commit `2727c14`（"Ship text-only 0731 default"）用的就是
`MTP_NUM_TOKENS=5 / MAX_NUM_SEQS=6 / GPU_MEMORY_UTILIZATION_TEXT=0.835`——本 lane 沿用同一形状，只把权重换成 NVFP4 版本。

### 1.3 官方卡片口径 + 本对该选哪个 checkpoint

**逐键 diff 的完整口径**（两个 `config.json` 都有 `quantization_config`，只差内容）：

| | `deepseek-ai/DeepSeek-V4-Flash-0731` | `nvidia/…-NVFP4` |
| --- | --- | --- |
| 公共部分 | `quant_method: fp8`、`fmt: e4m3`、`activation_scheme: dynamic`、`weight_block_size: [128,128]`、`scale_fmt: ue8m0` | 同 |
| MoE experts | （无 `moe_quant_algo`，即 MXFP4） | `moe_quant_algo: NVFP4`、`group_size: 16` |
| `quantized_layers` | 0 条 | **43 条**（逐层 `ffn.experts`） |
| `producer` | — | `modelopt / dsv4-nvfp4-experts`（卡片：modelopt v0.46.0） |

**官方精度对照**（同一源 checkpoint，`temperature=1.0` + `max` effort；τ²/Terminal-Bench/GDPval 用 `top_p=0.95`）：

| | GPQA Diamond | AA-LCR | τ² Telecom | SciCode | IFBench | Terminal-Bench 2.1 | GDPval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MXFP4（源） | 91.5 | 72.1 | 98.7 | 51.7 | 75.8 | 74.7 | 93.0 |
| NVFP4 | 91.5 | 71.8 | 97.9 | **52.1** | 75.5 | 73.7 | **93.2** |

→ 7 项里 5 项差 ≤0.8、2 项 NVFP4 反而更高，**精度基本等价**；差异主要来自只保留了一个量：第 2 阶段校准的 **activation amax**（权重是无损位转换）。

**本对（2×128 GB、单卡可用 119.2 GiB、TP=2）的容量账**（`util=0.835`，非 KV 开销 3.75 GiB/rank，实测行比例 136 821 token/GiB）：

| checkpoint | 总大小 | 权重/rank | KV 池/rank | 可缓存 token | 相当于 1M 的 |
| --- | --- | --- | --- | --- | --- |
| `deepseek-ai/DeepSeek-V4-Flash-0731`（FP8/MXFP4） | 155.43 GiB | 77.72 | **≈18.07 GiB** | ≈2.47 M | **≈2.36×** |
| `deepseek-ai/DeepSeek-V4-Flash-DSpark` | 155.43 GiB | 77.72 | ≈18.07 GiB | ≈2.47 M | ≈2.36× |
| `nvidia/…-0731-NVFP4`（本 lane） | 163.51 GiB | 81.76 | ≈14.03 GiB | ≈1.92 M | ≈1.83× |

**Δ = 4.04 GiB/rank ≈ 552 758 token ≈ 0.53× 上下文**。所以：

> 只要不需要图像输入，**这套 2×Spark 上默认最优是 `deepseek-ai/DeepSeek-V4-Flash-0731`**（同一份 encoder / hotfix 链 + 更小的权重 → 更大 KV）。
> NVFP4 lane 的价值在 **W4A4 内核栈的 A/B**，不在容量；需要图像输入则用默认的 Vision-Exp（156.29 GiB，KV ≈17.6 GiB）。

### 1.4 0731 vs Vision-Exp（同一 wrapper，可选第三条 lane）

**结构：62 个键里 49 个全等**，差开的全在“ Drafter / 视觉塔 / 一个数值的 eps”：

| | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| --- | --- | --- |
| `num_nextn_predict_layers` | **3** | **1** |
| `vision_*`（9 个键） | `dim 1024 / n_layers 32 / n_heads 16 / patch_size 14 / inter_dim 2816 / downsample 3 / max_n_token 384 / max_wh_ratio 8 / min_pixels 147456` | **无** |
| `rms_norm_eps` | `1e-20` | `1e-06` |
| `transformers_version` | `5.0.0` | `4.57.1` |
| 相同部分 | 43 层、`hidden_size 4096`、`dspark_block_size 5`、`max_position_embeddings 1048576`、`vocab_size 129280`、FP8(`e4m3`)+MXFP4 experts | 同 |

**权重张量（index 实测）**：72 633 vs 72 317 个（+316），其中 `vision.*` 259 + `aligner.*` 4；
`total_size` = `167 811 372 792` vs `166 878 536 440` B = **156.29 vs 155.43 GiB**（Δ 0.87 GiB 总量 = 0.43 GiB/rank，即塔+2 个 drafter 层）。
两边都是 **48 个 shard**。

**编码器（决定 §2.2 的门禁）**：`encoding/encoding_dsv4.py` 分别 36 707 B / 957 行 / **8 处** `IMAGE_PLACEHOLDER`
（md5 `06a743aa6802e28f1f0e9d5fe49f1def`）vs 29 001 B / 760 行 / **0 处**（md5 `cce69a92ded46ed2fd14bb23856e78ec`）。
Vision-Exp 额外带 `inference/vision.py`、`inference/image_processor.py`、`examples/example_vl.*` + 两张图（共 84 个文件 vs 74）。

**DSpark k（同一整除规则下的不同答案）**：`k ≥ dspark_block_size(5)` 且 `k % num_nextn_predict_layers == 0`
→ Vision-Exp（n_predict=3）**6 或 9**；0731（n_predict=1）**5 / 6 / 7 都行**。本仓库两条 lane 分别是 6 与 5。

**能力（两张官方卡片，同一协议：`max` effort、`temperature=1.0`；agent 类 `top_p=0.95`）**

| 基准 | Vision-Exp | 0731 | Δ |
| --- | --- | --- | --- |
| Terminal Bench 2.1 | 83.9 | 82.7 | +1.2 |
| NL2Repo | 57.7 | 54.2 | +3.5 |
| Cybergym | 75.3 | **76.7** | −1.4 |
| DeepSWE | 59.3 | 54.4 | +4.9 |
| Toolathlon-Verified | 75.9 | 70.3 | +5.6 |
| DSBench-Hard | 63.6 | 59.6 | +4.0 |
| AutomationBench (Public) | 25.7 | 25.1 | +0.6 |
| ApexBench (Pass@1) | 36.5 | 26.2 † | +10.3 |
| Agents' Last Exam | 27.3 | 25.2 † | +2.1 |
| Chartography | 64.3 | — | — |
| ZeroBench (Pass@5) | 35.0 | — | — |

† 卡片注：0731 在这两项上**忽略多模态元素**（纯文本作答），所以那两项不是严格同条件对比。

**一句话选型**：7 个纯文本 agent 基准里 Vision-Exp 平均 **+2.6**（6 胜 1 负，Cybergym 输 1.4），
且多出 4 项多模态能力；代价只有 **0.43 GiB/rank 权重**（≈59 433 token 的 KV，约占 1M 的 0.06×）。
所以：**带图戒要图相关 agent → Vision-Exp（仓库默认）**；只要文本、且想要最大的 KV 池 → 0731；
想 A/B W4A4 内核栈 → NVFP4（§1.3）。

---

## 2. 我做的三处（+2）设计取舍

### 2.1 复用而不是重写启动簇
`nvfp4.sh` 只做一件事：把 `ENV_FILE` / `COMPOSE_FILE` 指到本目录，然后 `exec` 仓库脚本。
理由：`start-deepseek-v4-flash-dspark.sh` 里的 RoCE GID 自动解析、worker 文件同步、
`--check` preflight（先 worker 后 head）、原子回滚，都在同一进程里，重写一遍最容易出错。
仓库脚本全部支持 `ENV_FILE` / `COMPOSE_FILE` 覆盖（`start/stop/status/logs/smoke/prepare` 都是 `${VAR:-默认}`）。

### 2.2 Vision-Exp hotfix 改成**按编码器自适配**
原 compose 里 `python3 /opt/hotfix-dsv4-vision-exp.py || exit 1;` 是无条件的，而该 patcher 对
不含 `IMAGE_PLACEHOLDER` 的编码器返回 `drift:no-image-placeholder` → `_write()` 抛 `SystemExit` → 整个 boot 以 1 退出
（见 `patches/hotfix-dsv4-vision-exp.py` 的 `patch_encoding_text`）。
所以派生 compose 把它包了一层：

```sh
if [ -f "$ENCODER_INSTALLED" ] && grep -q IMAGE_PLACEHOLDER "$ENCODER_INSTALLED"; then
  python3 /opt/hotfix-dsv4-vision-exp.py || exit 1
else
  echo "[dspark] text-only checkpoint: Vision-Exp image tower not applicable"
fi
```

* 0731 / NVFP4：跳过（正好等于该 checkpoint 的原生形态）。
* Vision-Exp：仍然应用（`IMAGE_PLACEHOLDER` 存在）。
* `|| exit 1` 保留：真应用时依旧 fail-closed。

### 2.3 Project directory 与 symlink（最容易踩的坑）
`docker compose` 的相对路径是**相对第一个 `-f` 文件所在目录**解析的：

* head：`-f __l8ng__/docker-compose.nvfp4.yml` → project dir = `__l8ng__/`
* worker：launcher 把同一份文件 scp 成 `$WORKER_DIR/docker-compose.dspark.yml` → project dir = 仓库根

于是 `./patches/…` 在两侧解析基准不同。处理办法就是那两个 symlink：
`__l8ng__/patches -> ../patches`、`__l8ng__/vllm_patch_gb10 -> ../vllm_patch_gb10`。
`check-nvfp4.py` 第 6 组门禁逐个 mount 检查两侧都能落到同一个真实文件；
再用 `docker compose config` 双渲染对拍（head / repo-root 各一次），除 project name 外**逐行相同**。

> 另外：`TP3_PATCH_DIR`、`VLLM_GB10_PATCH_DIR` 由 launcher 以**绝对路径**内联传入，本来就不依赖 project dir。
> **cwd 也无关**：三处 `docker compose config` 对拍（cwd = 仓库根 / `__l8ng__` / `/tmp`）除 `name:` 外逐行相同。
> 所以 `cd` 到仓库根或 `__l8ng__` 都可以；建议在仓库根用 `./__l8ng__/nvfp4.sh …`。

### 2.3.1 `.env` 读取链（全部自动，实测）

| 环节 | 行为 |
| --- | --- |
| wrapper | `DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)` → `ENV_FILE`/`COMPOSE_FILE` 是**绝对路径**；cwd=仓库根 / `__l8ng__` / `/tmp` 三种下 `bash -x` 打出同一对路径；`ENV_FILE` 优先级：export 值 → `.env.nvfp4` → `.env.nvfp4.example` |
| head | launcher 把 env 文件归一化进 `mktemp` 副本（去 BOM/CR、`chmod 600`，`COMPOSE_ENV_FILE`）再作 `--env-file`；所以本 lane 的文件叫 `.env.nvfp4` 也可以 |
| worker | 同一份字节被原子写成 `$WORKER_DIR/.env.dspark`（**名字固定**），远端命令一律 `--env-file .env.dspark`（相对 `cd $WORKER_DIR`） |
| `${HOME}` | env 文件里的嵌套 `${HOME}` 由 **compose 客户端进程环境**展开（实测：`HOME=…` 覆盖会跟随；`env -u HOME` 则 warning + `/.cache/huggingface`）。worker 侧 `HF_CACHE` 由 bash 先展开再以绝对值内联注入（`WORKER_HF_CACHE` 默认取 `HF_CACHE`），所以只有 head 依赖嵌套展开 |

两个需要人看一眼的点：① 若 shell 里预先 export 过旧的 `ENV_FILE`/`COMPOSE_FILE`，`${VAR:-…}` 会让旧值胜出（`echo $ENV_FILE` 核对，或 `env -u ENV_FILE -u COMPOSE_FILE`）；② worker 的 `.env.dspark` 每次 `start` 都被覆盖，不要手改。


### 2.4 保留 `--limit-mm-per-prompt`
纯文本 checkpoint 下它是惰性的（最多一行 warning），保留它可以让本 lane 与默认 lane 的
`vllm serve` argv **只在 model / served-name / k 三处不同**，A/B 时更干净。

### 2.5 `GPU_MEMORY_UTILIZATION` 由 launcher 注入
compose 里的 `--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.80}` 只在**不经过 launcher** 时取 0.80。
`start-*.sh` 会用 `GPU_MEMORY_UTILIZATION_TEXT`（本 lane `0.835`）覆盖并导出——所以别手填 `GPU_MEMORY_UTILIZATION`。

---

## 3. 执行步骤

### 3.0 HF_ENDPOINT 与下载路（镜像）

| 路 | 走哪个 endpoint | 备注 |
| --- | --- | --- |
| host `hf` CLI | `$HF_ENDPOINT`（已写 `https://hf-mirror.com` 到 `.env.nvfp4.example`） | 本 lane 推荐；`hf-assert.sh` 优先级：shell → env 文件 → mirror |
| 仓库 `prepare-*.sh`（容器） | 固定 `huggingface.co` | `docker run` 是**白名单式 `-e`**（HF_HOME / HF_HUB_OFFLINE=0 / TRANSFORMERS_OFFLINE=0 / HF_HUB_DISABLE_XET / 两个 timeout / DSPARK_MODEL / DSPARK_REVISION / HF_DOWNLOAD_WORKERS / token），**不转发 `HF_ENDPOINT`** |
| serve 容器 | 无需 endpoint | `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`，只读 `/cache/huggingface` |

用镜像时，权重直接由 host `hf` 写进同一个 hub（不要 `--local-dir`，vLLM 按 repo id 解 `models--…` 布局）：

```bash
uv tool install huggingface_hub            # 提供 hf / huggingface-cli
export HF_ENDPOINT=https://hf-mirror.com
hf env | grep -E "ENDPOINT|HF_HUB_CACHE"   # 确认落在 $HF_CACHE/hub
hf download nvidia/DeepSeek-V4-Flash-0731-NVFP4 --revision f1caa71142bd0be02f728c79f75042ac1e461579
# worker：rsync -a --info=progress2 "$HF_CACHE/" user@worker:"$HF_CACHE/"  或 DSPARK_WORKER_HF_NFS=1
```

实测（本开发机，走镜像）：`hf models info` 与 `hf download` 均成功，4 个小文件共 <30 KiB，断言全部成立（见下）。

### 3.1 开发机（dry-run，CPU-only，不需要 daemon）

```bash
bash -n __l8ng__/nvfp4.sh                      # 语法
./__l8ng__/hf-assert.sh                        # 走镜像的实断言（25 项 + 1 info）
./__l8ng__/nvfp4.sh check                      # 静态门禁（不需 daemon）
./__l8ng__/nvfp4.sh ci                         # 仓库 CPU 门禁（scripts/ci-validate.sh）
```

两个可选：`python3 __l8ng__/gen-compose.py`（**仅 upstream rebase 后第一步**，从上流 compose 重生成）、
`docker compose --env-file … -f … config`（不连 daemon 的渲染对拍）。

本机实测输出（2026-09-14）：`__l8ng__/hf-assert.sh` = **ALL ASSERTIONS PASSED**（含 `used_storage = 175,562,226,307 B = 163.51 GiB`、`num_nextn_predict_layers = 1`、`dspark_block_size = 5`、encoder md5 `cce69a92…` 无 `IMAGE_PLACEHOLDER`）；
`check` = **129 passed, 0 failed**；`ci` = `CI validate passed (CPU recipe gates only)`。

### 3.2 部署机

```bash
# 0) 工作副本（模板已填本对的值，直接可用）
cp __l8ng__/.env.nvfp4.example __l8ng__/.env.nvfp4

# 1) 两侧同镜像（digest 已钉）+ 免密核对
docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8

# 2) 权重（163.49 GiB）：head 下载 + rsync 给 worker，见 §3.4
PREPARE_WORKER=0 ./__l8ng__/nvfp4.sh prepare --yes

# 3) 起（launcher 自己按 rank：先 worker 后 head）
./__l8ng__/nvfp4.sh start
./__l8ng__/nvfp4.sh status
./__l8ng__/nvfp4.sh smoke
./__l8ng__/nvfp4.sh logs               # TAIL=400 可调
./__l8ng__/nvfp4.sh stop
```

一次性改绑定/端口：`./__l8ng__/nvfp4.sh start --host 0.0.0.0 --port 9000`。
重启后 dockerd 可能已用 `restart: unless-stopped` 拉起 rank，此时 `start` 以 **3** 退出（不是错误，systemd 写 `SuccessExitStatus=3`）。

### 3.3 启动日志该看到什么（trust live 数字）

```text
DSpark ... num_speculative_tokens=5, draft_sample_method=probabilistic
[dspark] text-only checkpoint: Vision-Exp image tower not applicable   ← §2.2 的跳过行
Available KV cache memory: ~14 GiB（估，见 §4）
GPU KV cache size: ~1.9M tokens（估）
Maximum concurrency for 1,048,576 tokens per request: ~1.8x
```

`/v1/models` 里应是 `"id": "deepseek-v4-flash-0731-nvfp4"`、`"max_model_len": 1048576`。

### 3.4 本对（head 10.0.0.1 / worker 10.0.0.2）完整流程 + rsync

**接口映射（两侧同名，所以不需要 `WORKER_NCCL_*` 覆盖）**

| 侧 | 选中的 | IP | 作用 |
| --- | --- | --- | --- |
| head | `enp1s0f1np1` (HCA `rocep1s0f1`) | 10.0.0.1/24 | RoCE 数据面 + TCP bootstrap（`MASTER_ADDR`） |
| worker | `enp1s0f1np1` (HCA `rocep1s0f1`) | 10.0.0.2/24 | 同上 |

其余都不入选：`enP7s7`（LAN 192.168.123.31 / .30，仅 TP=3 lane 用）、`enp1s0f0np0` 与
`enP2p1s0f0np0`（NO-CARRIER）、第二张 CX 卡的 `enP2p1s0f1np1`（无 IPv4）、`tailscale0`/`docker0`/`br-*`。
两个口当前 MTU 都是 **1500**；若日后改 jumbo，两侧同一个口一起改，再看 NCCL 日志里的 ring 带宽。

```bash
# 0) 两侧：同镜像（digest 已钉）；earlyoom 建议关（或把 vLLM 加进 oom 分数保护）
 on both nodes:  docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8

# 1) head 上：cp 工作副本 + 确认两侧 checkout 路径一致 + SSH 免密
cp __l8ng__/.env.nvfp4.example __l8ng__/.env.nvfp4
ssh l8ng@10.0.0.2 'pwd; ip -4 -o addr show dev enp1s0f1np1'   # pwd 与 head 一致则 WORKER_DIR 留空

# 2) 下载权重到 head（PREPARE_WORKER=0 → 不递归 worker，下面用 rsync）
 export HF_ENDPOINT=https://hf-mirror.com
 PREPARE_WORKER=0 ./__l8ng__/nvfp4.sh prepare --yes
 #   或者走 host hf（同样写进 $HOME/.cache/huggingface）：
   hf download nvidia/DeepSeek-V4-Flash-0731-NVFP4 --revision f1caa71142bd0be02f728c79f75042ac1e461579

# 3) rsync 到 worker（整个 HF 目录，不只 hub：JIT 缓存也在里面）
 rsync -a --info=progress2 ~/.cache/huggingface/ l8ng@10.0.0.2:/home/l8ng/.cache/huggingface/
 #   验证：两侧同一快照号
   ls ~/.cache/huggingface/hub/models--nvidia--DeepSeek-V4-Flash-0731-NVFP4/snapshots
   ssh l8ng@10.0.0.2 'ls ~/.cache/huggingface/hub/models--nvidia--DeepSeek-V4-Flash-0731-NVFP4/snapshots'

# 4) 起（launcher 自己先 worker 后 head，带 --check preflight）
 ./__l8ng__/nvfp4.sh start
 ./__l8ng__/nvfp4.sh status && ./__l8ng__/nvfp4.sh smoke
 ./__l8ng__/nvfp4.sh logs            # TAIL=400 可调
 curl -fsS http://10.0.0.1:8888/v1/models
```

**rsync 要点（为什么这样写）**

| 写法 | 原因 |
| --- | --- |
| `PREPARE_WORKER=0` | 仓库 `prepare` 默认会 ssh 到 worker 再跑一次下载；置 0 就只下 head，由 rsync 接手 |
| 拷整个 `~/.cache/huggingface/`（不是只 `hub/`） | compose 把 `TRITON_CACHE_DIR` / `TILELANG_CACHE_DIR` / `B12X_CUTE_COMPILE_CACHE_DIR` / `VLLM_CACHE_ROOT` / `flashinfer` / `nccl-fr` 全放在同一卷（`/cache/huggingface/…`），一并带过去可省首次 JIT |
| `rsync -a`（含 `-l`） | hub 的 `snapshots/<rev>/*` 是指向 `../../blobs/<sha>` 的**相对 symlink**，`-a` 原样保留链接，两侧目录同构 → 容器内直接可读 |
| 建 `/home/l8ng/.cache/huggingface` 先 `mkdir -p` | `rsync -a` 不会递推建多级空目，先建避免“no such directory” |
| 不加 `--delete` | 同一卷里还有别的 repo（戒答方向 `dspark-ablation/` 等），`--delete` 会误删 |

首次 boot 会编译几分钟（新快照 + 新形状），`healthcheck` 的 `start_period=480s` 就是为此；不要中途 restart。


---

## 4. 容量数学（估算，供对照 live 日志）

行格式与 0731 FP8 完全同（`config.json` 只有 `quantization_config` 不同，43 层、`compress_ratios`、`head_dim` 全等），
所以可以直接用本集群 measured 的 **17.04 GiB ↔ 2 331 430 tokens** 比例：
**136 821 tokens/GiB（≈7 849 B/token）**。

非 KV 开销由同一 measured 点反推：`0.83 × 119.2 − 78.15 − 17.04 = 3.75 GiB/rank`
（119.2 GiB = GB10 的 128 GB(十进制)，78.15 GiB = Vision-Exp 156.29 GiB / 2）。

| util | KV / rank | cached tokens | 对 1M 的并发度 |
| ---: | ---: | ---: | ---: |
| 0.835 | ≈14.0 GiB | ≈1.92 M | ≈1.83× |
| 0.86 | ≈17.0 GiB | ≈2.33 M | ≈2.22× |
| 0.88 | ≈19.4 GiB | ≈2.65 M | ≈2.53× |

权重：`163.49 / 2 = 81.75 GiB / rank`（比 FP8 的 77.72 GiB/rank 多 4.03 GiB，即 §1.1 的 scale 翻倍）。

若跑的是源 checkpoint（`deepseek-ai/DeepSeek-V4-Flash-0731`，77.72 GiB/rank），同一个池子公式下：

| util | KV / rank | cached tokens | 对 1M 的并发度 |
| ---: | ---: | ---: | ---: |
| 0.835 | ≈18.07 GiB | ≈2.47 M | ≈2.36× |
| 0.86 | ≈21.05 GiB | ≈2.88 M | ≈2.75× |
| 0.88 | ≈23.43 GiB | ≈3.20 M | ≈3.06× |
默认 `0.835`；若 live 日志的 KV pool 明显低于上表，先看 `--gpu-memory-utilization` 是否真被 launcher 注入成 0.835（§2.5）。
`max_model_len` / `max_num_seqs` 是**上限不是预留**：`6 × 320 K ≈ 1.92 M` 就是这个池子。

---

## 5. 与 nvidia 模型卡建议的对应关系

| nvidia 卡（B200，TP=8） | 本 recipe（2× GB10，TP=2） | 说明 |
| --- | --- | --- |
| `--tensor-parallel-size 8 --enable-expert-parallel` | `--tensor-parallel-size 2`，`mp`，`nnodes 2` | 两机 2 卡，无 EP |
| `--max-model-len 393216` | `1048576` | 本 recipe 主打 1M 上限 |
| `--kv-cache-dtype fp8` | `nvfp4_ds_mla` | 本 recipe 的 584-B sparse-MLA 信封 + issue #22 修复 |
| `--attention_config.use_fp4_indexer_cache=True` | `DSPARK_ENABLE_MXFP4_INDEXER_CACHE=1`（并要求 `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1`） | 同一个 fp4 indexer cache，默认 0 |
| `--moe-runner-backend flashinfer_trtllm_routed` | `--moe-backend flashinfer_b12x`（`VLLM_USE_B12X_MOE=1`） | GB10/sm_121a 路径 |
| `--tokenizer-mode/--tool-call-parser/--reasoning-parser deepseek_v4` | 同名三个 | 一致 |
| MTP/DSpark「未在本 release 验证，启用前自测」 | `MTP_NUM_TOKENS=5` + `probabilistic` | 见 §5.1 |

**k 的两个合法取值**（`n_predict=1` 整除任意 k，且 k ≥ `dspark_block_size=5`）：

| k | 来路 | capture = `seqs*(k+1)` | 上取 8 的倍数 |
| --- | --- | --- | --- |
| **5**（本 lane 默认） | DSpark 头本身 `dspark_block_size=5` | 6×6 = 36 | **40** |
| **7**（两张官方卡示例） | `{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}` | 6×8 = 48 | **48** |

A/B 时建议先跑 **7 + greedy** 对齐官方口径，再跑 **5 + probabilistic**（两次都要 `stop` + `start`）。

### 5.1 可选旋钮（都默认 0，一次只开一个）

| 变量 | 为什么在这条 lane 上值得试 |
| --- | --- |
| `ENABLE_VLLM_GB10_PATCH=1` | `modelopt_gb10_hybrid`：`M<128` 走 Marlin W4A16，`M≥128` 走 CUTLASS W4A4。**只对 NVFP4 制品有意义**（FP8 源没有 NVFP4 linear 层）。启动会 `pip install -e`（2026-09-08 起 fail-closed）。 |
| `DSPARK_ENABLE_MXFP4_INDEXER_CACHE=1` | 对应卡片的 fp4 indexer cache；需 `DSPARK_ENABLE_DEEPGEMM_SM121_ALIAS=1`（冷 JIT 缓存才能编出 sm_121 内核）。 |
| `DSPARK_ENABLE_ROPE_SWA_FIX=1` | 稀疏-SWA 层用普通 RoPE 而非 YaRN（vllm#54815）；0731/Vision-Exp 同一 `rope_scaling`。 |
| `DSPARK_ENABLE_DSPARK_SWA_PREFIX=1` / `DSPARK_ENABLE_DSML_RECOVERY=1` / `DSPARK_ENABLE_ISSUE144_EFFORT_ALIGN=1` | 前缀缓存重复 prompt、DSML 容错、跨 effort 的 block 共享。 |
| `ABLITERATED=1` | 同一份官方权重 + 18 KiB 拒答方向（`λ=3.5`、层 `10-42`）。本 lane 上可用，见 §5.2。 |

A/B 建议顺序：先只换 checkpoint（1→2→3），再逐个开 `ENABLE_VLLM_GB10_PATCH`、
`DSPARK_ENABLE_MXFP4_INDEXER_CACHE`。每次都要 `stop` + `start` 重建两个 rank（hotfix 写在容器可写层，`restart` 不会重放）。

### 5.2 `ABLITERATED`（runtime refusal-direction ablation）在本 lane 是否仍适用 —— **适用**

机制（`patches/hotfix-dsv4-runtime-ablation.py` + launcher）：

| 环节 | 事实 | 对 NVFP4 的影响 |
| --- | --- | --- |
| 权重 | 翻牌后**仍然 serve `DSPARK_MODEL_OFFICIAL`**，只是多下一个 18 KiB 方向；不拉 157 GiB Keys dump | 不变：仍是 `nvidia/DeepSeek-V4-Flash-0731-NVFP4` |
| 方向张量 | `broad`（或 `directions[0]`），**必须 `numel()==4096`**、有限、模>0 | nvidia `config.json` 的 `hidden_size` 与 0731 **完全相同 = 4096** → 维度校验直接通过 |
| 层范围 | 只给 `layers.10 … layers.42` 绑方向；launcher 还校验上界 ≤42 | `num_hidden_layers=43`（索引 0-42）**两份 checkpoint 相同** → 10-42 仍然有效 |
| 注入点 | 在 `x = self.attn(positions, x, None)` 之后做 `y ← y − λ(y·v)v`（fp32、`v` 已 L2 归一），Anemll 上 hook 站点数应为 **1** | 改的是镜像里的 `nvidia/model.py`，**与权重量化格式无关** |
| 方向出处 | `drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-Anchored-Tensors` 的 `ablit/refusal_direction_r1.pt`，sha256 `6e4d8a8f…`，**在 0731 FP8 DSpark 上标定** | 该 FP8 0731 正是 NVFP4 制品的 `base_model`，**出处一致**（比 Vision-Exp 的迁移更近） |
| AOT 缓存 | `VLLM_CACHE_ROOT/.dsv4_ablate_stamp` 记录 `enabled/λ/layers/direction/patch` 五个字段，任一变化即删 `torch_compile_cache` | 翻牌 = 一次全量重编译，首次 boot 明显变慢属正常 |

要留意的两点（NVFP4 特有）：

1. **`λ=3.5` 要在本 lane 上复核**。方向本身是同一份字节，但 NVFP4 的 routed-expert Linear 用 **dynamic 4-bit 激活**
   （`input_activations: num_bits 4, group_size 16`），而 FP8 源是 e4m3 激活 → hidden state 的尺度会有二阶漂移；
   `λ` 是作用在归一化方向上的标量，所以 3.5 是合理起点，但需要一次 live A/B（同一 prompt、`temperature: 0`）。
2. 层的编号来自 checkpoint，`attn / shared_experts / head / mtp` 在 NVFP4 里**保持 FP8 未被 4-bit 化**
   （`exclude_modules`），所以注入点（attention 之后）读到的仍是 FP8 精度激活，进一步缩小与 FP8 标定的差异。

操作步骤（两次 recreate，不是 `restart`）：

```bash
# 1) 在 https://huggingface.co/drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit 同意条款（需要 HF_TOKEN）
# 2) __l8ng__/.env.nvfp4 里 ABLITERATED=1
${EDITOR:-vi} __l8ng__/.env.nvfp4          # ABLITERATED=1
./__l8ng__/nvfp4.sh prepare --yes          # 下 RESPONSIBLE_USE.md + 18 KiB 方向（两侧都 stage，SHA-256 校验）
./__l8ng__/nvfp4.sh stop && ./__l8ng__/nvfp4.sh start
./__l8ng__/nvfp4.sh logs | grep dsv4-ablation   # 期望 [dsv4-ablation] applied to … (1 attention hook site(s))
```

失败即 fail-closed：缺 `RESPONSIBLE_USE.md`/方向、方向 sha 不符、`numel != 4096`、hook 站点数不在 `1|2|3`，
都会让容器在 exec vLLM 前退出。`ABLITERATED=0`（本 lane 默认）时该 patch 只维护 AOT stamp，等于惰性。

> `DSPARK_MODEL_ABLITERATED` 在本 lane 仍是 `drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit`：
> 它只是**条款所在的 gated repo id**，方向字节取自 `…-GA-0731-Dspark-Abliterated-Anchored-Tensors`（`files/README.md`）。

---

## 6. 故障表

| 现象 | 原因 / 处理 |
| --- | --- |
| boot 停在 `FATAL: … drift:no-image-placeholder` | 用的是**原始** `docker-compose.dspark.yml`（无条件调 vision patcher）。本 lane 必须 `-f __l8ng__/docker-compose.nvfp4.yml`（`nvfp4.sh` 已设）。 |
| `line 87: deepseek-v4-flash-0731: command not found` | launcher 用 bash **source** 这个 env 文件，所以多词值必须加引号（`SERVED_MODEL_NAME="a b"` 已加好；`check` 会跑 `bash -c '. file'` 断言 source-clean）。 |
| `[Errno 101] Network is unreachable`（容器内 httpx） | `prepare` 的 `docker run` **不带 `--network`** ⇒ 走默认 bridge，需要 host 转发。三选一：① `sudo sysctl -w net.ipv4.ip_forward=1` 后重跑；② 直接用 host `hf download`（§3.0，同时吃 `HF_ENDPOINT`）；③ `docker info` 看 `iptables` 是否被关掉。bridge 下 `--network host` 手跑同样的 python 片段可确认是哪一层断。 |
| `nvfp4: no env file` | 未 cp 工作副本：`cp __l8ng__/.env.nvfp4.example __l8ng__/.env.nvfp4`（只有 `.example` 时 wrapper 会自动退回用它）。若 `echo $ENV_FILE` 非空，说明被外部 export 覆盖，`env -u ENV_FILE` 重试。 |
| mount 报「no such file or directory」且路径少一层 | 那两个 symlink 没进 checkout（`git clone` 会带；`scp -r` 单层时注意）。跑 `python3 __l8ng__/check-nvfp4.py` 第 6 组门禁定位。 |
| KV pool 比 §4 表小很多 | 看 `non-default args` 里的 `gpu_memory_utilization`；没被注入就是绕过了 `start-*.sh`。 |
| `--max-cudagraph-capture-size` 出现 `Truncating … to 24` | 旧 Anemll 0.25.2 观察（`.env.dspark.example` 已记）。40 是请求值；以 live 日志为准。 |
| 改 `MTP_NUM_TOKENS` 后 decode 变慢但 prefill 不变 | 正常：capture 尺寸随 `seqs*(k+1)` 变。改 k 后要一并重测 c=6 聚合。 |
| `start` 退出码 3 | 已由 `restart: unless-stopped` 拉起，不是失败。冷启动才 `stop`。 |
| `prepare` 下载超时/不可达 | 镜像里的 `docker run` 不转发 `HF_ENDPOINT`（固定 `-e` 白名单），所以它总走 `huggingface.co`。用镜像时改走 host `hf download`（§3.0），同一个 hub 布局，serve 侧无感。 |
| NCCL 初始化要 ~2 min | 本对的 `NCCL_GIN_ENABLE=0` 可选（2026-09-05 实测 → ~13 s，serve 带宽不变）；仅影响 bootstrap。 |
| GID 解析 / `ibv_modify_qp 61` | 保持默认 `NCCL_IB_GID_AUTO=1`：两侧 HCA 同名（`rocep1s0f1`），launcher 逐 member 从 sysfs 校验后把 `NCCL_IB_GID_INDEX` 留空。只在禁用自动时才手动钉。 |

---

## 7. dry-run 未覆盖、需 live 确认

1. `nvfp4_ds_mla` + NVFP4 权重在 sm_121 上的 KV 池实测值（§4 是比例外推）。
2. `MTP_NUM_TOKENS=5` 的 draft acceptance / decode tok/s（nvidia 卡明确说 MTP 未在其验证内）；
   与官方示例的 **7 + greedy** 做一次双点 A/B（capture 40 vs 48，其余不动）。
3. `modelopt_gb10_hybrid` 与 `flashinfer_b12x` MoE 后端是否同时生效（`nvidia/model.py` 走 ModelOpt linear，MoE 走 b12x）。
4. `--limit-mm-per-prompt` 在纯文本 model 上是否只出 warning（本机无镜像，未在 Anemll 0.1.1 上跑过）。
5. `ABLITERATED=1` 下 `λ=3.5` 在 NVFP4（dynamic 4-bit 激活）上的效果：方向字节同源（0731 FP8 DSpark），
   但激活量化格式变了，需一次 `temperature: 0` 的 0/1 A/B。
