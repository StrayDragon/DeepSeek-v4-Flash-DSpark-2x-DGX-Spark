#!/usr/bin/env bash
# __l8ng__ entry point for the nvidia/DeepSeek-V4-Flash-0731-NVFP4 lane.
#
# Thin dispatcher over the repo's existing two-node launcher: it only pins
# ENV_FILE / COMPOSE_FILE to the __l8ng__ copies, so every sync, GID-resolve,
# preflight and rollback behaviour of ./start-deepseek-v4-flash-dspark.sh is
# reused unchanged. Nothing here talks to Docker directly.
#
# Usage: ./nvfp4.sh <prepare|start|stop|status|logs|smoke|check|ci> [extra args]
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"

# Both files are absolute; the compose-relative `./patches/...` mounts resolve
# through __l8ng__/patches -> ../patches (see README "Project directory").
# Env file precedence: exported ENV_FILE -> .env.nvfp4 -> .env.nvfp4.example
# (same "copy the example" style as the repo's .env.dspark.example).
_if="${ENV_FILE:-}"
if [ -z "$_if" ]; then
  if   [ -f "$DIR/.env.nvfp4" ];         then _if="$DIR/.env.nvfp4"
  elif [ -f "$DIR/.env.nvfp4.example" ]; then _if="$DIR/.env.nvfp4.example"
  else echo "nvfp4: no env file; run: cp $DIR/.env.nvfp4.example $DIR/.env.nvfp4" >&2; exit 2
  fi
fi
export ENV_FILE="$_if"
export COMPOSE_FILE="${COMPOSE_FILE:-$DIR/docker-compose.nvfp4.yml}"

usage() {
  cat <<'EOF'
Usage: ./nvfp4.sh <command> [args]

  prepare [--yes]   download nvidia/DeepSeek-V4-Flash-0731-NVFP4 into HF_CACHE
                    (head + worker; or head only with DSPARK_WORKER_HF_NFS=1)
  start             bring up rank 1 (worker) then rank 0 (head), TP=2
  stop              tear the pair down (--nfs also drops dspark-nfs)
  status            container/health summary for both ranks
  logs              tail both ranks (TAIL=... to change)
  smoke             one-shot chat probe against the served name
  check             CPU-only static gates for this lane (no Docker needed)
  ci                repo-wide CPU recipe gates (scripts/ci-validate.sh)

Environment: ENV_FILE / COMPOSE_FILE may be overridden; otherwise this
directory's .env.nvfp4 (or .env.nvfp4.example when not copied yet) and
docker-compose.nvfp4.yml are used.
EOF
}

cmd="${1:-}"
[ -n "$cmd" ] || { usage; exit 2; }
shift || true

[ "$cmd" = prepare ] || [ "$cmd" = start ] || [ "$cmd" = check ] || \
  echo "nvfp4: ENV_FILE=$ENV_FILE" >&2

case "$cmd" in
  prepare) exec bash "$ROOT/prepare-dspark-model-cache.sh" "$@" ;;
  start)   exec bash "$ROOT/start-deepseek-v4-flash-dspark.sh" "$@" ;;
  stop)    exec bash "$ROOT/stop-deepseek-v4-flash-dspark.sh" "$@" ;;
  status)  exec bash "$ROOT/status-deepseek-v4-flash-dspark.sh" "$@" ;;
  logs)    exec bash "$ROOT/logs-deepseek-v4-flash-dspark.sh" "$@" ;;
  smoke)   exec bash "$ROOT/smoke-deepseek-v4-flash-dspark.sh" "$@" ;;
  check)   exec python3 "$DIR/check-nvfp4.py" "$@" ;;
  ci)      exec bash "$ROOT/scripts/ci-validate.sh" "$@" ;;
  -h|--help|help) usage ;;
  *) echo "Unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
