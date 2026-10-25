#!/usr/bin/env bash
# 3DRC Clean Minimal Terminal Logger Module

get_term_width() {
    local cols
    cols=$(tput cols 2>/dev/null || echo 80)
    if [ "$cols" -gt 100 ]; then
        echo 100
    elif [ "$cols" -lt 60 ]; then
        echo 60
    else
        echo "$cols"
    fi
}

log_header() {
    local title="$1"
    echo "[STEP]  $title"
}

log_info() {
    echo "[INFO]  $*"
}

log_ok() {
    echo "[OK]    $*"
}

log_warn() {
    echo "[WARN]  $*"
}

log_fatal() {
    echo "[FATAL] $*" >&2
    exit 1
}

# Stage timing helpers: call bench_start before the measured work and
# bench_record after it succeeds. Entries go to outputs/benchmark_timings.json
# (override with BENCHMARK_FILE_3DRC).
_3DRC_LOGGER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

_bench_now() {
    if [ -n "${EPOCHREALTIME:-}" ]; then
        echo "${EPOCHREALTIME/,/.}"
    else
        date +%s
    fi
}

bench_start() {
    _3DRC_BENCH_START="$(_bench_now)"
}

# Usage: bench_record <stage> <scene> <output_dir> [KEY=VALUE ...]
bench_record() {
    local stage="$1" scene="$2" output_dir="$3"
    shift 3
    if [ -z "${_3DRC_BENCH_START:-}" ]; then
        log_warn "bench_record called without bench_start; skipping timing for ${stage}."
        return 0
    fi
    # Absolute logical path so the record does not depend on the caller's cwd
    case "$output_dir" in
        /*) ;;
        *) output_dir="$PWD/$output_dir" ;;
    esac
    local elapsed
    elapsed=$(awk -v s="$_3DRC_BENCH_START" -v e="$(_bench_now)" 'BEGIN { printf "%.2f", e - s }')
    local kv_args=()
    local kv
    for kv in "$@"; do
        kv_args+=("--set" "$kv")
    done
    python3 "$_3DRC_LOGGER_ROOT/src/utils/log_benchmark.py" \
        --stage "$stage" \
        --dataset "$scene" \
        --seconds "$elapsed" \
        --output-dir "$output_dir" \
        --project-root "$_3DRC_LOGGER_ROOT" \
        --file "${BENCHMARK_FILE_3DRC:-$_3DRC_LOGGER_ROOT/outputs/benchmark_timings.json}" \
        "${kv_args[@]}" || log_warn "Failed to record benchmark timing for ${stage}."
}
