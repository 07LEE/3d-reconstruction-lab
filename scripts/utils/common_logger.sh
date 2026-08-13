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
    local width
    width=$(get_term_width)
    local sep
    sep=$(printf '=%.0s' $(seq 1 "$width"))
    echo ""
    echo "$sep"
    echo "  $title"
    echo "$sep"
    echo ""
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
