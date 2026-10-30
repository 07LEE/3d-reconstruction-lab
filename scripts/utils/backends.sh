#!/usr/bin/env bash
# 3DRC Training Backend Lookups
#
# Usage (sourced): source scripts/utils/backends.sh
# configs/backends.txt lists the training backends and explains its columns.
#   backend_names [separator]    every backend name, joined by the separator (default: a space)
#   backend_exists <name>        succeeds for a listed backend
#   backend_script <name>        stage script under scripts/
#   backend_model_subdir <name>  model directory under outputs/<scene>/
#   backend_has_shortcut <name>  succeeds when "run_3drc.sh <name>" trains that backend
#   backend_step <name>          step label taken from the script name, e.g. 2c
#   backend_title <name>         display name, e.g. 2D Gaussian Splatting
#   backend_header <name>        e.g. "Step 2c: 2D Gaussian Splatting Training"
#   backend_variant <name>       diff_gaussian_rasterization build the backend runs with
#   backend_repo <name>          checkout whose render.py and metrics.py evaluate the backend's models
#   backend_infer <path>         backend whose models live under the path
# Lookups of a backend that is not listed fail.

_BACKENDS_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/configs/backends.txt"

# Prints one column of a backend's row
_backend_column() {
    awk -v name="$1" -v col="$2" '/^[[:space:]]*#/ { next } $1 == name { print $col; found = 1 } END { exit !found }' "$_BACKENDS_FILE"
}

backend_names() {
    awk -v sep="${1:- }" '/^[[:space:]]*#/ { next } NF { printf "%s%s", out, $1; out = sep } END { print "" }' "$_BACKENDS_FILE"
}

backend_exists() {
    _backend_column "$1" 1 > /dev/null
}

backend_script() {
    _backend_column "$1" 2
}

backend_model_subdir() {
    _backend_column "$1" 3
}

backend_has_shortcut() {
    [ "$(_backend_column "$1" 4)" = "yes" ]
}

backend_step() {
    local script
    script=$(backend_script "$1") || return 1
    script="${script%%_*}"
    echo "${script#0}"
}

backend_title() {
    local title
    title=$(_backend_column "$1" 5) || return 1
    echo "${title//_/ }"
}

backend_header() {
    local step title
    step=$(backend_step "$1") && title=$(backend_title "$1") || return 1
    echo "Step ${step}: ${title} Training"
}

backend_variant() {
    _backend_column "$1" 6
}

backend_repo() {
    _backend_column "$1" 7
}

# Backends are tried in the order listed. A model directory matches by prefix (2dgs_run2 counts as 2dgs).
# When none does, a path under the directory that holds a backend's model counts for the first backend
# there (3dgs/<anything> is 3dgs).
backend_infer() {
    local path="$1" name subdir
    while read -r name subdir; do
        case "$path" in */"$subdir"*) echo "$name"; return 0 ;; esac
    done < <(awk '/^[[:space:]]*#/ { next } NF { print $1, $3 }' "$_BACKENDS_FILE")
    while read -r name subdir; do
        case "$subdir" in
            */*) case "$path" in */"${subdir%/*}"/*) echo "$name"; return 0 ;; esac ;;
        esac
    done < <(awk '/^[[:space:]]*#/ { next } NF { print $1, $3 }' "$_BACKENDS_FILE")
    return 1
}
