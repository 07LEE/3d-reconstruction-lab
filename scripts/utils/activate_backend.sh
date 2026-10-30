#!/usr/bin/env bash
# 3DRC Backend Environment Activation
#
# Usage (sourced): source scripts/utils/activate_backend.sh && activate_backend <backend>
# Backends: the training backends in configs/backends.txt, plus tsdf, sugar, milo, grouping
#
# One shared 'gs' environment. Every backend ships its own diff_gaussian_rasterization under
# the same module name, so each build lives in .ext/<variant>/ and is put first on PYTHONPATH
# for that backend only (ADR 0018).
# GS_ENV_NAME overrides the environment name (default: gs).
# GS_EXT_ROOT overrides the .ext/ location (must match the one used by 00_setup_environment.sh).

_ACTIVATE_BACKEND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$(dirname "${BASH_SOURCE[0]}")/backends.sh"

# diff_gaussian_rasterization build each backend needs (see GS_RASTERIZER_VARIANTS in 00_setup_environment.sh)
# Training backends come from configs/backends.txt; the mesh and segmentation backends are listed here
backend_rasterizer_variant() {
    backend_variant "$1" && return
    case "$1" in
        tsdf|sugar) echo "inria" ;;
        milo)       echo "milo" ;;
        grouping)   echo "grouping" ;;
        *) return 1 ;;
    esac
}

# Activates the backend's environment and exports BACKEND_EXT_PATH .
activate_backend() {
    local backend="$1"
    local env_name="${GS_ENV_NAME:-gs}"
    local variant

    variant="$(backend_rasterizer_variant "$backend")" || { echo "[FATAL] Unknown backend: '${backend}'" >&2; exit 1; }
    local conda_base nounset=0
    conda_base=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
    if [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
        # conda's activation scripts reference unset variables
        [[ $- == *u* ]] && nounset=1
        set +u
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate "$env_name" || true
        [ "$nounset" = "1" ] && set -u
        if [ "${CONDA_DEFAULT_ENV:-}" != "$env_name" ]; then
            echo "[FATAL] Conda environment '${env_name}' not found or activation failed. Run: ./scripts/00_setup_environment.sh --env ${env_name}" >&2
            exit 1
        fi
    fi

    BACKEND_EXT_PATH="${GS_EXT_ROOT:-${_ACTIVATE_BACKEND_ROOT}/.ext}/${variant}"
    if [ ! -d "$BACKEND_EXT_PATH" ]; then
        echo "[FATAL] Rasterizer build '${variant}' not found at ${BACKEND_EXT_PATH}. Run: ./scripts/00_setup_environment.sh --env ${env_name}" >&2
        exit 1
    fi
    export PYTHONPATH="${BACKEND_EXT_PATH}${PYTHONPATH:+:$PYTHONPATH}"
    export BACKEND_EXT_PATH
}
