#!/usr/bin/env bash
# 3DRC Backend Environment Activation
#
# Usage (sourced): source scripts/utils/activate_backend.sh && activate_backend <backend>
# Backends: 3dgs, 2dgs, tsdf, planargs, sugar, scaffoldgs, mipsplatting, milo, grouping
#
# GS_ENV_MODE=unified (default): one shared 'gs' environment. Every backend ships its own
# diff_gaussian_rasterization under the same module name, so each build lives in
# .ext/<variant>/ and is put first on PYTHONPATH for that backend only (ADR 0018).
# GS_ENV_MODE=split: one Conda environment per backend family (ADR 0001).
# GS_EXT_ROOT overrides the .ext/ location (must match the one used by 00_setup_environment.sh).

_ACTIVATE_BACKEND_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

backend_split_env() {
    case "$1" in
        3dgs|2dgs|tsdf) echo "gs_train" ;;
        planargs|sugar) echo "gs_sugar" ;;
        scaffoldgs)     echo "gs_scaffold" ;;
        mipsplatting)   echo "gs_mipsplatting" ;;
        milo)           echo "gs_milo" ;;
        grouping)       echo "gs_group" ;;
        *) return 1 ;;
    esac
}

# diff_gaussian_rasterization build each backend needs (see GS_RASTERIZER_VARIANTS in 00_setup_environment.sh)
backend_rasterizer_variant() {
    case "$1" in
        3dgs|2dgs|tsdf|planargs|sugar) echo "inria" ;;
        scaffoldgs)   echo "scaffold" ;;
        mipsplatting) echo "mipsplatting" ;;
        milo)         echo "milo" ;;
        grouping)     echo "grouping" ;;
        *) return 1 ;;
    esac
}

# Activates the backend's environment and exports BACKEND_EXT_PATH (empty in split mode).
activate_backend() {
    local backend="$1"
    local mode="${GS_ENV_MODE:-unified}"
    local variant env_name

    variant="$(backend_rasterizer_variant "$backend")" || { echo "[FATAL] Unknown backend: '${backend}'" >&2; exit 1; }
    case "$mode" in
        split)   env_name="$(backend_split_env "$backend")" ;;
        unified) env_name="gs" ;;
        *) echo "[FATAL] GS_ENV_MODE must be 'split' or 'unified', got '${mode}'" >&2; exit 1 ;;
    esac

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

    BACKEND_EXT_PATH=""
    if [ "$mode" = "unified" ]; then
        BACKEND_EXT_PATH="${GS_EXT_ROOT:-${_ACTIVATE_BACKEND_ROOT}/.ext}/${variant}"
        if [ ! -d "$BACKEND_EXT_PATH" ]; then
            echo "[FATAL] Rasterizer build '${variant}' not found at ${BACKEND_EXT_PATH}. Run: ./scripts/00_setup_environment.sh --env gs" >&2
            exit 1
        fi
        export PYTHONPATH="${BACKEND_EXT_PATH}${PYTHONPATH:+:$PYTHONPATH}"
    fi
    export BACKEND_EXT_PATH
}
