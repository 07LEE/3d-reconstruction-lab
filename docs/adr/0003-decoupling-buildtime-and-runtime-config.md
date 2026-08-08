# 0003. Decouple Buildtime and Runtime Configurations

Status: Accepted (2026-08-07)

Context:
default_config.sh was sourcing setup_build_env.sh, which included NVCC pre-flight checks and exit 1 statements. Non-compiling stages like SfM pose estimation failed immediately if NVCC was absent. Additionally, set -eo pipefail and exit in sourced scripts terminated interactive shell sessions.

Decision:

1. Remove setup_build_env.sh sourcing from default_config.sh.
2. Source setup_build_env.sh strictly prior to compilation (e.g. Step 4 in 00_setup_environment.sh).
3. Remove set -eo pipefail and replace exit 1 with return 1 2>/dev/null || exit 1 in setup_build_env.sh.
4. Remove -allow-unsupported-compiler flag.

Consequences:
Automated build checks do not run on runtime scripts. Compiler environment setup must be sourced explicitly when adding new build paths.
