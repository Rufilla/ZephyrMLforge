#!/usr/bin/env bash
#
# ZephyrMLForge Setup Script
# Initialises the Python environment and verifies the Zephyr workspace
#
# Usage: ./setup.sh
#
# Prerequisites:
#   - An initialised west workspace (west init && west update), found automatically
#     at ~/zephyrproject or /opt/zephyrproject, or named by MLFORGE_WEST_WORKSPACE
#   - A Zephyr SDK under ~, ~/.local or /opt
#   - Python 3.10+, cmake, ninja or make
#
# This script is idempotent and non-interactive.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing=()
    command -v python3 &> /dev/null || missing+=("python3")
    command -v pip3    &> /dev/null || missing+=("pip3")
    command -v git     &> /dev/null || missing+=("git")
    command -v cmake   &> /dev/null || missing+=("cmake")
    if ! command -v ninja &> /dev/null && ! command -v make &> /dev/null; then
        missing+=("ninja or make")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing prerequisites: ${missing[*]}"
        exit 1
    fi

    # Compared as a version, not as a number: 'bc' reads 3.9 and 3.10 as decimals and
    # decides 3.9 is the larger, so a 3.9 interpreter passed a 3.10 requirement.
    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
        log_error "Python 3.10+ required, found $(python3 -V)"
        exit 1
    fi
    log_info "Found $(python3 -V)"

    if command -v dot &> /dev/null; then
        log_info "Found graphviz (model architecture diagrams enabled)"
    else
        log_warn "graphviz not found; model architecture diagrams will be skipped"
        log_warn "  Install with: sudo apt install graphviz"
    fi
}

find_workspace() {
    local candidate
    for candidate in \
        "${ZEPHYR_BASE:+$(dirname "${ZEPHYR_BASE}")}" \
        "${MLFORGE_WEST_WORKSPACE:-}" \
        "${HOME}/zephyrproject" \
        "/opt/zephyrproject"
    do
        if [[ -n "${candidate}" && -d "${candidate}/.west" && -d "${candidate}/zephyr" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

check_zephyr_workspace() {
    log_info "Looking for a west workspace..."

    if ! WEST_WORKSPACE="$(find_workspace)"; then
        log_error "No initialised west workspace found."
        log_error "Looked at \$ZEPHYR_BASE, \$MLFORGE_WEST_WORKSPACE, ~/zephyrproject, /opt/zephyrproject"
        log_error "Initialise one first:"
        log_error "  mkdir -p ~/zephyrproject && cd ~/zephyrproject && west init && west update"
        exit 1
    fi

    local version
    version=$(grep VERSION_MAJOR "${WEST_WORKSPACE}/zephyr/VERSION" | cut -d= -f2 | tr -d ' ')
    log_info "Found Zephyr workspace at ${WEST_WORKSPACE} (v${version}.x)"
}

check_zephyr_sdk() {
    log_info "Looking for a Zephyr SDK..."

    local sdk
    sdk=$(for candidate in "${HOME}"/zephyr-sdk-* "${HOME}"/.local/zephyr-sdk-* /opt/zephyr-sdk-*; do
        [[ -d "${candidate}" ]] && echo "${candidate}"
    done | sort -V | tail -1)

    if [[ -n "${sdk}" ]]; then
        log_info "Found Zephyr SDK at ${sdk}"
        export ZEPHYR_SDK_INSTALL_DIR="${sdk}"
    else
        log_warn "No Zephyr SDK found under ~, ~/.local or /opt"
        log_warn "  Install from https://github.com/zephyrproject-rtos/sdk-ng/releases"
    fi
}

setup_python_venv() {
    log_info "Setting up the Python virtual environment..."

    if [[ -d "${VENV_DIR}" ]]; then
        log_info "Virtual environment already exists at ${VENV_DIR}"
    else
        python3 -m venv "${VENV_DIR}"
        log_info "Created virtual environment at ${VENV_DIR}"
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    pip install --quiet --upgrade pip wheel setuptools
    pip install --quiet -e "${SCRIPT_DIR}[dev]"
    log_info "Installed the pipeline package and its dependencies"

    # west is unusable without these: its build command imports jsonschema and others,
    # and fails with a bare ModuleNotFoundError rather than anything actionable.
    if [[ -f "${WEST_WORKSPACE}/zephyr/scripts/requirements.txt" ]]; then
        pip install --quiet -r "${WEST_WORKSPACE}/zephyr/scripts/requirements.txt"
        log_info "Installed the Zephyr Python requirements"
    else
        log_warn "No Zephyr requirements.txt in ${WEST_WORKSPACE}/zephyr/scripts"
    fi
}

setup_tflite_micro() {
    log_info "Checking the TFLite Micro module..."

    if [[ -d "${WEST_WORKSPACE}/optional/modules/lib/tflite-micro" ]] ||
       [[ -d "${WEST_WORKSPACE}/modules/lib/tflite-micro" ]]; then
        log_info "TFLite Micro module already present"
        return 0
    fi

    log_info "Adding the TFLite Micro module (this may take a moment)..."
    (
        cd "${WEST_WORKSPACE}"
        west config manifest.project-filter -- +tflite-micro
        west update tflite-micro
    )

    if [[ -d "${WEST_WORKSPACE}/optional/modules/lib/tflite-micro" ]]; then
        log_info "TFLite Micro module installed"
    else
        log_warn "TFLite Micro installation may have failed. Run manually:"
        log_warn "  cd ${WEST_WORKSPACE} && west config manifest.project-filter -- +tflite-micro"
        log_warn "  cd ${WEST_WORKSPACE} && west update tflite-micro"
    fi
}

setup_environment_file() {
    local env_file="${SCRIPT_DIR}/.env"

    if [[ -f "${env_file}" ]]; then
        log_info "Environment file already exists at ${env_file}"
        return
    fi

    cat > "${env_file}" << 'EOF'
# ZephyrMLForge environment configuration

# AI provider API keys (one is required unless ai_agent.provider is 'pseudo')
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=

# Override the west workspace location if it is not found automatically
# MLFORGE_WEST_WORKSPACE=/opt/zephyrproject
EOF

    log_info "Created ${env_file}"
    log_warn "Add your API key to ${env_file} before running with a real provider"
}

verify_installation() {
    log_info "Verifying the installation..."

    if west --version &> /dev/null; then
        log_info "west: $(west --version)"
    else
        log_error "west is installed but not runnable; check the Zephyr requirements above"
        return 1
    fi

    if [[ -d "${WEST_WORKSPACE}/optional/modules/lib/tflite-micro" ]] ||
       [[ -d "${WEST_WORKSPACE}/modules/lib/tflite-micro" ]]; then
        log_info "TFLite Micro: present"
    else
        log_warn "TFLite Micro: absent; firmware builds will fail"
    fi

    if command -v qemu-system-arm &> /dev/null; then
        log_info "QEMU: $(command -v qemu-system-arm)"
    else
        log_warn "QEMU: not on PATH; 'source activate.sh' adds the SDK's copy"
    fi

    python3 -c "import tensorflow" 2>/dev/null && log_info "TensorFlow: installed" \
        || log_warn "TensorFlow: not installed"

    command -v zephyr-ml-forge &> /dev/null && log_info "CLI: zephyr-ml-forge available" \
        || log_warn "CLI: zephyr-ml-forge not on PATH"

    log_info "Running the test suite..."
    python3 -m pytest "${SCRIPT_DIR}/tests" -q || log_warn "Test suite failed"
}

main() {
    log_info "=== ZephyrMLForge setup ==="

    check_prerequisites
    check_zephyr_workspace
    check_zephyr_sdk
    setup_python_venv
    setup_tflite_micro
    setup_environment_file
    verify_installation

    log_info "=== Setup complete ==="
    log_info ""
    log_info "Next steps:"
    log_info "  1. Add an API key to .env, unless running with ai_agent.provider: pseudo"
    log_info "  2. source activate.sh"
    log_info "  3. zephyr-ml-forge run --simulate"
}

main "$@"
