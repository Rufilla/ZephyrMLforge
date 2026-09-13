#!/usr/bin/env bash
# Source this file to activate the ZephyrMLForge environment
# Usage: source activate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prints the first directory that looks like an initialised west workspace.
mlforge_find_workspace() {
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

# Prints the highest-numbered Zephyr SDK found, or nothing.
mlforge_find_sdk() {
    local candidate
    for candidate in "${HOME}"/zephyr-sdk-* "${HOME}"/.local/zephyr-sdk-* /opt/zephyr-sdk-*; do
        [[ -d "${candidate}" ]] && echo "${candidate}"
    done | sort -V | tail -1
}

if [[ -d "${SCRIPT_DIR}/.venv" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/bin/activate"
else
    echo "Error: virtual environment not found. Run ./setup.sh first."
    return 1
fi

MLFORGE_WORKSPACE="$(mlforge_find_workspace)" || true
if [[ -n "${MLFORGE_WORKSPACE}" ]]; then
    export ZEPHYR_BASE="${MLFORGE_WORKSPACE}/zephyr"
else
    echo "Warning: no west workspace found. Set MLFORGE_WEST_WORKSPACE to its path."
fi

MLFORGE_SDK="$(mlforge_find_sdk)"
if [[ -n "${MLFORGE_SDK}" ]]; then
    export ZEPHYR_SDK_INSTALL_DIR="${MLFORGE_SDK}"
fi

# QEMU ships inside the SDK from 0.17 onwards; the standalone hosttools install is
# the older layout. Without one of these on PATH, 'west build -t run' has no emulator.
for hosttools in \
    "${ZEPHYR_SDK_INSTALL_DIR:-}/hosttools/sysroots/x86_64-pokysdk-linux/usr/bin" \
    "${HOME}/zephyr-hosttools/sysroots/x86_64-pokysdk-linux/usr/bin"
do
    if [[ -d "${hosttools}" ]]; then
        export PATH="${hosttools}:${PATH}"
        break
    fi
done

# LinkServer installs outside PATH, and west reports only 'required program LinkServer
# not found' when it is missing, which does not say where to look.
for linkserver in /usr/local/LinkServer /usr/local/LinkServer_* /opt/nxp/LinkServer*; do
    if [[ -x "${linkserver}/LinkServer" ]]; then
        export PATH="${linkserver}:${PATH}"
        break
    fi
done

if [[ -f "${ZEPHYR_BASE:-}/zephyr-env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ZEPHYR_BASE}/zephyr-env.sh"
fi

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.env"
    set +a
fi

echo "ZephyrMLForge environment activated"
echo "  ZEPHYR_BASE: ${ZEPHYR_BASE:-not found}"
echo "  Zephyr SDK:  ${ZEPHYR_SDK_INSTALL_DIR:-not found}"
echo "  Python:      $(command -v python3)"
echo "  west:        $(west --version 2>/dev/null || echo 'not found')"
echo "  QEMU:        $(command -v qemu-system-arm || echo 'not found')"
echo "  LinkServer:  $(command -v LinkServer || echo 'not found')"
