"""Parsing of build output and firmware console output."""

from __future__ import annotations

from pipeline.runners.qemu import parse_arena_used, parse_device_error, parse_latency_ms
from pipeline.runners.zephyr import ZephyrRunner

# Emitted by west after linking, copied from a frdm_mcxn947 build.
BOARD_BUILD_LOG = """
[353/353] Linking CXX executable zephyr/zephyr.elf
Memory region         Used Size  Region Size  %age Used
           FLASH:       84096 B         2 MB      4.01%
             RAM:       53708 B       320 KB     16.39%
           SRAMX:           0 B        96 KB      0.00%
        IDT_LIST:           0 B        32 KB      0.00%
"""

# One METRICS block as the firmware prints it.
METRICS_OUTPUT = """
STATUS=ready
METRICS_START
latency_ns=35462
latency_us=35
latency_ms=0.035462
arena_used=772
num_runs=100
output_val=0.482906
METRICS_END
Inference complete
"""

READELF_PROGRAM_HEADERS = """
Program Headers:
  Type           Offset   VirtAddr   PhysAddr   FileSiz MemSiz  Flg Align
  LOAD           0x0000f8 0x10000000 0x10000000 0x0147f0 0x0147f0 RE  0x8
  LOAD           0x0148e8 0x30000000 0x100147f0 0x000420 0x000d1c RW  0x8

 Section to Segment mapping:
"""


def test_reads_flash_and_ram_from_the_build_summary():
    assert ZephyrRunner._parse_memory_regions(BOARD_BUILD_LOG) == (82, 52)


def test_ignores_regions_that_are_neither_flash_nor_ram():
    # IDT_LIST and SRAMX are present in the log; neither contributes.
    flash_kb, ram_kb = ZephyrRunner._parse_memory_regions(BOARD_BUILD_LOG)

    assert flash_kb == 82
    assert ram_kb == 52


def test_reports_no_memory_when_the_build_printed_no_summary():
    assert ZephyrRunner._parse_memory_regions("ninja: no work to do.\n") == (None, None)


def test_counts_an_initialised_data_segment_against_both_memories():
    # The second segment loads from flash and lives in RAM, so it counts twice.
    flash_bytes, ram_bytes = ZephyrRunner._parse_stat_file(READELF_PROGRAM_HEADERS)

    assert flash_bytes == 0x147F0 + 0x420
    assert ram_bytes == 0xD1C


def test_prefers_nanoseconds_for_the_latency():
    assert parse_latency_ms(METRICS_OUTPUT) == 0.035462


def test_falls_back_to_milliseconds_when_nanoseconds_are_absent():
    assert parse_latency_ms("latency_ms=0.557176\n") == 0.557176


def test_falls_back_to_microseconds_when_only_they_are_present():
    assert parse_latency_ms("latency_us=250\n") == 0.25


def test_treats_a_zero_latency_as_unmeasured():
    # A target without a cycle counter reports zero, which would otherwise pass
    # every latency constraint.
    assert parse_latency_ms("latency_ns=0\nlatency_us=0\nlatency_ms=0.000000\n") is None


def test_reports_no_latency_when_the_block_is_missing():
    assert parse_latency_ms("STATUS=ready\n") is None


def test_reads_the_arena_bytes():
    assert parse_arena_used(METRICS_OUTPUT) == 772


def test_reports_no_arena_when_absent():
    assert parse_arena_used("STATUS=ready\n") is None


def test_surfaces_the_firmware_failure_reason():
    output = "STATUS=error reason=allocate_tensors arena_size=8192\n"

    assert parse_device_error(output) == "reason=allocate_tensors arena_size=8192"


def test_reports_no_device_error_on_a_healthy_boot():
    assert parse_device_error(METRICS_OUTPUT) is None
