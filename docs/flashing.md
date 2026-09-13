<!--
  Copyright (C) 2025 Rufilla Ltd
-->

# Flashing the FRDM-MCXN947

Developed against the NXP FRDM-MCXN947 (Cortex-M33, 512 KB RAM, 2 MB flash) with its
MCU-LINK CMSIS-DAP probe. Any Zephyr board carrying the TFLite Micro module should work:
set `hardware.board`, & `execution_environment.flash_runner` if the board's default
runner is not installed.

## LinkServer

NXP LinkServer is the board's default runner & the one that works here:

```bash
sudo apt-get install -y libusb-1.0-0-dev whiptail dfu-util usbutils hwdata
curl -O https://www.nxp.com/lgfiles/updates/mcuxpresso/LinkServer_25.7.33.x86_64.deb.bin
chmod +x LinkServer_25.7.33.x86_64.deb.bin
sudo ./LinkServer_25.7.33.x86_64.deb.bin acceptLicense skipIdeSelect
```

Where:

- `acceptLicense skipIdeSelect` is NXP's non-interactive install; it accepts their licence
  agreement on your behalf
- It installs to `/usr/local/LinkServer/`, which is **not** added to `PATH` — add it, or
  `west flash` reports `required program LinkServer not found`. `activate.sh` adds it when
  it finds it there.
- `LinkServer probes` lists attached probes & is the quickest check that it works

Then set `execution_environment.flash_runner: linkserver`.

## pyocd

Lighter in principle (`pip install pyocd && pyocd pack install MCXN947`), but on this
board it reached the part & then faulted in the pack's flash algorithm — `target was not
halted as expected after calling flash algorithm routine (IPSR=3)` — under both sector &
chip erase, & failed earlier still with `connect_mode=under-reset`. LinkServer flashed
first time.

## The serial link

After flashing, the host waits for the USB CDC endpoint to come back & then talks to the
firmware over it: `PING`/`PONG` to check the command loop, `BENCH` for a fresh latency
measurement, & `INFER <values>` for one prediction. The endpoint may return under a
different name than `execution_environment.serial_port`; any other `/dev/ttyACM*` is
accepted as a fallback, with a warning, which would pick the wrong device if two boards
were attached.
