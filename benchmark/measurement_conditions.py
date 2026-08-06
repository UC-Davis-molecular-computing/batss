"""Record the machine conditions a timing measurement was taken under.

Wall-clock benchmarks on a laptop are not comparable across power states. On battery, Windows caps
the processor well below its plugged-in clock, so a number measured on battery cannot be compared
with one measured on AC -- and a battery draining across a long session throttles progressively,
which biases whatever is measured later.

None of the earlier results in this directory recorded any of this, so it is not possible after the
fact to tell which of them are comparable. Every harness should stamp its output with
`snapshot()` so the question is answerable next time.

`comparable(a, b)` gives the check a reader actually wants: whether two result sets were taken under
conditions similar enough to be compared.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Any

_BATTERY_STATUS = {1: "on_battery", 2: "on_ac"}


def _powershell(command: str) -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip()
    except Exception:
        return ""


def snapshot() -> dict[str, Any]:
    """Power source, clock speed and power scheme, as far as the platform will report them."""

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "power_source": "unknown",
        "battery_percent": None,
        "current_clock_mhz": None,
        "max_clock_mhz": None,
        "power_scheme": None,
    }
    if platform.system() != "Windows":
        return info

    battery = _powershell(
        "$b=Get-CimInstance Win32_Battery -EA SilentlyContinue; "
        "if($b){ \"$($b.BatteryStatus)|$($b.EstimatedChargeRemaining)\" } else { 'none|' }")
    if "|" in battery:
        status, charge = battery.split("|", 1)
        try:
            info["power_source"] = _BATTERY_STATUS.get(int(status), f"status_{status}")
        except ValueError:
            info["power_source"] = "no_battery"
        info["battery_percent"] = int(charge) if charge.strip().isdigit() else None

    clocks = _powershell(
        "$c=Get-CimInstance Win32_Processor | Select-Object -First 1; "
        "\"$($c.CurrentClockSpeed)|$($c.MaxClockSpeed)\"")
    if "|" in clocks:
        cur, mx = clocks.split("|", 1)
        info["current_clock_mhz"] = int(cur) if cur.strip().isdigit() else None
        info["max_clock_mhz"] = int(mx) if mx.strip().isdigit() else None

    scheme = _powershell("powercfg /getactivescheme")
    info["power_scheme"] = scheme.strip() or None
    return info


def describe(info: dict[str, Any] | None = None) -> str:
    info = info if info is not None else snapshot()
    parts = [f"power={info.get('power_source')}"]
    if info.get("battery_percent") is not None:
        parts.append(f"charge={info['battery_percent']}%")
    if info.get("current_clock_mhz"):
        parts.append(f"clock={info['current_clock_mhz']}MHz")
    return "  ".join(parts)


def comparable(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, str]:
    """Were these two measurement sets taken under conditions that can be compared?"""

    if a.get("power_source") != b.get("power_source"):
        return False, (f"different power source: {a.get('power_source')} vs "
                       f"{b.get('power_source')} -- the processor clock differs between them, so "
                       f"the timings are not comparable")
    ca, cb = a.get("current_clock_mhz"), b.get("current_clock_mhz")
    if ca and cb and abs(ca - cb) / max(ca, cb) > 0.05:
        return False, f"clock differs by more than 5%: {ca} MHz vs {cb} MHz"
    return True, "conditions match"


if __name__ == "__main__":
    import json
    info = snapshot()
    print(json.dumps(info, indent=2))
    print("\n" + describe(info))
    if info.get("power_source") == "on_battery":
        print("\nWARNING: on battery. The processor is clocked below its plugged-in speed, and a\n"
              "draining battery throttles progressively, which biases whatever is measured later\n"
              "in a long run. Timings taken here cannot be compared with timings taken on AC.")
