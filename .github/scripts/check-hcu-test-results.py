#!/usr/bin/env python3
"""Allow only the HCU failures observed and reviewed in the baseline run."""

import re
import sys
from pathlib import Path


def node_id(raw):
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    marker = raw.find("test/")
    return raw[marker:] if marker >= 0 else raw


def main():
    log_path, xml_path, allowlist_path, status = sys.argv[1:]
    status = int(status)
    log = Path(log_path).read_text(encoding="utf-8", errors="replace")
    if not Path(xml_path).is_file():
        raise SystemExit("pytest did not produce a JUnit report")
    if status == 0:
        print("HCU tests passed")
        return
    if status != 1:
        raise SystemExit(f"pytest exited with infrastructure/collection status {status}")

    expected = {
        line.strip()
        for line in Path(allowlist_path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    failed = [node_id(match) for match in re.findall(r"(?m)^FAILED\s+(\S+)", log)]
    errors = re.findall(r"(?m)^ERROR\s+(.+)", log)
    summary = re.search(r"(?m)^=+\s+(\d+) failed(?:,|\s+in)", log)
    if errors or not summary or len(failed) != int(summary.group(1)):
        raise SystemExit("pytest failed without a complete, recognized failure summary")
    unexpected = sorted(set(failed) - expected)
    if unexpected:
        raise SystemExit("New HCU test failures:\n" + "\n".join(unexpected))
    print(f"Only {len(failed)} reviewed HCU baseline failures remain; no new failures")


if __name__ == "__main__":
    main()
