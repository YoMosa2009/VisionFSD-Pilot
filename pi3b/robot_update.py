"""Bounded boot update. Never stash local edits or install packages at boot."""
from __future__ import annotations

import datetime
import os
from pathlib import Path
import subprocess
import sys
import time


class BootUpdater:
    def __init__(self, root: Path):
        self.root = root
        self.pi = root / "pi3b"

    def git(self, *args, timeout=15):
        # Installation historically chmod'ed 100644 launchers. Ignore ONLY
        # working-tree executable bits, never source or staged content changes.
        return subprocess.run(
            ["git", "-c", "core.filemode=false", "-C", str(self.root), *args],
            check=True, capture_output=True, timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        ).stdout

    def report(self, message):
        print("auto-update: " + message, flush=True)
        try:
            log = self.pi / "logs"
            log.mkdir(exist_ok=True)
            (log / "update-status.txt").write_text(
                datetime.datetime.now(datetime.timezone.utc).isoformat()
                + " " + message + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"auto-update: cannot save status: {exc}", flush=True)

    def run(self):
        if (self.pi / "logs/update-blocked").exists():
            self.report("FAILED previous rollback; run update.sh manually before starting")
            return 1
        if os.environ.get("VISIONFSD_AUTO_UPDATE", "1") == "0":
            self.report("disabled")
            return 0
        old = None
        checked_out = False
        try:
            ref_file = self.pi / ".install-ref"
            ref = ref_file.read_text().strip() if ref_file.exists() else "codex/pi3b-runtime"
            if not ref or ref.startswith("-") or any(c.isspace() for c in ref):
                raise ValueError("invalid installation ref; run update.sh manually")
            old = self.git("rev-parse", "HEAD").decode().strip()
            # HEAD covers staged as well as unstaged edits. Checking all tracked
            # paths also prevents a checkout from disturbing unrelated changes.
            if self.git("diff", "HEAD", "--name-only").strip():
                self.report(f"SKIPPED local tracked edits; installed={old}; run update.sh manually")
                return 0
            timeout = min(30.0, max(1.0, float(os.environ.get("VISIONFSD_AUTO_UPDATE_TIMEOUT_S", "12"))))
            for attempt in range(3):
                try:
                    self.git("fetch", "--depth", "1", "origin", ref, timeout=timeout)
                    break
                except (subprocess.SubprocessError, OSError):
                    if attempt == 2:
                        raise
                    self.report(f"network check failed; retry {attempt + 2}/3")
                    time.sleep(2)
            target = self.git("rev-parse", "FETCH_HEAD").decode().strip()
            if target == old:
                self.report(f"UP TO DATE installed={old} ref={ref}")
                return 0
            if self.git("show", f"{target}:pi3b/requirements.txt") != (self.pi / "requirements.txt").read_bytes():
                self.report(f"MANUAL UPDATE REQUIRED dependencies changed; installed={old} target={target}")
                return 0
            # Compile candidate Python without importing it or touching installed
            # files; syntax errors must not replace the working installation.
            names = self.git("ls-tree", "-r", "--name-only", target, "pi3b").decode().splitlines()
            for name in names:
                if name.endswith(".py"):
                    compile(self.git("show", f"{target}:{name}"), name, "exec")
            checked_out = True
            self.git("checkout", "--detach", target, timeout=30)
            if self.git("rev-parse", "HEAD").decode().strip() != target:
                raise RuntimeError("checked-out commit differs from fetched commit")
            ref_file.write_text(ref + "\n")
            version = (self.pi / "VERSION").read_text().strip()
            self.report(f"UPDATED version={version} installed={target} ref={ref}")
            return 2
        except (subprocess.SubprocessError, OSError, ValueError, SyntaxError, RuntimeError) as exc:
            detail = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                detail += " " + exc.stderr.decode(errors="replace").strip()
            if checked_out and old:
                try:
                    self.git("checkout", "--detach", old, timeout=30)
                    self.report(f"ROLLED BACK installed={old}; {detail}")
                    return 0
                except (subprocess.SubprocessError, OSError) as rollback_error:
                    (self.pi / "logs").mkdir(exist_ok=True)
                    (self.pi / "logs/update-blocked").write_text(str(rollback_error))
                    self.report(f"FAILED rollback; runtime withheld: {rollback_error}")
                    return 1
            self.report(f"SKIPPED installed={old or 'unknown'}; {detail}")
            return 0


if __name__ == "__main__":
    sys.exit(BootUpdater(Path(__file__).resolve().parents[1]).run())
