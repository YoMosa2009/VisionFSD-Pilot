from __future__ import annotations

import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_update import BootUpdater


class BootUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / "origin"
        self.install = self.root / "pi"
        self.origin.mkdir()
        self.git(self.origin, "init", "-b", "codex/pi3b-runtime")
        self.git(self.origin, "config", "user.email", "test@example.invalid")
        self.git(self.origin, "config", "user.name", "Test")
        self.git(self.origin, "config", "core.autocrlf", "false")
        (self.origin / "pi3b").mkdir()
        self.write("VERSION", "old\n")
        self.write("requirements.txt", "unchanged\n")
        self.write("robot_autonomy.py", "VERSION = 'old'\n")
        self.commit()
        self.git(self.root, "-c", "core.autocrlf=false", "clone", str(self.origin), str(self.install))
        self.git(self.install, "config", "core.autocrlf", "false")
        self.updater = BootUpdater(self.install)
        self.old = self.git(self.install, "rev-parse", "HEAD").strip()
        self.write("VERSION", "new\n")
        self.write("robot_autonomy.py", "VERSION = 'new'\n")
        self.commit()
        self.target = self.git(self.origin, "rev-parse", "HEAD").strip()

    def git(self, root, *args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode()

    def write(self, name, content):
        (self.origin / "pi3b" / name).write_text(content, newline="\n")

    def commit(self):
        self.git(self.origin, "add", "pi3b")
        self.git(self.origin, "commit", "-m", "fixture")

    def test_actual_fetch_checkout_next_boot_and_missing_ref(self):
        self.assertEqual(self.updater.run(), 2)
        self.assertEqual(self.git(self.install, "rev-parse", "HEAD").strip(), self.target)
        self.assertEqual((self.install / "pi3b/VERSION").read_text().strip(), "new")
        result = subprocess.check_output([sys.executable, "-c", "import robot_autonomy; print(robot_autonomy.VERSION)"], cwd=self.install / "pi3b")
        self.assertEqual(result.strip(), b"new")
        self.assertEqual(self.updater.run(), 0)
        self.assertIn("UP TO DATE", (self.install / "pi3b/logs/update-status.txt").read_text())

    def test_boot_launcher_reexec_runs_fetched_version(self):
        bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        if not Path(bash).exists():
            self.skipTest("Bash unavailable")
        source = Path(__file__).resolve().parents[1]
        # Run the actual launcher/updater; replace only hardware/runtime and
        # Linux utilities absent on Windows. Real Git fetch and checkout run.
        for name in ("run_robot.sh", "auto_update.sh", "robot_update.py"):
            self.write(name, (source / name).read_text())
        self.commit()
        self.git(self.install, "fetch", "origin")
        self.git(self.install, "checkout", "--detach", "FETCH_HEAD")
        self.write("VERSION", "boot-new\n")
        self.write("robot_autonomy.py", "print('BOOT_RUNTIME_NEW', flush=True)\n")
        self.commit()
        venv = self.install / "pi3b/.venv/bin"
        venv.mkdir(parents=True)
        python = venv / "python"
        python.write_text("#!/usr/bin/env bash\nexec '" + sys.executable.replace("\\", "/") + "' \"$@\"\n", newline="\n")
        python.chmod(0o755)
        fakebin = self.root / "bin"
        fakebin.mkdir()
        for name, body in (("flock", "exit 0"), ("sleep", "exit 0")):
            path = fakebin / name
            path.write_text("#!/usr/bin/env bash\n" + body + "\n", newline="\n")
            path.chmod(0o755)
        env = {**os.environ, "PATH": str(fakebin) + os.pathsep + os.environ["PATH"]}
        result = subprocess.run([bash, str(self.install / "pi3b/run_robot.sh")], env=env,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        log = (self.install / "pi3b/logs/robot.log").read_text()
        self.assertIn("BOOT_RUNTIME_NEW", log)
        self.assertIn("version: boot-new", log)
        self.assertEqual(log.count("=== VisionFSD auto-update:"), 1)
        self.assertIn("auto-update exit status: 2", log)

    def test_staged_and_unstaged_edits_are_preserved(self):
        path = self.install / "pi3b/robot_autonomy.py"
        path.write_text("local = True\n")
        for staged in (False, True):
            if staged:
                self.git(self.install, "add", "pi3b/robot_autonomy.py")
            self.assertEqual(self.updater.run(), 0)
            self.assertEqual(path.read_text(), "local = True\n")
            self.assertEqual(self.git(self.install, "rev-parse", "HEAD").strip(), self.old)

    def test_permission_only_worktree_difference_does_not_block(self):
        # Force an executable index entry, then strip the worktree mode. This
        # exercises Git's mode comparison even on Windows without POSIX chmod.
        self.git(self.install, "config", "user.email", "test@example.invalid")
        self.git(self.install, "config", "user.name", "Test")
        self.git(self.install, "update-index", "--chmod=+x", "pi3b/robot_autonomy.py")
        self.git(self.install, "commit", "-m", "mode fixture")
        os.chmod(self.install / "pi3b/robot_autonomy.py", 0o644)
        self.git(self.install, "config", "core.filemode", "true")
        self.assertTrue(self.git(self.install, "diff", "--name-only").strip())
        self.assertEqual(self.updater.run(), 2)

    def test_network_retries_then_installs(self):
        original = self.updater.git
        attempts = []
        def flaky(*args, **kwargs):
            if args[0] == "fetch":
                attempts.append(1)
                if len(attempts) < 3:
                    raise subprocess.TimeoutExpired("git fetch", 0.01)
            return original(*args, **kwargs)
        with patch.object(self.updater, "git", side_effect=flaky), patch("robot_update.time.sleep"):
            self.assertEqual(self.updater.run(), 2)
        self.assertEqual(len(attempts), 3)

    def test_offline_retains_old_release(self):
        self.git(self.install, "remote", "set-url", "origin", str(self.root / "missing"))
        with patch("robot_update.time.sleep"):
            self.assertEqual(self.updater.run(), 0)
        self.assertEqual(self.git(self.install, "rev-parse", "HEAD").strip(), self.old)

    def test_syntax_error_never_replaces_installed_files(self):
        self.write("robot_autonomy.py", "broken (\n")
        self.commit()
        self.assertEqual(self.updater.run(), 0)
        self.assertEqual(self.git(self.install, "rev-parse", "HEAD").strip(), self.old)

    def test_requirements_change_requires_manual_update(self):
        self.write("requirements.txt", "new dependency\n")
        self.commit()
        self.assertEqual(self.updater.run(), 0)
        self.assertEqual(self.git(self.install, "rev-parse", "HEAD").strip(), self.old)

    def test_checkout_conflict_preserves_untracked_file(self):
        self.write("new.py", "x = 1\n")
        self.commit()
        path = self.install / "pi3b/new.py"
        path.write_text("precious\n")
        self.assertEqual(self.updater.run(), 0)
        self.assertEqual(path.read_text(), "precious\n")
        self.assertEqual(self.git(self.install, "rev-parse", "HEAD").strip(), self.old)

    def test_post_checkout_failure_rolls_back(self):
        self.git(self.origin, "rm", "pi3b/VERSION")
        self.git(self.origin, "commit", "-m", "missing version")
        self.assertEqual(self.updater.run(), 0)
        self.assertEqual(self.git(self.install, "rev-parse", "HEAD").strip(), self.old)

    def test_rollback_failure_withholds_runtime(self):
        self.git(self.origin, "rm", "pi3b/VERSION")
        self.git(self.origin, "commit", "-m", "missing version")
        original = self.updater.git
        def fail_rollback(*args, **kwargs):
            if args == ("checkout", "--detach", self.old):
                raise OSError("disk error")
            return original(*args, **kwargs)
        with patch.object(self.updater, "git", side_effect=fail_rollback):
            self.assertEqual(self.updater.run(), 1)


if __name__ == "__main__":
    unittest.main()
