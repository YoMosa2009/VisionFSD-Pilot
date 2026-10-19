"""Run the setup shell against temporary files and fake privileged commands."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

BASH = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
SOURCE = Path(__file__).resolve().parents[1]
PRELUDE = r"""
export PATH="/usr/bin:/bin:$PATH"
sudo() {
  printf '%s\n' "$*" >>"$CALLS"
  if [[ "$1" == "-n" ]]; then shift; else return 99; fi
  case "$1" in
    true) [[ "${DENY_SUDO:-0}" != 1 ]] ;;
    tee) shift; command tee "$@" ;;
    modprobe) [[ "${BUSY_DRIVER:-0}" != 1 ]] ;;
    udevadm) return 0 ;;
    *) return 98 ;;
  esac
}
lsmod() { echo 'hid_mcp2221 16384 0'; }
export -f sudo lsmod
source "$1" --repair
"""


@unittest.skipUnless(Path(BASH).exists(), "Bash unavailable")
class MCPSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rules = self.root / 'rules'
        self.blacklist = self.root / 'blacklist'
        self.calls = self.root / 'calls'
        script = (SOURCE / 'setup_mcp2221.sh').read_text(encoding='utf-8')
        script = script.replace('/etc/udev/rules.d/99-visionfsd-mcp2221.rules', self.rules.as_posix())
        script = script.replace('/etc/modprobe.d/visionfsd-mcp2221.conf', self.blacklist.as_posix())
        self.script = self.root / 'setup_mcp2221.sh'
        self.script.write_text(script, encoding='utf-8', newline='\n')

    def run_repair(self, **flags):
        return subprocess.run([BASH, '-c', PRELUDE, 'fixture', self.script.as_posix()],
                              env={**os.environ, 'CALLS': self.calls.as_posix(), **flags},
                              capture_output=True, text=True, timeout=10)

    def test_old_marker_does_not_skip_missing_rules_or_loaded_driver(self):
        (self.root / '.mcp2221-system-v1').touch()
        result = self.run_repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('SUBSYSTEM=="hidraw"', self.rules.read_text())
        self.assertEqual(self.blacklist.read_text().strip(), 'blacklist hid_mcp2221')
        calls = self.calls.read_text()
        self.assertIn('-n modprobe -r hid_mcp2221', calls)
        self.assertIn('--action=add --subsystem-match=hidraw', calls)
        self.assertNotIn('apt-get', calls)
        self.assertNotIn('update-initramfs', calls)
        self.assertTrue(all(line.startswith('-n ') for line in calls.splitlines()))

    def test_repeated_repair_does_not_rewrite_correct_rules(self):
        self.assertEqual(self.run_repair().returncode, 0)
        self.calls.write_text('')
        result = self.run_repair()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('tee ', self.calls.read_text())
        self.assertIn('modprobe', self.calls.read_text())

    def test_denied_sudo_does_not_change_files_or_claim_success(self):
        result = self.run_repair(DENY_SUDO='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.rules.exists())
        self.assertFalse((self.root / '.mcp2221-system-v1').exists())
        self.assertIn('noninteractive sudo', result.stderr)

    def test_failed_driver_unload_is_not_reported_as_success(self):
        result = self.run_repair(BUSY_DRIVER='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / '.mcp2221-system-v1').exists())
        self.assertNotIn('conflict cleared', result.stdout)

    def test_boot_repair_timeout_is_bounded_and_runtime_still_starts(self):
        # Real timeout/launcher; fake runtime and startup dependencies, no USB.
        root = self.root
        (root / '.venv/bin').mkdir(parents=True)
        (root / '.venv/bin/python').write_text('#!/usr/bin/env bash\necho RUNTIME_STARTED\n', newline='\n')
        (root / 'auto_update.sh').write_text('exit 0\n', newline='\n')
        (root / 'setup_mcp2221.sh').write_text('sleep 30\n', newline='\n')
        launcher = root / 'run_robot.sh'
        launcher.write_text((SOURCE / 'run_robot.sh').read_text(), newline='\n')
        (root / 'VERSION').write_text('test\n')
        # Skip device-enumeration sleeps without affecting the timeout's child.
        prelude = r"""
export PATH="/usr/bin:/bin:$PATH"
flock() { return 0; }
compgen() { echo /dev/ttyACM0; }
export -f flock compgen
chmod +x "$1/.venv/bin/python"
bash "$1/run_robot.sh"
"""
        result = subprocess.run([BASH, '-c', prelude, 'fixture', root.as_posix()],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        log = (root / 'logs/robot.log').read_text()
        self.assertIn('MCP2221 boot repair incomplete', log)
        self.assertIn('RUNTIME_STARTED', log)


if __name__ == '__main__':
    unittest.main()
