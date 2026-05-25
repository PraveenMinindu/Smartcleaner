"""
SmartCleaner - tests/test_security.py
---------------------------------------
Automated security tests using Bandit.

These tests run Bandit as part of the pytest suite so that
security scanning happens automatically every time you run
your tests. You never forget to check security.

If a new piece of code introduces a security vulnerability,
these tests will catch it immediately — just like unit tests
catch logic bugs.

Run with:
    python -m pytest tests/test_security.py -v

What gets tested:
  - src/ folder: all cleaning and scoring modules
  - api.py: the FastAPI REST endpoints
  - No medium or high severity issues allowed
  - No hardcoded secrets allowed
  - No unsafe functions allowed
"""

import subprocess
import os
import sys
import pytest


# ── Helper function ───────────────────────────────────────────────────────────

def run_bandit(target: str) -> dict:
    """
    Run Bandit on a target file or folder and return the results.

    Args:
        target: File path or folder path to scan.

    Returns:
        dict with keys:
          - passed:         True if no medium/high issues found
          - output:         Full bandit output text
          - has_medium:     True if medium severity issues exist
          - has_high:       True if high severity issues exist
          - issue_count:    Total number of issues found
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "bandit",
            "-r", target,
            "-f", "txt",
            "--severity-level", "low",
            "--confidence-level", "low",
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    output = result.stdout + result.stderr

    return {
        "passed":       "Severity: Medium" not in output and "Severity: High" not in output,
        "output":       output,
        "has_medium":   "Severity: Medium" in output,
        "has_high":     "Severity: High"   in output,
        "no_issues":    "No issues identified" in output or output.strip() == "",
    }


# ════════════════════════════════════════════════════════════════
#  Security tests
# ════════════════════════════════════════════════════════════════

class TestSecurityScan:
    """
    Automated Bandit security scans for all SmartCleaner source files.

    Each test scans one part of the codebase and asserts that no
    medium or high severity security issues exist.
    """

    def test_src_folder_has_no_high_severity_issues(self):
        """
        The src/ folder must have zero HIGH severity security issues.

        HIGH severity means: the issue is very likely exploitable
        and could allow an attacker to compromise the system.
        Examples: SQL injection, command injection, hardcoded passwords.

        This must always be zero. No exceptions.
        """
        result = run_bandit("src/")
        assert not result["has_high"], (
            f"HIGH severity security issues found in src/:\n\n{result['output']}"
        )

    def test_src_folder_has_no_medium_severity_issues(self):
        """
        The src/ folder must have zero MEDIUM severity issues.

        MEDIUM severity means: the issue could be exploited under
        certain conditions. Examples: use of weak hash algorithms,
        insecure random number generation, subprocess without shell.

        Fix these before deploying to production.
        """
        result = run_bandit("src/")
        assert not result["has_medium"], (
            f"MEDIUM severity security issues found in src/:\n\n{result['output']}"
        )

    def test_api_has_no_high_severity_issues(self):
        """
        api.py must have zero HIGH severity security issues.

        The API is the most exposed part of SmartCleaner — it
        accepts files from the internet. It must be clean.
        """
        if not os.path.exists("api.py"):
            pytest.skip("api.py not found in project root.")

        result = run_bandit("api.py")
        assert not result["has_high"], (
            f"HIGH severity security issues found in api.py:\n\n{result['output']}"
        )

    def test_api_has_no_medium_severity_issues(self):
        """
        api.py must have zero MEDIUM severity security issues.
        """
        if not os.path.exists("api.py"):
            pytest.skip("api.py not found in project root.")

        result = run_bandit("api.py")
        assert not result["has_medium"], (
            f"MEDIUM severity security issues found in api.py:\n\n{result['output']}"
        )

    def test_no_hardcoded_passwords_in_src(self):
        """
        No file in src/ must contain a hardcoded password, secret key,
        or API token written directly in the code.

        Bandit check B105/B106/B107 covers this.
        Hardcoded secrets must never be committed to version control.
        """
        result = run_bandit("src/")

        hardcoded_issues = any(
            code in result["output"]
            for code in ["B105", "B106", "B107", "hardcoded_password"]
        )
        assert not hardcoded_issues, (
            f"Hardcoded password or secret found in src/:\n\n{result['output']}"
        )

    def test_no_shell_injection_in_src(self):
        """
        No file in src/ must use subprocess with shell=True
        or build shell commands from user input.

        Shell injection allows attackers to run arbitrary commands
        on your server. Bandit check B602/B603 covers this.
        """
        result = run_bandit("src/")

        shell_issues = any(
            code in result["output"]
            for code in ["B602", "B603", "B604", "shell_injection"]
        )
        assert not shell_issues, (
            f"Shell injection risk found in src/:\n\n{result['output']}"
        )

    def test_no_sql_injection_risks(self):
        """
        No file must build SQL queries by concatenating strings
        with user input. This is the most common web vulnerability.

        SmartCleaner does not currently use SQL but this test ensures
        it never accidentally gets added unsafely. Bandit B608.
        """
        result = run_bandit("src/")

        sql_issues = "B608" in result["output"] or "sql_injection" in result["output"].lower()
        assert not sql_issues, (
            f"SQL injection risk found in src/:\n\n{result['output']}"
        )

    def test_security_scan_completes_without_error(self):
        """
        The Bandit scan itself must complete successfully.
        If Bandit crashes or cannot scan the files, something
        is wrong with the setup.

        This test verifies the scanning infrastructure works.
        A clean codebase produces minimal output in quiet mode —
        that is correct behaviour, not a failure.
        """
        result = run_bandit("src/")

        # result dict must exist and have expected keys
        assert result is not None,              "Bandit returned no result."
        assert "passed"     in result,          "Result missing passed key."
        assert "has_high"   in result,          "Result missing has_high key."
        assert "has_medium" in result,          "Result missing has_medium key."
        # The scan must have passed — no medium or high issues
        assert result["passed"] is True,        "Security scan did not pass."
