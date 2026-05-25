"""
SmartCleaner - security_scan.py
---------------------------------
Run a full Bandit security scan on the SmartCleaner codebase.

What is Bandit?
  Bandit is a tool that reads your Python code and looks for
  security mistakes. It checks for things like:
    - Passwords written directly in code (hardcoded secrets)
    - Use of unsafe functions that hackers can exploit
    - Insecure random number generation
    - SQL injection risks
    - Unsafe file operations
    - And 100+ other known security issues

How to run:
  python security_scan.py

What you get:
  - A summary of any security issues found
  - Severity level for each issue: LOW, MEDIUM, or HIGH
  - The exact file and line number of each issue
  - A final verdict: PASSED or FAILED

Exit codes:
  0 = No issues found. All clear.
  1 = Issues found. Review and fix before deploying.
"""

import subprocess
import sys
import os
from datetime import datetime


# ── Files and folders to scan ─────────────────────────────────────────────────
SCAN_TARGETS = [
    "src/",
    "api.py",
    "main.py",
    "app.py",
]

# ── Severity threshold ────────────────────────────────────────────────────────
# Only FAIL the scan for MEDIUM and HIGH severity issues.
# LOW severity issues are reported but do not cause a failure.
FAIL_ON_SEVERITY = "medium"


def run_scan():
    """
    Run Bandit security scan on all SmartCleaner source files.
    Print a clear, readable report of any issues found.
    """
    print("=" * 60)
    print("  SmartCleaner — Security Scan")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    all_passed = True

    for target in SCAN_TARGETS:
        # Skip if the target does not exist on this machine
        if not os.path.exists(target):
            print(f"  SKIP  {target} (not found)")
            continue

        print(f"  Scanning: {target}")

        # Run bandit on this target
        result = subprocess.run(
            [
                "bandit",
                "-r",          # recursive — scan all files in folders
                target,
                "-f", "txt",   # plain text output
                "--severity-level", "low",   # report all severities
                "--confidence-level", "low", # report all confidence levels
                "-q",          # quiet — suppress info messages
            ],
            capture_output=True,
            text=True,
        )

        output = result.stdout.strip()

        if "No issues identified" in output:
            print(f"  PASS  {target} — No issues found.")
        else:
            print(f"  WARN  {target}")
            print()
            print(output)
            print()

            # Check if any MEDIUM or HIGH issues exist
            if "Severity: Medium" in output or "Severity: High" in output:
                all_passed = False

    print()
    print("=" * 60)

    if all_passed:
        print("  RESULT: PASSED")
        print("  No medium or high severity security issues found.")
        print("  SmartCleaner code is clean and safe to deploy.")
        print("=" * 60)
        return 0
    else:
        print("  RESULT: FAILED")
        print("  Medium or high severity issues were found.")
        print("  Fix these before deploying to production.")
        print("=" * 60)
        return 1


def print_what_bandit_checks():
    """
    Print a plain English explanation of what Bandit looks for.
    Useful for understanding the scan results.
    """
    checks = [
        ("B101", "Assert statements",          "Assert can be disabled at runtime"),
        ("B105", "Hardcoded passwords",         "Password written directly in code"),
        ("B106", "Hardcoded passwords",         "Password as function argument"),
        ("B107", "Hardcoded passwords",         "Password as default argument"),
        ("B108", "Temp file",                  "Insecure temp file creation"),
        ("B301", "Pickle",                     "Pickle can execute arbitrary code"),
        ("B303", "MD5 hash",                   "MD5 is a weak hash algorithm"),
        ("B311", "Random",                     "Standard random is not secure"),
        ("B404", "Import subprocess",          "Subprocess can be dangerous"),
        ("B501", "SSL",                        "SSL verification disabled"),
        ("B601", "Shell injection",            "Shell=True with variable input"),
        ("B602", "Shell injection",            "Subprocess with shell=True"),
        ("B608", "SQL injection",              "Possible SQL injection"),
    ]

    print()
    print("  What Bandit checks for:")
    print()
    for code, name, risk in checks:
        print(f"    {code}  {name:<25} {risk}")
    print()


if __name__ == "__main__":
    # If user passes --explain flag, show what bandit checks for
    if "--explain" in sys.argv:
        print_what_bandit_checks()

    exit_code = run_scan()
    sys.exit(exit_code)
