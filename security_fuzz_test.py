"""
SmartCleaner - security_fuzz_test.py
--------------------------------------
Sends malicious and broken inputs to the API and checks
that the server never crashes (never returns 500).

Run with:
    python security_fuzz_test.py

Make sure API is running first:
    uvicorn api:app --reload
"""

import requests
import json
from datetime import datetime

API_URL = "http://localhost:8000"
RESULTS = []


def test(name: str, response, expect_no_crash: bool = True):
    passed = response.status_code != 500
    RESULTS.append({
        "test":   name,
        "status": "PASS" if passed else "FAIL",
        "code":   response.status_code,
    })
    icon = "OK  " if passed else "FAIL"
    print(f"  {icon} [{response.status_code}] {name}")


print("\nSmartCleaner API Fuzzer")
print("=" * 55)
print(f"Target: {API_URL}")
print(f"Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 55)

# Check server is running
try:
    requests.get(f"{API_URL}/health", timeout=3)
except Exception:
    print("\nERROR: API is not running.")
    print("Start it with: uvicorn api:app --reload")
    exit(1)

# ── Category 1: Wrong file types ──────────────────────────────
print("\nCategory 1 — Wrong file types")

r = requests.post(f"{API_URL}/clean",
    files={"file": ("test.txt", b"hello world", "text/plain")})
test("Plain text file", r)

r = requests.post(f"{API_URL}/clean",
    files={"file": ("virus.exe", b"MZ\x90\x00" * 100, "text/csv")})
test("EXE renamed as CSV", r)

r = requests.post(f"{API_URL}/clean",
    files={"file": ("image.csv", b"\xff\xd8\xff\xe0" * 100, "text/csv")})
test("JPEG binary as CSV", r)

r = requests.post(f"{API_URL}/clean",
    files={"file": ("empty.csv", b"", "text/csv")})
test("Empty file", r)

r = requests.post(f"{API_URL}/clean",
    files={"file": ("noext", b"name,age\nalice,25", "text/csv")})
test("File with no extension", r)

# ── Category 2: Malicious CSV content ────────────────────────
print("\nCategory 2 — Malicious content inside CSV")

sql = b"name,age\n'; DROP TABLE users; --,25\nOR 1=1,30"
r = requests.post(f"{API_URL}/clean",
    files={"file": ("sql.csv", sql, "text/csv")})
test("SQL injection in CSV", r)

xss = b"name,age\n<script>alert('xss')</script>,25"
r = requests.post(f"{API_URL}/clean",
    files={"file": ("xss.csv", xss, "text/csv")})
test("XSS payload in CSV", r)

null_bytes = b"name,age\nali\x00ce,25\nbob,30"
r = requests.post(f"{API_URL}/clean",
    files={"file": ("null.csv", null_bytes, "text/csv")})
test("Null bytes in CSV", r)

long_str = "A" * 100000
long_payload = f"name,age\n{long_str},25".encode()
r = requests.post(f"{API_URL}/clean",
    files={"file": ("long.csv", long_payload, "text/csv")})
test("100000 character cell value", r)

path_traversal = b"name\nalice"
r = requests.post(f"{API_URL}/clean",
    files={"file": ("../../etc/passwd.csv", path_traversal, "text/csv")})
test("Path traversal in filename", r)

unicode_data = "name,age\n田中,25\n鈴木,30".encode("utf-8")
r = requests.post(f"{API_URL}/clean",
    files={"file": ("unicode.csv", unicode_data, "text/csv")})
test("Unicode characters in CSV", r)

emoji_data = "name,age\n😀Alice,25\n🔥Bob,30".encode("utf-8")
r = requests.post(f"{API_URL}/clean",
    files={"file": ("emoji.csv", emoji_data, "text/csv")})
test("Emoji in CSV content", r)

# ── Category 3: Broken CSV structure ─────────────────────────
print("\nCategory 3 — Broken CSV structure")

r = requests.post(f"{API_URL}/clean",
    files={"file": ("noheader.csv", b"alice,25\nbob,30", "text/csv")})
test("CSV with no header row", r)

r = requests.post(f"{API_URL}/clean",
    files={"file": ("onlyheader.csv", b"name,age,salary", "text/csv")})
test("CSV with headers only", r)

r = requests.post(f"{API_URL}/clean",
    files={"file": ("broken.csv", b"a,b,c\n1,2\n3,4,5,6", "text/csv")})
test("Inconsistent column count", r)

r = requests.post(f"{API_URL}/clean",
    files={"file": ("commas.csv", b",,,\n,,,\n,,,", "text/csv")})
test("Only commas no data", r)

r = requests.post(f"{API_URL}/clean",
    files={"file": ("semi.csv", b"name;age\nalice;25", "text/csv")})
test("Semicolon separated file", r)

# ── Category 4: Large data ────────────────────────────────────
print("\nCategory 4 — Large data")

many_cols = ",".join([f"col{i}" for i in range(500)])
many_data = ",".join(["value"] * 500)
r = requests.post(f"{API_URL}/clean",
    files={"file": ("cols.csv", f"{many_cols}\n{many_data}".encode(), "text/csv")})
test("CSV with 500 columns", r)

rows = "name,age\n" + "alice,25\n" * 10000
r = requests.post(f"{API_URL}/clean",
    files={"file": ("rows.csv", rows.encode(), "text/csv")})
test("CSV with 10000 rows", r)

# ── Category 5: Wrong endpoints and methods ───────────────────
print("\nCategory 5 — Wrong endpoints and methods")

r = requests.get(f"{API_URL}/clean")
test("GET on POST /clean endpoint", r)

r = requests.post(f"{API_URL}/clean",
    files={"wrong_field": ("test.csv", b"name\nalice", "text/csv")})
test("Wrong form field name", r)

r = requests.post(f"{API_URL}/clean")
test("No file attached", r)

r = requests.get(f"{API_URL}/does_not_exist")
test("Non-existent endpoint", r)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("RESULTS SUMMARY")
print("=" * 55)

passed = sum(1 for r in RESULTS if r["status"] == "PASS")
failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
total  = len(RESULTS)

print(f"  Total : {total}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print()

if failed > 0:
    print("FAILED TESTS (server returned 500 — these are crashes):")
    for r in RESULTS:
        if r["status"] == "FAIL":
            print(f"  - {r['test']} (got {r['code']})")
else:
    print("All tests passed.")
    print("SmartCleaner handled every malicious input gracefully.")

print("=" * 55)

# Save report
report = {
    "target":    API_URL,
    "timestamp": datetime.now().isoformat(),
    "total":     total,
    "passed":    passed,
    "failed":    failed,
    "results":   RESULTS,
}
with open("fuzz_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nFull report saved to: fuzz_report.json")