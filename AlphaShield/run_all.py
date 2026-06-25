"""AlphaShield supervisor script for environment checks, tests, and recovery logging."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict

import yaml

ROOT = Path(__file__).resolve().parent
REQUIRED_DIRECTORIES = ["config", "src", "tests", "models", "logs"]
REQUIREMENTS = ROOT / "requirements.txt"
ERROR_LOG = ROOT / "logs" / "error_trace.log"


def parse_requirements() -> Dict[str, str]:
    requirements: Dict[str, str] = {}
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "==" in stripped:
            name, version = stripped.split("==", 1)
            requirements[name.lower()] = version
    return requirements


def environment_scan() -> Dict[str, str]:
    for directory in REQUIRED_DIRECTORIES:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    settings_path = ROOT / "config" / "settings.yaml"
    with settings_path.open("r", encoding="utf-8") as handle:
        yaml.safe_load(handle)
    mismatches: Dict[str, str] = {}
    for package, expected in parse_requirements().items():
        distribution_name = "scikit-learn" if package == "scikit-learn" else package
        try:
            installed = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            mismatches[package] = f"missing; expected {expected}"
            continue
        if installed != expected:
            mismatches[package] = f"installed {installed}; expected {expected}"
    return mismatches


def run_tests() -> int:
    command = [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"pytest failed with exit code {completed.returncode}\n{completed.stdout}\n{completed.stderr}")
    return completed.returncode


def recover_from_failure(exc: BaseException) -> None:
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    ERROR_LOG.write_text(trace, encoding="utf-8")
    test_file = ROOT / "tests" / "test_core.py"
    if test_file.exists():
        source = test_file.read_text(encoding="utf-8")
        normalized = source.replace("\r\n", "\n").replace("\t", "    ")
        if normalized != source:
            test_file.write_text(normalized, encoding="utf-8")


def main() -> int:
    mismatches = environment_scan()
    if mismatches:
        print("Environment package mismatches detected:")
        for package, detail in sorted(mismatches.items()):
            print(f"- {package}: {detail}")
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            return run_tests()
        except BaseException as exc:
            recover_from_failure(exc)
            if attempt == max_attempts:
                print(f"Validation failed after {max_attempts} attempts. See {ERROR_LOG}.", file=sys.stderr)
                return 1
            print(f"Validation attempt {attempt} failed; recovery applied and tests will retry.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
