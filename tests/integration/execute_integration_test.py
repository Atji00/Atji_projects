import subprocess
import sys

from scoring_model.runtime.paths import ProjectPaths

TESTS_DIR = ProjectPaths().tests / "integration"
REPORTS_DIR = ProjectPaths().tests / "integration_tests_reports"


def execute_test(run_all: bool = True, test_name: str | None = None) -> None:

    REPORTS_DIR.mkdir(exist_ok=True)

    if run_all:

        test_files = sorted(TESTS_DIR.glob("test_*.py"))

    else:

        if not test_name:

            test_name = input("Enter filename to be tested (ex: test_train_pipeline.py) : ").strip()

        test_files = sorted(TESTS_DIR.glob(test_name))

        if not test_files:

            print(f"\nNo files found ! : {test_name}")
            sys.exit(1)

    failed = False

    for test_file in test_files:

        report_name = f"{test_file.stem}.html"
        report_path = REPORTS_DIR / report_name

        print(f"\n{'=' * 70}")
        print(f"Running: {test_file}")
        print(f"Report : {report_path}")
        print(f"{'=' * 70}\n")

        result = subprocess.run(
                                    [
                                        sys.executable,
                                        "-m",
                                        "pytest",
                                        str(test_file),
                                        "--html",
                                        str(report_path),
                                        "--self-contained-html",
                                    ],
                                    check = False
                                )

        if result.returncode != 0:

            failed = True

            print(f"\nFAILED: {test_file}")

    if failed:

        sys.exit(1)

    print("\nAll integration test reports have been generated.")


if __name__ == "__main__":
    execute_test(False)
