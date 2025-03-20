if __name__ == "__main__":
    import sys

    import pytest

    package_name = "bayesiangsn"
    pytest_args = [
        "--cov-config=.coveragerc",
        "--cov=bayesiangsn/core",
        "--cov=bayesiangsn/utils",
        "--verbose",
        "--junitxml=./reports/junit_report.xml",
        "--cov-branch",
        "--new-first",
    ]
    pytest_args = pytest_args + sys.argv[1:]
    sys.exit(pytest.main(pytest_args))
