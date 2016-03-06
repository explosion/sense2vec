import pytest


def pytest_addoption(parser):
    parser.addoption("--models", action="store_true",
        help="include tests that require full models")


def pytest_runtest_setup(item):
    for opt in ['models']:
        if opt in item.keywords and not item.config.getoption("--%s" % opt):
            pytest.skip("need --%s option to run" % opt)
