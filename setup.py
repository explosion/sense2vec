#!/usr/bin/env python
from __future__ import unicode_literals

import os
import io
from setuptools import setup, find_packages


def setup_package():
    package_name = "sense2vec"
    root = os.path.abspath(os.path.dirname(__file__))
    # Read in package meta from about.py
    about_path = os.path.join(root, package_name, "about.py")
    with io.open(about_path, encoding="utf8") as f:
        about = {}
        exec(f.read(), about)

    setup(name="sense2vec", version=about["__version__"], packages=find_packages())


if __name__ == "__main__":
    setup_package()
