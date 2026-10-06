"""Compatibility shim for older system pip/setuptools installations.

Modern installers use pyproject.toml. Keeping this tiny mirror lets users
bootstrap TeamCache on vendor images whose preinstalled setuptools predates
PEP 621, without changing the runtime package.
"""

from pathlib import Path

from setuptools import find_packages, setup


setup(
    name="teamcache",
    version="0.3.0",
    description="Team-aware prefix-cache admission and cluster routing for SGLang",
    author="Mu Yuan",
    author_email="muyuan@cuhk.edu.hk",
    url="https://github.com/yuanmu97/teamcache-ncomms/tree/main/teamcache",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"teamcache": ["data/*.json"]},
    entry_points={"console_scripts": ["teamcache=teamcache.cli:main"]},
)
