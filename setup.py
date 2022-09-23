from setuptools import find_packages, setup

setup(
    name="stock_tools",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6,<4.0",
    setup_requires=["setuptools_scm"],
    use_scm_version={"version_scheme": "python-simplified-semver"},
)
