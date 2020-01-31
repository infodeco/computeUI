from setuptools import setup

setup(
    name="admUI",
    version="1.0",
    description="This package provides functions to compute the function UI introduced in by Bertschinger et al. [\"Quantifying unique information\", Entropy 2014, 16(4), 2161-2183].",
    author="Man Foo",
    author_email="foomail@foo.com",
    packages=["admUI"],
    package_dir={"admUI": "python"},
    install_requires=[
        "numpy",
    ],
    extras_require = {
        "dit": ["dit>=1.2.3"]
    }
)