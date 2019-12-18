from setuptools import setup

setup(
    name="infodecom",
    version="1.0",
    description="A useful module",
    author="Man Foo",
    author_email="foomail@foo.com",
    packages=["admUI"],
    package_dir={"admUI": "python"},
    install_requires=[
        "numpy",
        "dit"
    ], #external packages as dependencies
)