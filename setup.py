import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepstochlog",
    version="0.0.1",
    author="Thomas Winters, Giuseppe Marra, Robin Manhaeve, Luc De Raedt",
    author_email="firstname.lastname@kuleuven.be",
    description="Neural Stochastic Logic Programming",
    licence="Apache License 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=["deepstochlog", "deepstochlog.*"]),
    python_requires='>=3',
    package_data={'deepstochlog': ['*.pl']},
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "pyparsing",
        "dgl",
    ],
)
