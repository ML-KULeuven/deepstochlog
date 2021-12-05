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
    install_requires=[
        "torch~=1.5.1",
        "torchvision~=0.6.1",
        "numpy~=1.18.1",
        "pandas~=1.2.4",
        "pyparsing~=2.4.7",
        "dgl~=0.6.1",
    ],
)