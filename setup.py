import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stoqcompiler",
    version="0.2.0",
    author="Ryan Shaffer",
    author_email="ryan@ryanshaffer.net",
    description="Toolset for stochastic approximate quantum compilation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rmshaffer/stoq-compiler",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
