import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="config-state",
    version="1.0.0",
    author="Nicolas Pinchaud",
    author_email="nicolas.pinchaud@gmail.com",
    description="config-state is a Python module for implementing classes with "
    "the `ConfigState` pattern. The pattern augments classes with  "
    "configuration and state semantics enabling their "
    "instantiation through configuration files together with an "
    "improved control over their (de)serialization logics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nicolaspi/config-state",
    packages=setuptools.find_packages(exclude=('tests', 'examples')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=['pyyaml'],
)
