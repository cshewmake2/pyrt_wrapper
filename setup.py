import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyrt-wrapper",
    version="0.0.1",
    author="James Darby, Christian Shewmake",
    author_email="cshewmake2@gmail.com",
    description="Real-time midi listener callback interface.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/james-julius/pyrt-wrapper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
