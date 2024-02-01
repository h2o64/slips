import setuptools

with open("requirements.txt", "r") as file:
    requirements = [line.strip() for line in file]

setuptools.setup(
    name="slips",
    version="0.0.1",
    author_email="",
    description="",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
