from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Read requirements.txt and return a clean list of packages.
    """
    requirements: List[str] = []

    with open(file_path, encoding="utf-8") as file_obj:
        requirements = [
            line.strip()
            for line in file_obj
            if line.strip() and not line.startswith("#")
        ]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Aditya Raikar",
    author_email="adityaraikar5555@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)