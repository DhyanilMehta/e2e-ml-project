from typing import List
from setuptools import find_packages, setup

HYPHEN_E_DOT = "-e ."

def get_requirements(path: str) -> List[str]:
    '''
    This will return a list of requirements fetched from the provided file path
    '''
    with open(path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name="e2e-ml-project",
    version="0.0.1",
    author="Dhyanil Mehta",
    author_email="dhyanilm2399@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)