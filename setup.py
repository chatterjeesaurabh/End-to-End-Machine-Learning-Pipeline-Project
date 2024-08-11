from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='DimondPricePrediction',
    version='1.0.0',
    author='Saurabh Chatterjee',
    author_email='saurabhchatterjee38@gmail.com',
    install_requires=[get_requirements('requirements_dev.txt')],
    packages=find_packages()        # considers only those folders with '__init__.py' file
)