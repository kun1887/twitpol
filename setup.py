from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()

requirements = [x.strip() for x in content]

setup(name="twitpol", packages=find_packages(), install_requires=requirements)
