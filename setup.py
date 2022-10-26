from setuptools import setup
from setuptools import find_packages

setup(
    name='TrioIJK_data_handler',
    version='0.1',
    description='IJK simulations dataloader and visualisation tool',
    author='Yanis Zatout',
    author_email='yanis.zatout@cnrs.fr',
    url='https://github.com/YanisZatout/TrioIJK_data_handler',
    packages=find_packages("datahandling"),
)
