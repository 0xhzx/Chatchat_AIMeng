from setuptools import setup, find_packages

# just an example

setup(
    name='Chatchat',
    version='1.0.0',
    url='https://github.com/0xhzx/Chatchat_AIMeng',
    author='Zhihan',
    description='Chatchat - A Multifunctional integrated LLM application',
    packages=find_packages(),    
    install_requires=['torch >= 2.2.2'],
)