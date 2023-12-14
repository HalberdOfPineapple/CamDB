from setuptools import setup, find_packages

def load_requirements(filename='requirements.txt'):
    with open(filename, 'r') as file:
        return [line.rstrip() for line in file.read().splitlines()]

setup(
    name='camtune',
    version='0.1',
    packages=find_packages(),
    install_requires=load_requirements(),
    author='Wenxuan Li',
    author_email='wl446@cam.ac.uk',
    description='LA-MCTS-based DBMS configuration tuning framework',
)
