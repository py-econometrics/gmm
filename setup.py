from setuptools import setup, find_packages

# Function to read the contents of the requirements.txt file
def read_requirements():
    with open('requirements.txt', 'r') as req:
        return req.read().splitlines()

setup(
    name='gmm',
    version='0.1',
    packages=find_packages(),
    install_requires=read_requirements(),
    # Additional metadata about your package
    author='Apoorva Lal',
    author_email='lal.apoorva@gmail.com',
    description='Estimation of statistical parameters defined as solutions to moment conditions',
    license='MIT',
    keywords='gmm, statistics, econometrics',
    url='https://github.com/apoorvalal/gmm',
)
