from setuptools import setup, find_packages

setup(
    name='code-gan',
    version='0.1.0',
    packages=find_packages(include=['models', 'train', 'utils', 'data']),
    install_requires=[
            'torch',
            'tqdm',
            'transformers',
            'numpy'
        ]
)