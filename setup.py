from setuptools import setup

setup(
    name='codegan',
    version='0.1.0',
    packages=['data', 'models', 'train', 'utils'],
    install_requires=['transformers',
                      'numpy',
                      'torch',
                      'tqdm',
                      'sentencepiece',
                      'nltk'
                      ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ]


)