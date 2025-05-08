from setuptools import setup, find_packages

setup(
    name='ctc-algorithms',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy'
    ],
    author='Harshil Chudasama',
    description='A Python package for CTC loss and gradient computation.',
    url='https://github.com/yourusername/ctc-algorithms',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ]
)
