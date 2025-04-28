from setuptools import setup, find_packages

setup(
    name='llm_bench',
    version='0.1.0',
    description='A description of what llm_bench does',
    author='Your Name',
    packages=find_packages(include=['llm_bench',]),
    install_requires=[
        # List your package dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)