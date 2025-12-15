"""
Setup script for PythoC - Python DSL to LLVM IR compiler
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read version from __init__.py
def read_version():
    init_path = os.path.join(os.path.dirname(__file__), 'pythoc', '__init__.py')
    with open(init_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '0.2.1'

setup(
    name='pythoc',
    version=read_version(),
    author='PythoC Compiler Team',
    description='PythoC: A Python DSL compiler that maps statically-typed Python subset to LLVM IR, providing C-equivalent capabilities with Python syntax',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/1flei/PythoC',
    project_urls={
        'Bug Tracker': 'https://github.com/1flei/PythoC/issues',
        'Documentation': 'https://github.com/1flei/PythoC/blob/master/README.md',
        'Source Code': 'https://github.com/1flei/PythoC',
    },
    packages=find_packages(exclude=['test', 'test.*', 'docs', 'backup', 'build']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Compilers',
        'Topic :: Software Development :: Code Generators',
    ],
    python_requires='>=3.8',
    install_requires=[
        'llvmlite>=0.40.0,<0.45.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
        ],
    },
    keywords='compiler llvm python static-typing code-generation',
    license='MIT',
    include_package_data=True,
    zip_safe=False,
)
