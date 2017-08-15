#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'tensorflow',
    'numpy',
    'scipy',
    'requests',
    'tqdm',
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
]

setup(
    name='tf_datasets',
    version='0.1.0',
    description="Python scripts to download public datasets and generate tfrecords for it.",
    long_description=readme,
    author="Thibaut Mattio",
    author_email='thibaut.mattio@gmail.com',
    url='https://github.com/tmattio/tf_datasets',
    packages=find_packages(include=['tf_datasets']),
    entry_points={
        'console_scripts': [
            'tf_make_datasets = tf_datasets.make_dataset:run',
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='tf_datasets',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
