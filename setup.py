#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob

from setuptools import find_packages, setup

NAME = 'chatbot_llm'

setup(
    name=NAME,
    version='0.1.1',
    license='Apache-2.0',
    description='Lifecycle chatbot backend aligned to the ROS4HRI dialogue contract',
    author='juanbeck',
    author_email='juanbeck@icloud.com',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + NAME, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['res/' + NAME]),
        ('share/' + NAME + '/launch', glob('launch/*.launch.py')),
        ('share/ament_index/resource_index/pal_system_module',
         ['module/' + NAME]),
        ('share/' + NAME + '/module', ['module/' + NAME + '_module.yaml']),
        ('share/ament_index/resource_index/pal_configuration.' + NAME,
            ['config/' + NAME]),
        ('share/' + NAME + '/config', glob('config/*.y*ml')),
    ],
    tests_require=['pytest'],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'start_node = ' + NAME + '.start_node:main'
        ],
    },
)
