from setuptools import setup, find_packages

setup(
    name='unblink',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(
        where='src',
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    test_suite='tests',
    entry_points={
        'console_scripts': [
            'unblink=unblink.main:main'
        ]
    },
    description='A tool for replacing closed eyes in photos',
    author='Noam Isachar, Ilya Andreev, Raj Krishnan',
    url='https://github.com/noamisachar/unblink',
    license='GNU General Public License v3.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    keywords='unblink blink eyes image processing'
)
