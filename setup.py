import setuptools
import sys


test_requires = []
test_suite = "tests"
if sys.version_info[:2] == (2, 6):
    test_requires.append("unittest2")
    test_suite = "unittest2.collector"

setuptools.setup(
    name="bson-numpy-codec",
    version="0.1",
    packages=["bson_numpy"],
    install_requires=["numpy"],
    #ext_modules=[module],
    author="A. Jesse Jiryu Davis",
    author_email="jesse@emptysquare.net",
    description="High-performance conversion between NumPy arrays and BSON",
    #long_description=readme_content,
    keywords="bson mongodb numpy array",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Environment :: Console",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Database"
    ],
    #url=
    test_suite=test_suite,
)
