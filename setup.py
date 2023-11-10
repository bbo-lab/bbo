import pathlib
from setuptools import find_packages, setup
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="bbo_bbo",
    version="0.2.3",
    description="Base tools for MPINB BBO group",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bbo-lab/bbo",
    author="BBO-lab @ caesar",
    author_email="kay-michael.voit@caesar.de",
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=['bbo'],
    include_package_data=True,
    install_requires=["pyyaml"],
)
