import pathlib
from setuptools import setup, find_namespace_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='mirabolic',
    packages=find_namespace_packages(),
    version='0.0.1',
    license='MIT',
    description='Statistical and Machine Learning tools from Mirabolic',
    long_description=README,
    long_description_content_type="text/markdown",
    author='Bill Bradley',
    url='https://github.com/Mirabolic/mirabolic',
    download_url='https://github.com/Mirabolic/mirabolic/archive/refs/tags/v0.0.2.tar.gz',  # noqa: E501
    include_package_data=True,
    keywords=['Statistics', 'Machine Learning'],
    install_requires=[
        'tensorflow>=2.4.1',
        'numpy>=1.19.2',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
