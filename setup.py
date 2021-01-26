import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepModel",
    version="0.0.2",
    author="lujie",
    author_email="597906300@qq.com",
    description="A PyPi test project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Excuses123/deepModel",
    download_url='https://github.com/Excuses123/deepModel/tags',
    packages=setuptools.find_packages(exclude=["tests", "tests.models", "tests.layers"]),
    install_requires=['h5py','requests'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="Apache-2.0",
    keywords=['deep learning', 'tensorflow', 'tensor']
)