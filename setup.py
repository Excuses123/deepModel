import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepModel",
    version="0.0.4",
    author="lujie",
    author_email="597906300@qq.com",
    description="A PyPi test project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Excuses123/deepModel",
    download_url='https://github.com/Excuses123/deepModel/tags',
    packages=setuptools.find_packages(exclude=["test", "test.models"]),
    python_requires=">=3.0",
    install_requires=['h5py', 'requests'],
    extras_require={
        "cpu": ["tensorflow>=1.14.0"],
        "gpu": ["tensorflow-gpu>=1.14.0"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    license="Apache-2.0",
    keywords=['deep learning', 'tensorflow', 'tensor']
)