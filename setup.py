from setuptools import find_packages, setup

setup(
    name="harl",
    version="1.0.0",
    author="PKU-MARL",
    description="PyTorch implementation of HARL Algorithms",
    url="https://github.com/PKU-MARL/HARL",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "pyyaml>=5.3.1",
        "tensorboard>=2.2.1",
        "tensorboardX",
        "setproctitle",
    ],
    extras_require={
        "lbf-vmas": [
            "gym==0.26.2",
            "gymnasium==1.1.1",
            "lbforaging==2.0.0",
            "vmas[gymnasium]==1.4.3",
        ],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
