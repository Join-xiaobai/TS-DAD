from setuptools import setup, find_packages

setup(
    name="tsdad",
    version="1.0.0",
    description="Attribute-Preserving Topology-Decoupled Learning for Attributed Graph Anomaly Detection",
    author="TS-DAD Contributors",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torch-geometric>=2.0.0",
        "pygod>=0.3.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
