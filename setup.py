from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="HAIPW",
    version="0.1",
    author="Javier Abad & Piersilvio de Bartolomeis",
    author_email="javier.abadmartinez@ai.ethz.ch",
    description="Python implementation of the methods introduced in the paper: Efficient Randomized Experiments Using Foundation Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaabmar/HAIPW",
    package_dir={"": "HAIPW"},
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="machine learning, language models, causal inference, randomized trials",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "openai",
        "anthropic",
        "scikit-learn",
        "scipy",
        "tqdm",
        "datasets",
        "transformers",
        "bitsandbytes",
        "accelerate"
    ],
    python_requires=">=3.12",
)
