from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cinic10-mlp-cnn-comparison",
    version="1.0.0",
    author="Ashesi University Student",
    description="Comparing MLP and CNN architectures for image classification on CINIC-10",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/cinic10-mlp-cnn-comparison",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "jupyter",
            "pytest",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "cinic10-api=src.api.fastapi_app:main",
        ],
    },
)