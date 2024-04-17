from setuptools import find_packages, setup

setup(
    name="python-rag",
    version="1.0.0",
    author="lodgify.com",
    author_email="sohravb@lodgify.com",
    packages=find_packages(),
    test_suite="test",
    install_requires=[
        "wheel",
        "pandas>=1.0.0,<3.0.0",  
        "transformers>=4.0.0,<5.0.0",  
        "ragas>=0.0.22,<1.0.0",  
        "pypdf>=4.0.0,<5.0.0",  
        "python-dotenv>=1.0.0,<2.0.0",
        "langchain==0.1.16",
        "chromadb==0.3.29",
        # "pybind11==2.12.0",
    ],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
        "pytest-timeout",
    ],
    extras_require={
        'test': [
            "pytest",
            "pytest-timeout",
        ],
    },
)
