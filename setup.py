from setuptools import setup, find_packages

setup(
    name="medcaption-lstm",
    version="1.0.0",
    author="Ankit Thepe",
    description="Medical Image Captioning using VGG16 + LSTM on ROCO dataset",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.10",
        "flask>=2.2",
        "numpy>=1.23",
        "Pillow>=9.0",
        "scikit-learn>=1.1",
        "nltk>=3.7",
        "tqdm>=4.64",
    ],
)
