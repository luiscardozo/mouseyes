from setuptools import setup, find_packages

setup(name="mouseyes",
    version="1.0.1",
    description="Move your mouse with your webcam using OpenVINO AI",
    packages=find_packages('src'),
    package_dir={'': 'src'})