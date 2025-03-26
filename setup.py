from setuptools import setup, find_packages


setup(
    name="lsoc",
    version="0.1.0",

    # Package discovery
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    # Package dependencies defined inline
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "plotly",
        "autograd",
        "scikit-learn",
    ],

    # Python version compatibility
    python_requires=">=3.8",

    # Package metadata
    author="Alistair Reid",
    author_email="alistair.reid@gradientinstitute.org",
    description="Analysis tools for latent capabilities.",

)
