from setuptools import setup, find_packages

setup(
    name='fmri_tools',
    version='0.1.0',
    description='fMRI tools',
    author='Alejandro Veloz',
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "balloon-netsim = fmri_tools.balloon_netsim_cli:main",
            # "fmri-preprocess = fmri_tools.preprocess_cli:main",
            # "fmri-analyze = fmri_tools.analyze_cli:main",
        ],
    },
    install_requires=[
        # your dependencies
    ],
    python_requires='>=3.7',
)
