import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='attributionpriors',  
     version='0.1.2',
     author="The Lee Lab at UW",
     author_email="psturm@cs.washington.edu",
     description="Tools for training explainable models using attribution priors.",
     long_description="""
         This package contains tools to train deep learning models using attribution priors.
         Through attribution priors, practitioners can constraint models to 
         behave more intuitively on a wide variety of tasks. For usage guidelines,
         see [the GitHub repo](https://github.com/suinleelab/attributionpriors).
         For more details about how to use attribution priors, see [the arXiv paper](https://arxiv.org/abs/1906.10670).
     """,
     long_description_content_type="text/markdown",
     url="https://github.com/suinleelab/attributionpriors",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
         "Intended Audience :: Science/Research",
         "Topic :: Scientific/Engineering"
     ],
    install_requires=['numpy', 'tensorflow'],  
 )
