import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='attributionpriors',  
     version='0.1',
     scripts=['ops'] ,
     author="Gabriel Erion, Joseph D. Janizek, and Pascal Sturmfels",
     author_email="",
     description="Tools for training explainable models using attribution priors.",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/suinleelab/attributionpriors",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
	"Intended Audience :: Science/Research",
	"Topic :: Scientific/Engineering :: Artifical Intelligence",
     ],
 )