from setuptools import setup

setup(
   name='styrene',
   version='0.1.0',
   description='Styrene reactor modeling package.',
   author='Bruno Scalia C. F. Leite',
   author_email='bruscalia12@gmail.com',
   packages=['styrene'],
   install_requires=['numpy==1.19.*',
                     'scipy>=1.7.*',
                     'pandas>=1.3.*',
		     'collocation @ git+https://github.com/bruscalia/collocation#egg=collocation',
		     'matplotlib',
           ],
)