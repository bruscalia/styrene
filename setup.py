from setuptools import setup

setup(
   name='styrene',
   version='0.1.0.rc1',
   description='Styrene reactor modeling package.',
   author='Bruno Scalia C. F. Leite',
   author_email='bruscalia12@gmail.com',
   packages=['styrene'],
   install_requires=[
      'numpy==1.20.*',
      'scipy>=1.7.*',
      'pandas>=1.1.*',
      'matplotlib',
      ],
)