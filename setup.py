from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
   long_description = fh.read()


setup(
   name='styrene',
   version='0.1.1',
   description='Styrene reactor modeling package.',
   long_description=long_description,
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
