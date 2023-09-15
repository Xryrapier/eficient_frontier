from setuptools import setup, find_packages



with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]
print(len(requirements))



setup(name='eficient_frontier',
      description="project package for eficient frontier",
      packages=find_packages(),
      install_requires=requirements)
