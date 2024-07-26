from setuptools import setup, find_packages


def readme():
  with open('README.md', 'r') as f:
    return f.read()

setup(
  name='celldx-py',
  version='1.0.2',
  author='HistAI',
  author_email='apchelnikov@hist.ai',
  description='celldx-py is the Python Library to interact with the Hibou model inference API',
  long_description=readme(),
  long_description_content_type='text/markdown',
  url='https://github.com/HistAI/celldx-py',
  packages=find_packages(),
  install_requires=['requests', 'opencv-python', 'numpy'],
  classifiers=[
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent'
  ],
  project_urls={
    'Documentation': 'https://celldx.hist.ai/docs/'
  },
  python_requires='>=3.8'
)
