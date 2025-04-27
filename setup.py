from setuptools import setup, find_packages


setup(name='PPLR',
      version='1.0.0',
      description='Part-based Pseudo Label Refinement for Unsupervised Person Re-identification',
      author='Yoonki Cho',
      author_email='yoonki@kaist.ac.kr',
      url='https://github.com/yoonkicho/PPLR',
      install_requires=[
          'numpy', 'torch>=2.4.1', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu==1.6.3', 'scikit-image>=0.19.3','torchaudio>=0.12.0','pydantic<2','transformers==4.30.0','tensorboard==2.9.1','wandb','aim==3.27.0'],
      packages=find_packages()
      )