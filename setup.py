from setuptools import setup, find_packages, Extension
from codecs import open
from os import path

try:
  import numpy as np
except ImportError:
  exit('Please install numpy>=1.11.2 first.')

try:
  from Cython.Build import cythonize
  from Cython.Distutils import build_ext
except ImportError:
  USE_CYTHON = False
else:
  USE_CYTHON = True

__version__ = '0.0.1'

here = path.abspath(path.dirname(__file__))

cmdclass = {}

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
  Extension(
    'rec.restricted_svd',
    ['rec/restricted_svd' + ext],
    include_dirs=[np.get_include()]
  ),
  ]

if USE_CYTHON:
  ext_modules = cythonize(extensions, gdb_debug=True)
  cmdclass.update({'build_ext': build_ext})
else:
  ext_modules = extensions

setup(
  name='rec_dmh',
  author='Dany Haddad',
  author_email='danyhaddad43@gmail.com',

  description=('recsys'),

  version=__version__,
  url='danyhaddad.me',

  license='GPLv3+',
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 2.7',
  ],
  keywords='recommender recommendation system',

  packages=find_packages(exclude=['tests*']),
  include_package_data=True,
  ext_modules=ext_modules,
  cmdclass=cmdclass)
