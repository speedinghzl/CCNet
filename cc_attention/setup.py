#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 qiang.zhou <theodoruszq@gmail.com>


from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name="rcca2",
      ext_modules=[
          CUDAExtension(
              'rcca', 
              ['src/lib_cffi.cpp', 'src/ca.cu'],
              extra_compile_args = ["-std=c++11"]
              ), 
          ],
      cmdclass={'build_ext': BuildExtension})


