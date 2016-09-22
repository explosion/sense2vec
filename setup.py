#!/usr/bin/env python
from __future__ import print_function
import os
import shutil
import subprocess
import sys
import contextlib
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc
from distutils import ccompiler, msvccompiler

try:
    from setuptools import Extension, setup
except ImportError:
    from distutils.core import Extension, setup


PACKAGES = [
    'sense2vec',
    'sense2vec.tests'
]

MOD_NAMES = [
    'sense2vec.vectors'
]


# By subclassing build_extensions we have the actual compiler that will be used which is really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
compile_options =  {'msvc'  : ['/Ox', '/EHsc'],
                    'other' : ['-O3', '-Wno-unused-function',
                               '-fno-stack-protector']}
link_options    =  {'msvc'  : [],
                    'other' : ['-fno-stack-protector']}


if os.environ.get('USE_BLAS') == '1':
    compile_options['other'].extend([
        '-DUSE_BLAS=1',
        '-fopenmp'])
#else:
#    link_options['other'].extend([
#        '-fopenmp']) 
#

class build_ext_subclass(build_ext):
    def build_extensions(self):
        for e in self.extensions:
            e.extra_compile_args = compile_options.get(
                self.compiler.compiler_type, compile_options['other'])
        for e in self.extensions:
            e.extra_link_args = link_options.get(
                self.compiler.compiler_type, link_options['other'])
        build_ext.build_extensions(self)


def generate_cython(root, source):
    print('Cythonizing sources')
    p = subprocess.call([sys.executable,
                         os.path.join(root, 'bin', 'cythonize.py'),
                         source])
    if p != 0:
        raise RuntimeError('Running cythonize failed')


def is_source_release(path):
    return os.path.exists(os.path.join(path, 'PKG-INFO'))


def clean(path):
    for name in MOD_NAMES:
        name = name.replace('.', '/')
        for ext in ['.so', '.html', '.cpp', '.c']:
            file_path = os.path.join(path, name + ext)
            if os.path.exists(file_path):
                os.unlink(file_path)


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))
    src_path = 'sense2vec'

    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        return clean(root)

    with chdir(root):
        with open(os.path.join(root, src_path, 'about.py')) as f:
            about = {}
            exec(f.read(), about)

        with open(os.path.join(root, 'README.rst')) as f:
            readme = f.read()

        include_dirs = [
            get_python_inc(plat_specific=True),
            os.path.join(root, 'include')]

        if (ccompiler.new_compiler().compiler_type == 'msvc'
            and msvccompiler.get_build_version() == 9):
            include_dirs.append(os.path.join(root, 'include', 'msvc9'))

        ext_modules = []
        for mod_name in MOD_NAMES:
            mod_path = mod_name.replace('.', '/') + '.cpp'
            ext_modules.append(
                Extension(mod_name, [mod_path],
                    language='c++', include_dirs=include_dirs))

        if not is_source_release(root):
            generate_cython(root, src_path)

        setup(
            name=about['title'],
            zip_safe=False,
            packages=PACKAGES,
            package_data={'': ['*.pyx', '*.pxd', '*.h']},
            description=about['summary'],
            author=about['author'],
            author_email=about['email'],
            version=about['version'],
            url=about['uri'],
            license=about['license'],
            ext_modules=ext_modules,
            install_requires=[
                'numpy>=1.7',
                'spacy>=0.100,<0.102',
                'preshed>=0.46,<0.47',
                'murmurhash>=0.26,<0.27',
                'cymem>=1.30,<1.32',
                'sputnik>=0.9.0,<0.10.0'],
            classifiers=[
                'Development Status :: 4 - Beta',
                'Environment :: Console',
                'Intended Audience :: Developers',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: MIT License',
                'Operating System :: POSIX :: Linux',
                'Operating System :: MacOS :: MacOS X',
                'Operating System :: Microsoft :: Windows',
                'Programming Language :: Cython',
                'Programming Language :: Python :: 2.6',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3.3',
                'Programming Language :: Python :: 3.4',
                'Programming Language :: Python :: 3.5',
                'Topic :: Scientific/Engineering'],
            cmdclass = {
                'build_ext': build_ext_subclass},
        )


if __name__ == '__main__':
    setup_package()
