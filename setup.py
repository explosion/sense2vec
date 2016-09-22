#!/usr/bin/env python
from __future__ import print_function
import os
import shutil
import subprocess
import sys
import contextlib
from distutils.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc

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

class build_ext_options:
    def build_options(self):
        for e in self.extensions:
            e.extra_compile_args = compile_options.get(
                self.compiler.compiler_type, compile_options['other'])
        for e in self.extensions:
            e.extra_link_args = link_options.get(
                self.compiler.compiler_type, link_options['other'])


class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        build_ext.build_extensions(self)


def generate_cython(root, source):
    print('Cythonizing sources')
    p = subprocess.call([sys.executable,
                         os.path.join(root, 'bin', 'cythonize.py'),
                         source])
    if p != 0:
        print(sys.executable)
        print(os.path.join(root, 'bin', 'cythonize.py'))
        print(source)
        raise RuntimeError('Running cythonize failed')


def import_include(module_name):
    try:
        return __import__(module_name, globals(), locals(), [], 0)
    except ImportError:
        raise ImportError('Unable to import %s. Create a virtual environment '
                          'and install all dependencies from requirements.txt, '
                          'e.g., run "pip install -r requirements.txt".' % module_name)


def copy_include(src, dst, path):
    assert os.path.isdir(src)
    assert os.path.isdir(dst)
    shutil.copytree(
        os.path.join(src, path),
        os.path.join(dst, path))


def prepare_includes(path):
    include_dir = os.path.join(path, 'include')
    if os.path.exists(include_dir):
        shutil.rmtree(include_dir)
    os.mkdir(include_dir)

    numpy = import_include('numpy')
    copy_include(numpy.get_include(), include_dir, 'numpy')

    murmurhash = import_include('murmurhash')
    copy_include(murmurhash.get_include(), include_dir, 'murmurhash')


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

    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        return clean(root)

    with chdir(root):
        about = {}
        with open(os.path.join(root, "sense2vec", "about.py")) as f:
            exec(f.read(), about)

        include_dirs = [
            get_python_inc(plat_specific=True),
            os.path.join(root, 'include')]

        ext_modules = []
        for mod_name in MOD_NAMES:
            mod_path = mod_name.replace('.', '/') + '.cpp'
            ext_modules.append(
                Extension(mod_name, [mod_path],
                    language='c++', include_dirs=include_dirs))

        if not is_source_release(root):
            generate_cython(root, 'sense2vec')
            prepare_includes(root)

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
                'numpy',
                'ujson>=1.34',
                'spacy>=0.100,<0.101',
                'preshed>=0.46,<0.47',
                'murmurhash>=0.26,<0.27',
                'cymem>=1.30,<1.31',
                'sputnik>=0.9.0,<0.10.0'],
            cmdclass = {
                'build_ext': build_ext_subclass},
        )


if __name__ == '__main__':
    setup_package()
