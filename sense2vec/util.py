import sputnik
from sputnik.dir_package import DirPackage
from sputnik.package_list import (PackageNotFoundException,
                                  CompatiblePackageNotFoundException)

from . import about


def get_package(data_dir):
    if not isinstance(data_dir, six.string_types):
        raise RuntimeError('data_dir must be a string')
    return DirPackage(data_dir)


def get_package_by_name(name=None, via=None):
    try:
        return sputnik.package(about.title, about.version,
                               name or about.default_model, data_path=via)
    except PackageNotFoundException as e:
        raise RuntimeError("Model not installed. Please run 'python -m "
                           "sense2vec.download' to install latest compatible "
                           "model.")
    except CompatiblePackageNotFoundException as e:
        raise RuntimeError("Installed model is not compatible with sense2vec "
                           "version. Please run 'python -m sense2vec.download "
                           "--force' to install latest compatible model.")
