import hashlib
import os

from bbo import path_management


header_version = 0.1


def check_header(file, recursive=None):
    return check_dep_header(file, recursive)


def check_dep_header(file, recursive=None):
    # Recursive may be bool or an integer to
    if recursive is None:
        recursive = False

    with open(file, 'rb') as f:
        if f.read(1) != b'#':
            raise NoDependencyHeaderError(f"{file} does not have a dependency header", file)

    dep_headers = {}
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                key, value = line[2:].split(':', 1)
                if key.startswith('dep-'):
                    (key_type, key) = key[4:].split('_', 1)
                else:
                    continue
                if key not in dep_headers:
                    dep_headers[key] = {}
                dep_headers[key][key_type] = value.strip()

    for key in dep_headers:
        dep_file = path_management.decode_path(dep_headers[key]["file"])

        if 'hash' in dep_headers[key]:
            dep_hash = dep_headers[key]['hash']
            if not calculate_sha256(dep_file) == dep_hash:
                raise DependencyError(f"Dependency {key}: {dep_file} does not have the correct sha256 hash {dep_hash}",
                                      file)

        if 'mtime' in dep_headers[key]:
            dep_date = dep_headers[key]['mtime']
            ti_m = str(int(os.path.getmtime(dep_file) * 1000))   # Milliseconds should be fine, float may differ
            if not ti_m == dep_date:
                raise DependencyError(f"Dependency {key}: {dep_file} changed {ti_m}, expected {dep_date}.",
                                      file)

        try:
            if recursive is True:
                check_dep_header(dep_file, recursive=True)
            elif isinstance(recursive, int) and recursive > 0:
                check_dep_header(dep_file, recursive=recursive - 1)
        except NoDependencyHeaderError:
            pass


    return True


def make_header(dep_dict=None, do_path_management=True):
    header_dict = make_header_dict(dep_dict=dep_dict, do_path_management=do_path_management)
    return "".join([f"# {k}: {v}\n" for k, v in header_dict.items()])


def make_header_dict(dep_dict=None, do_path_management=True):
    if dep_dict is None:
        dep_dict = {}

    header_dict = make_dep_header_dict(dep_dict, do_path_management=do_path_management)
    header_dict["header_version"] = header_version

    return header_dict


def make_dep_header_dict(dep_dict, do_path_management=True, mtime=False, hash=True):
    header = {}
    for key in dep_dict:
        if do_path_management:
            dep_file = path_management.encode_path(dep_dict[key])
        else:
            dep_file = dep_dict[key]

        if hash:
            header["dep-file_" + key] = dep_file
            header["dep-hash_" + key] = calculate_sha256(dep_dict[key])

        if mtime:
            header["dep-file_" + key] = dep_file
            header["dep-mtime_" + key] = str(int(os.path.getmtime(dep_file) * 1000))

    return header


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class DependencyError(Exception):
    def __init__(self, msg, file):
        self.msg = msg
        self.file = file

class NoDependencyHeaderError(DependencyError):
    pass
