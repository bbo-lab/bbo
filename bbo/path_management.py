import os
from pathlib import Path
import yaml
import sys
import re

def decode_path(path, exist_required=True, replace_dict=None, debug=False):
    is_path = True if isinstance(path, Path) else False
    path = Path(path).as_posix()

    if not "{" in path or not "}" in path:
        return Path(path) if is_path else path

    if replace_dict is None:
        replace_dict = get_replace_dict(with_brackets=False, no_trailing_slash=True, return_list=True)
    replace_dict = {f"{{{k}}}": v for k, v in replace_dict.items()}

    for key, value in replace_dict.items():
        if not key in path:
            continue
        if debug:
            print(f"====== {key} ======")
        if not isinstance(value, list):
            value = [value]
        for v in value:
            decoded_path = Path(path.replace(key, v))
            if decoded_path.expanduser().resolve().exists():  # resolve() necessary because .../[nonexisting folder]/../... is False
                if debug:
                    print(f"Success: {value} - {decoded_path.expanduser().resolve().as_posix()} found.")
                return decoded_path if is_path else decoded_path.as_posix()
            elif debug:
                print(f"{value} - {decoded_path.expanduser().resolve().as_posix()} does not exist.")
        if not exist_required:
            path = path.replace(key, value[0])

    return Path(path) if is_path else path


def encode_path(path, replace_dict=None):
    if replace_dict is None:
        replace_dict = get_replace_dict(with_brackets=False, no_trailing_slash=True, return_list=True)
    replace_dict = {f"{{{k}}}": v for k, v in replace_dict.items()}

    is_path = True if isinstance(path, Path) else False
    path = Path(path).expanduser().resolve().as_posix()

    for key, value in replace_dict.items():
        if not isinstance(value, list):
            value = [value]
        for v in value:
            path = Path(path.replace(v, key)).as_posix()

    return Path(path) if is_path else path


def get_custom_replace_dict():
    config_file = Path("~/.bbo/replace_dict.yml").expanduser().resolve()

    # Create config file if it doesn't exist
    if not config_file.is_file():
        os.makedirs(Path("~/.bbo").expanduser().resolve(), exist_ok=True)
        with open(config_file, "w"):
            pass

    # Open config file
    with open(config_file, "r") as yaml_file:
        custom_replace_dict = yaml.safe_load(yaml_file)
    # An empty file returns None, fix that
    if custom_replace_dict is None:
        custom_replace_dict = {}
    # Python does not know ~, resolve that
    for key in custom_replace_dict:
        if isinstance(custom_replace_dict[key], list):
            custom_replace_dict[key] = [Path(v).expanduser().resolve().as_posix() for v in custom_replace_dict[key]]
        else:
            custom_replace_dict[key] = Path(custom_replace_dict[key]).expanduser().resolve().as_posix()

    return custom_replace_dict


def get_default_replace_dict():
    default_replace_dict = {
        'dropbox': Path("~/Dropbox/").expanduser().resolve().as_posix(),
        'projects_repos': Path("~/code/projects/").expanduser().resolve().as_posix(),
        'python_packages_repos': Path("~/code/python_packages/").expanduser().resolve().as_posix(),
    }

    if sys.platform.startswith('linux'):
        for path in [
            "/media/smb/soma-fs.ad01.caesar.de/bbo/",
            "/media/smb/soma.ad01.caesar.de/bbo/"
        ]:
            try:
                if Path(path).is_dir():
                    default_replace_dict["storage"] = path
                    break
            except OSError as e:
                if e.errno == 116:  # Stale file handle, sometimes happens with SOMA
                    default_replace_dict["storage"] = path
    elif sys.platform.startswith('win') or sys.platform.startswith('cygwin'):
        if not Path("s:/").exists() and Path("o:/").exists():
            default_replace_dict["storage"] = "o:/"
        else:
            default_replace_dict["storage"] = "s:/"
    else:
        print("Warning unknown platform", sys.platform, file=sys.stderr)

    return default_replace_dict


def get_replace_dict(with_brackets=False, no_trailing_slash=False, inverse=False, return_list=False):
    # Note: replace_dict should usually not be used outside this module, use encode_path and decode_path for
    #  path manipulation
    replace_dict = get_default_replace_dict() | get_custom_replace_dict()

    if not return_list:
        for k in replace_dict:
            if isinstance(replace_dict[k], list):
                replace_dict[k] = replace_dict[k][0]

    if with_brackets:
        replace_dict = {f"{{{k}}}": v for k, v in replace_dict.items()}

    if no_trailing_slash:
        replace_dict = remove_trailing_slashes(replace_dict)

    if inverse:
        replace_dict = get_inverse_dict(replace_dict)

    return replace_dict


def replace_by_dict(text, replace_dict=None, inverse=False):
    # Deprecated: use encode_path and decode_path to support multiple replacements
    is_path = False
    if isinstance(text, Path):
        text = text.as_posix()
        is_path = True


    if replace_dict is None:
        replace_dict = get_replace_dict(with_brackets=True, no_trailing_slash=True)

    if inverse:
        replace_dict = get_inverse_dict(replace_dict)

    pattern = re.compile("|".join(re.escape(key) for key in replace_dict.keys()))

    def replace_match(match):
        return replace_dict[match.group(0)]

    if is_path:
        text = Path(text)

    return pattern.sub(replace_match, text)


def remove_trailing_slashes(replace_dict):
    new_replace_dict = {}
    for key, value in replace_dict.items():
        if value[-1] == "/":
            value = value[:-1]
        new_replace_dict[key] = value

    return new_replace_dict


def get_inverse_dict(replace_dict):
    return {v: k for k, v in replace_dict.items()}
