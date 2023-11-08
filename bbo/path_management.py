import os
from pathlib import Path
import yaml
import sys


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
        custom_replace_dict[key] = Path(custom_replace_dict[key]).expanduser().resolve().as_posix()

    return custom_replace_dict


def get_default_replace_dict():
    default_replace_dict = {
        'dropbox': Path("~/Dropbox/").expanduser().resolve().as_posix()
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
        default_replace_dict["storage"] = "s:/"
    else:
        print("Warning unknown platform", sys.platform, file=sys.stderr)

    return default_replace_dict


def get_replace_dict():
    return get_default_replace_dict() | get_custom_replace_dict()
