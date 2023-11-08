import yaml
from pathlib import Path
import re
import os
from pprint import pprint
import sys
from bbo.path_management import get_replace_dict

uuid = "f1f5ba7b_baf1_4f83_b940_a5f9e8468c45"


def load(yml_path, replace_dict=None, replace_dict_set_file=False, include=True, dependencies=None, ignore_missing=False):
    if replace_dict is None:
        replace_dict = get_replace_dict()

    if isinstance(yml_path, str):
        yml_path = Path(yml_path)

    yml_path.expanduser().resolve()

    if not yml_path.is_file():
        raise FileNotFoundError(yml_path.as_posix())

    try:
        with open(yml_path, 'r') as file:
            yaml_string = file.read()
    except:
        print(yml_path)
        raise

    # !include cannot be read by the yaml reader, we replace with a unique id
    yaml_string = re.sub(r'(\s*)!include:', r"\1" + uuid + ":", yaml_string)
    yaml_dict = yaml.safe_load(yaml_string)
    if yaml_dict is None:
        yaml_dict = {}

    # TODO: It might be worth checking https://stackoverflow.com/questions/528281/how-can-i-include-a-yaml-file-inside-another
    replace_include(yaml_dict, yml_path.parent, replace_dict, include, dependencies, ignore_missing)

    if "file" not in replace_dict or replace_dict_set_file:
        replace_dict["file"] = yml_path.parent.as_posix()
    yaml_dict = replace_placeholders(yaml_dict, replace_dict)

    return yaml_dict


def replace_include(yaml_dict: dict, path: Path, replace_dict=None,
                    include=True, dependencies=None, ignore_missing=False):
    for key in list(yaml_dict.keys()):  # Keys will change during iteration
        if key == uuid:
            subdict_path = Path(yaml_dict[key]).expanduser()

            if not subdict_path.is_absolute():
                subdict_path = (path / subdict_path).expanduser().resolve()
            else:
                subdict_path = subdict_path.resolve()
            if dependencies is not None:
                dependencies.add(subdict_path)
            if include and (not ignore_missing or subdict_path.is_file()):
                yaml_dict.pop(uuid)
                subdict = load(subdict_path.as_posix(), replace_dict, replace_dict_set_file=True,
                               include=include, dependencies=dependencies, ignore_missing=ignore_missing)

                for subkey in subdict:
                    if subkey in yaml_dict:
                        merge_dicts(yaml_dict[subkey], subdict[subkey])
                    else:
                        yaml_dict[subkey] = subdict[subkey]
        elif isinstance(yaml_dict[key], dict):
            replace_include(yaml_dict[key], path, replace_dict, include, dependencies, ignore_missing)

    return yaml_dict


def merge_dicts(target_dict, merge_dict):
    for key in merge_dict:
        if key in target_dict:
            if isinstance(target_dict[key], dict) and isinstance(merge_dict[key], dict):
                merge_dicts(target_dict[key], merge_dict[key])
            else:
                target_dict[key] = merge_dict[key]
        else:
            target_dict[key] = merge_dict[key]


def replace_placehodlers_recursive(entry, replace_list):
    if isinstance(entry, str):
        for rep in replace_list:
            entry = entry.replace(rep[0], rep[1])
    if isinstance(entry, dict):
        for key in entry:
            entry[key] = replace_placehodlers_recursive(entry[key], replace_list)
    if isinstance(entry, list):
        for i in range(len(entry)):
            entry[i] = replace_placehodlers_recursive(entry[i], replace_list)
    return entry


def replace_placeholders(yaml_dict: dict, replace_dict=None):
    if replace_dict == None or len(replace_dict) == 0:
        return yaml_dict
    replace_list = [("{"+item[0]+"}",item[1]) for item in replace_dict.items()]
    return replace_placehodlers_recursive(yaml_dict, replace_list)
