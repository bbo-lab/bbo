import collections
import copy


def deepmerge(source, destination, inplace=False):
    if not inplace:
        destination = copy.deepcopy(destination)
    for k, v in source.items():
        if isinstance(v, collections.abc.Mapping):
            destination[k] = deepmerge(v, destination.get(k, {}), inplace=True)
        else:
            destination[k] = v
    return destination
