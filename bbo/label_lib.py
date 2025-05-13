import os
from pathlib import Path
import numpy as np
import yaml
import itertools
from bbo.exceptions import NoDataException
import re
import logging
from packaging.version import Version

logger = logging.getLogger(__name__)


def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())


yaml.add_representer(np.ndarray, ndarray_representer)

version = "1.0"


def update(labels, labeler="_unknown", do_purge_nans=False):
    # Old ACM-style labels
    if all([isinstance(k, int) for k in labels.keys()]):
        return acm_to_labels(labels, labeler)

    # Before versioning
    if "version" not in labels:
        labels["version"] = "0.1"

    if not isinstance(labels['version'], str):
        labels['version'] = str(labels['version'])

    if Version(labels["version"]) < Version("0.3"):
        if "labeler" not in labels:
            labels["labeler"] = {}
        if "labeler_list" not in labels:
            labels["labeler_list"] = [labeler]
        if labeler not in labels["labeler_list"]:
            labels["labeler_list"].append(labeler)

        labeled_frame_idxs = get_labeled_frame_idxs(labels)
        labeler_idx = labels["labeler_list"].index(labeler)

        for f_idx in labeled_frame_idxs:
            if f_idx not in labels["labeler"]:
                labels["labeler"][f_idx] = labeler_idx
            if "fr_times" in labels and f_idx not in labels["fr_times"]:
                labels["fr_times"][f_idx] = 0
        labels["version"] = "0.3"

    if Version(labels["version"]) < Version("0.4"):
        labeler = {}
        point_times = {}
        for ln in labels["labels"]:
            labeler[ln] = {}
            point_times[ln] = {}
            for fr_idx in labels['labels'][ln]:
                data_shape = get_data_shape(labels)
                labeler[ln][fr_idx] = np.ones(data_shape[0], dtype=np.uint16) * labels['labeler'][fr_idx]
                nanmask = np.any(np.isnan(labels["labels"][ln][fr_idx]), axis=1)
                labeler[ln][fr_idx][nanmask] = 0
                factor = labels['fr_times'][fr_idx] if 'fr_times' in labels else 1
                point_times[ln][fr_idx] = np.ones(data_shape[0], dtype=np.uint64) * factor

        labels["point_times"] = point_times
        labels["labeler"] = labeler
        if 'fr_times' in labels:
            labels.pop("fr_times")

    if Version(labels["version"]) < Version("1.0"):
        labels["action_list"] = ["create", "delete"]
        labels = convert_v0_to_v1(labels)

    # Bring labeler list in shape (add specials etc.)
    make_global_lists([labels])

    if do_purge_nans:
        purge_nans(labels)

    labels["version"] = version
    return labels


def purge_nans(labels):
    if legacy := (Version(labels["version"]) < Version("1.0")):
        labels = convert_v0_to_v1(labels)

    del_list = []
    for ln in labels["labels"]:
        for fr_idx in labels['labels'][ln]:
            if np.all(np.isnan(labels['labels'][ln][fr_idx]["coords"])):
                del_list.append((ln, fr_idx))

    for ln, fr_idx in del_list:
        del labels["labels"][ln][fr_idx]

    if legacy:
        labels = convert_v1_to_v0(labels)

    return labels


def get_empty_labels():
    return {
        'version': version,
        'labels': {},
        'action_list': ["create", "delete"],
        'labeler_list': ["_unmarked", "_unknown"],
    }


def acm_to_labels(acmlabels, labeler="_unknown"):
    labels = get_empty_labels()
    if labeler not in labels['labeler_list']:
        labels['labeler_list'].append(labeler)
    uk_idx = labels['labeler_list'].index(labeler)

    def initialize_label_name(labels, label_name):
        if label_name not in labels["labels"]:
            labels["labels"][label_name] = {}
        if label_name not in labels["labeler"]:
            labels["labeler"][label_name] = {}
        if label_name not in labels["point_times"]:
            labels["point_times"][label_name] = {}

    for fr_idx in acmlabels:
        for ln in acmlabels[fr_idx]:
            initialize_label_name(labels, ln)
            labels['labels'][ln][fr_idx] = acmlabels[fr_idx][ln]
            labels['point_times'][ln][fr_idx] = np.ones((acmlabels[fr_idx][ln].shape[0])) * uk_idx
            labels['labeler'][ln][fr_idx] = np.zeros((acmlabels[fr_idx][ln].shape[0]))

    labels = update(labels)
    return labels


def labels_to_acm(labels):
    acmlabels = {}
    for ln in labels['labels']:
        for fr_idx in labels['labels'][ln]:
            acmlabels[fr_idx][ln] = labels['labels'][ln][fr_idx]["coords"]
    return acmlabels


def load(file_path, load_npz=False, v0_format=None):
    logger.log(logging.DEBUG, f"Loading {file_path}")
    if v0_format is None:
        v0_format = True
        logger.warning("DEPRECATED FORMAT: For new implementations, use v0_format=False. "
                       "Behavior will be changed after publication of bird paper. Use v0_format=True to "
                       "suppress this message.")

    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.with_suffix(".yml").is_file() and not (load_npz and file_path.suffix == ".npz"):
        file_path = file_path.with_suffix(".yml")
        try:
            with open(file_path.as_posix(), 'r') as f:
                labels = read_label_yaml(f)
        except Exception as e:
            # Fall back to very slow pyyaml reader
            logger.log(logging.WARN, "WARNING: Fallback to pyYAML")
            logger.log(logging.WARN, e)
            raise e
            labels = load_raw_yaml(file_path)
    elif file_path.with_suffix(".npz").is_file():
        if not load_npz:
            raise FileNotFoundError(f"There should not be any bbo-labelgui npzs without yml file left. "
                                    "Check if this is an error (e.g. yml file intentionally deleted).")
        file_path = file_path.with_suffix(".npz")
        labels = np.load(file_path, allow_pickle=True)["arr_0"][()]
        logger.log(logging.WARN, f"Loaded deprecated npz file! {file_path.as_posix()}")
    else:
        raise FileNotFoundError(file_path.as_posix())

    try:
        labels = update(labels, labeler=file_path.parent.parent.stem)
    except KeyError as e:
        logger.log(logging.ERROR, "Did not find expected keys in ", file_path.as_posix())
        raise e

    if v0_format:
        labels = convert_v1_to_v0(labels)

    return labels


def convert_v0_to_v1(labels):
    if Version(labels["version"]) >= Version("1.0"):
        return labels

    if "_unknown" not in labels['labeler_list']:
        labels['labeler_list'].insert(0, "_unknown")

    labels_new = {
        'labels': {},
        'labeler_list': labels['labeler_list'],
        'action_list': labels['action_list'],
        'version': "1.0",
    }
    
    idx_unknown = labels['labeler_list'].index("_unknown")
    for ln in labels['labels']:
        labels_new['labels'][ln] = {}
        for fr_idx in labels['labels'][ln]:
            labels_new['labels'][ln][fr_idx] = {
                'coords': labels['labels'][ln][fr_idx],
                'labeler': np.full(labels['labels'][ln][fr_idx].shape[0], idx_unknown, dtype=np.uint16),
                'point_times': np.zeros(labels['labels'][ln][fr_idx].shape[0], dtype=float),
            }
        try:
            for fr_idx in labels['labels'][ln]:
                labels_new['labels'][ln][fr_idx]['labeler'] = labels['labeler'][ln][fr_idx]
                labels_new['labels'][ln][fr_idx]['point_times'] = labels['point_times'][ln][fr_idx]
        except:
            logger.log(logging.WARN, "Additional info was not found in file")

    return update(labels_new)


def convert_v1_to_v0(labels):
    if Version(labels["version"]) < Version("1.0"):
        return labels

    labels_old = {
        'labels': {},
        'labeler': {},
        'point_times': {},
        'labeler_list': labels['labeler_list'],
        'action_list': labels['action_list'],
        'version': "0.4",
    }
    for ln in labels['labels']:
        labels_old['labels'][ln] = {}
        labels_old['labeler'][ln] = {}
        labels_old['point_times'][ln] = {}
        for fr_idx in labels['labels'][ln]:
            labels_old['labels'][ln][fr_idx] = labels['labels'][ln][fr_idx]["coords"]
            labels_old['labeler'][ln][fr_idx] = labels['labels'][ln][fr_idx]["labeler"]
            labels_old['point_times'][ln][fr_idx] = labels['labels'][ln][fr_idx]["point_times"]
    return labels_old


def load_raw_yaml(file_path: Path):
    with open(file_path.as_posix(), 'r') as f:
        labels = yaml.safe_load(f)

    for ln in labels["labels"]:
        for fr_idx in labels["labels"][ln]:
            labels["labels"][ln][fr_idx] = np.array(labels["labels"][ln][fr_idx])

    for ln in labels["labeler"]:
        for fr_idx in labels["labeler"][ln]:
            labels["labeler"][ln][fr_idx] = np.array(labels["labeler"][ln][fr_idx])

    for ln in labels["point_times"]:
        for fr_idx in labels["point_times"][ln]:
            labels["point_times"][ln][fr_idx] = np.array(labels["point_times"][ln][fr_idx])

    return labels


def save(file_path, labels, yml_only=False, v0_format=False):
    # if v0_format is None:
    #     v0_format = True
    #     logger.warning("DEPRECATED FORMAT: For new implementations, use v0_format=False. "
    #                    "Behavior will be changed after publication of bird paper. Use v0_format=True to "
    #                    "suppress this message.")
    labels = update(labels)

    if v0_format:
        labels = convert_v1_to_v0(labels)

    if isinstance(file_path, str):
        file_path = Path(file_path)

    os.makedirs(file_path.parent.as_posix(), exist_ok=True)

    if not yml_only:
        # We still save npz for a while in case there is an issue with yml. This is ABSOLUTELY DEPRECATED
        # and should not be used in any circumstance except for emergencies with the yml files.
        np.savez(file_path.with_suffix(".npz").as_posix(), labels)

    with open(file_path.with_suffix(".yml").as_posix(), 'w') as f:
        write_label_yaml(f, labels)


def get_labels(labels, allow_empty=False):
    if allow_empty:
        return list(labels["labels"].keys())
    else:
        return [lm for lm in labels["labels"] if len(labels["labels"][lm].keys()) > 0]


def get_labeled_frame_idxs(labels, label_set=None):
    frames = set()
    if label_set is None:
        label_set = labels["labels"].keys()
    for ln in label_set:
        if ln in labels["labels"]:
            frames.update(set(labels["labels"][ln].keys()))

    return np.array(sorted(list(frames))).astype(int)


def get_labels_from_frame(labels, frame_idx):
    if legacy := (Version(labels["version"]) < Version("1.0")):
        labels = convert_v0_to_v1(labels)

    frame_labels = {}

    for lb in labels["labels"]:
        if frame_idx in labels["labels"][lb] and not np.all(np.isnan(labels["labels"][lb][frame_idx]["coords"])):
            frame_labels[lb] = labels["labels"][lb][frame_idx]["coords"]

    return frame_labels


def get_frame_labelers(labels, fr_idx, cam_idx=None):
    if legacy := (Version(labels["version"]) < Version("1.0")):
        labels = convert_v0_to_v1(labels)

    labeler_idxs = set()
    for ln in labels["labels"]:
        if fr_idx in labels["labels"][ln]:
            if cam_idx is None:
                labeler_idxs.update(labels["labels"][ln][fr_idx]["labeler"])
            else:
                labeler_idxs.update(labels["labels"][ln][fr_idx]["labeler"][(cam_idx,),])

    labelers = [labels["labeler_list"][i] for i in labeler_idxs]
    if "_unmarked" in labelers:
        labelers.pop(labelers.index("_unmarked"))
    return labelers


def merge(labels_list: list, target_file=None, overwrite=False, yml_only=False, times_to_0=True, nan_deletes=False):
    # Load data from files
    labels_list = [update(ll) if isinstance(ll, dict) else load(ll, v0_format=False) for ll in labels_list]
    # Normalize path of target_file
    if isinstance(target_file, str) or isinstance(target_file, Path):
        target_path = Path(target_file).expanduser().resolve()
        target_labels = load(target_path, v0_format=False)
    elif target_file is None:
        if len(labels_list) == 0:
            return get_empty_labels()

        target_labels = labels_list[0]
        labels_list = labels_list[1:]
    else:
        target_labels = target_file

    make_global_lists([target_labels]+labels_list)

    # Merge file-wise
    data_shape = None
    for labels in [target_labels] + labels_list:
        try:
            data_shape = get_data_shape(labels)
            break
        except NoDataException:
            pass

    if data_shape is None:
        logger.log(logging.WARN, f"Merging aborted, could not determine data structure")
        return target_file

    index_unmarked = target_labels["labeler_list"].index("_unmarked")  # Labeler are already matched
    index_create = target_labels["action_list"].index("create")
    index_delete = target_labels["action_list"].index("delete")
    default_action = np.ones(data_shape[0], dtype=int) * index_create
    for i_labels, labels in enumerate(labels_list):
        logger.log(logging.INFO, f"Merging {i_labels + 1}/{len(labels_list) - 1} label sources")

        for ln in labels["labels"]:
            if ln not in target_labels["labels"]:
                target_labels["labels"][ln] = {}
            for fr_idx in labels["labels"][ln]:
                if fr_idx not in target_labels["labels"][ln]:
                    target_labels["labels"][ln][fr_idx] = {
                        'coords': np.full(data_shape, np.nan),
                        'labeler': np.ones(data_shape[0], dtype=np.uint16) * index_unmarked,
                        'point_times': np.zeros(data_shape[0], dtype=float)
                    }

                target_entry = target_labels["labels"][ln][fr_idx]
                source_entry = labels["labels"][ln][fr_idx]

                # Source is not labeled unmarked
                transfer_mask = source_entry['labeler'] != index_unmarked
                # Action is create
                transfer_mask &= source_entry.get("action", default_action) == index_create
                # Target time is older. We do <= to be able to do in place corrections in the merged files.
                transfer_mask &= target_entry["point_times"] <= source_entry["point_times"]
                # Source is not nan
                if not nan_deletes:
                    transfer_mask &= np.all(~np.isnan(source_entry["coords"]), axis=-1)

                if not overwrite:
                    transfer_mask &= np.all(~np.isnan(target_entry["coords"]))

                target_entry["coords"][transfer_mask] = source_entry["coords"][transfer_mask]
                target_entry["labeler"][transfer_mask] = source_entry["labeler"][transfer_mask]
                target_entry["point_times"][transfer_mask] = source_entry["point_times"][transfer_mask]

    sort_dictionaries(target_labels)

    if times_to_0:
        set_point_times_to_zero(target_labels)

    if target_file is not None:
        save(target_file, target_labels, yml_only=yml_only)
        logger.log(logging.INFO, f"Saved  {target_file.as_posix()}")
    return target_labels


def set_point_times_to_zero(labels, exclude_users=()):
    exlude_user_idxs = np.array([labels["labeler_list"].index(u) for u in exclude_users if u in labels["labeler_list"]])
    for ln in labels["labels"]:
        for fr_idx in labels["labels"][ln]:
            mask = np.isin(labels["labels"][ln][fr_idx]["labeler"], exlude_user_idxs)
            labels["labels"][ln][fr_idx]["point_times"][mask] = 0


def combine_cams(labels_list: list, target_file=None, yml_only=False):
    raise RuntimeError(f"Function combine_cams currently not implemented")


def reorder_cams(labels, new_order: tuple, target_file=None, yml_only=False, reorder_times=True):
    # Normalize path of target_file
    if isinstance(target_file, str):
        target_file = Path(target_file).expanduser().resolve()

    # Load data from files
    if isinstance(labels, str):
        labels = Path(labels)
    if isinstance(labels, Path):
        labels = labels.expanduser().resolve()
        labels = load(labels.as_posix())

    for ln in labels["labels"]:
        for fr_idx in labels["labels"][ln]:
            fields_list = ["coords", "labeler", "action"]
            if reorder_times:
                fields_list = fields_list.append("point_times")

            for field in fields_list:
                if field in labels["labels"][ln][fr_idx]:
                    labels["labels"][ln][fr_idx][field] = labels["labels"][ln][fr_idx][field][new_order,]

    if target_file is not None:
        save(target_file, labels, yml_only=yml_only)
    return labels


def sort_dictionaries(target_labels):
    # Sort dictionaries
    for label in target_labels["labels"]:
        target_labels["labels"][label] = dict(sorted(target_labels["labels"][label].items()))
    target_labels["labels"] = dict(sorted(target_labels["labels"].items()))


def make_global_lists(labels_list):
    # This changes labels_list in place!!!
    # Create a new global list of all labelers

    # Lists in first entry should be preserved to minimize changes in target file for git tracking
    labeler_list_all = labels_list[0]["labeler_list"].copy()
    for labels in labels_list[1:]:
        for labeler in labels["labeler_list"]:
            if labeler not in labeler_list_all:
                labeler_list_all.append(labeler)
    action_list_all = labels_list[0]["action_list"].copy()
    for labels in labels_list[1:]:
        for action in labels["action_list"]:
            if action not in action_list_all:
                action_list_all.append(action)

    # Get specials to the front. This superseeds preserving target_file order.
    if "_unmarked" in labeler_list_all:
        labeler_list_all.pop(labeler_list_all.index("_unmarked"))
    if "_unknown" in labeler_list_all:
        labeler_list_all.pop(labeler_list_all.index("_unknown"))
    labeler_list_all.insert(0, "_unknown")
    labeler_list_all.insert(0, "_unmarked")

    # Get default actions to the front, This superseeds preserving target_file order.
    if "create" in action_list_all:
        action_list_all.pop(action_list_all.index("create"))
    if "delete" in action_list_all:
        action_list_all.pop(action_list_all.index("delete"))
    action_list_all.insert(0, "delete")
    action_list_all.insert(0, "create")

    # Rewrite to global index list
    for labels in labels_list:
        if labels is None:
            continue
        for ln in labels["labels"]:
            for fr_idx in labels['labels'][ln]:
                for i, labeler_idx in enumerate(labels['labels'][ln][fr_idx]["labeler"]):
                    labeler = labels["labeler_list"][labeler_idx]
                    labels['labels'][ln][fr_idx]["labeler"][i] = labeler_list_all.index(labeler)
                if "action" in labels["labels"][ln][fr_idx]:
                    for i, action_idx in enumerate(labels['labels'][ln][fr_idx]["action"]):
                        action = labels["action_list"][action_idx]
                        labels['labels'][ln][fr_idx]["action"][i] = action_list_all.index(action)
        labels["labeler_list"] = labeler_list_all.copy()
        labels["action_list"] = action_list_all.copy()


def get_data_shape(labels):
    for ln in labels['labels']:
        for fr_idx in labels['labels'][ln]:
            if Version(labels['version']) < Version("1.0"):
                return labels['labels'][ln][fr_idx].shape
            else:
                return labels['labels'][ln][fr_idx]["coords"].shape
    raise NoDataException("Did not find any data to determine data shape")


def get_n_cams(labels):
    return get_data_shape(labels)[0]


def to_array(labels,
             extract_frame_idxs=None, extract_labels=None,  # Extract only these parts of data
             time_bases=None,  # Rearrange after these time bases for cams
             label_identity=None  # Treat as identical labels (nanmean if both are labeled)
             ):
    logger.log(logging.WARN, "DEPRECATED: Use to_numpy")
    return to_numpy(labels, extract_frame_idxs, extract_labels, time_bases, label_identity)


def to_numpy(labels,
             extract_frame_idxs=None, extract_labels=None,  # Extract only these parts of data
             time_bases=None,  # Rearrange after these times for cams
             time_bases_complete=False,  # Supplied time bases are the full time bases of the video
             strip_nans=False,  # Remove all times that are fully nan
             label_identity=None,  # Treat as identical labels (nanmean if both are labeled)
             ):
    if legacy := (Version(labels["version"]) < Version("1.0")):
        labels = convert_v0_to_v1(labels)

    cams_n = get_n_cams(labels)

    if extract_frame_idxs is None:
        extract_frame_idxs = get_labeled_frame_idxs(labels)

    if extract_labels is None:
        extract_labels = get_labels(labels)

    scalar_label = isinstance(extract_labels, str)
    if scalar_label:
        extract_labels = (extract_labels,)

    if time_bases is None:
        time_bases = [extract_frame_idxs for _ in range(cams_n)]

    if time_bases_complete:
        time_bases = [tb[extract_frame_idxs] for tb in time_bases]

    assert np.all([len(tb) == len(extract_frame_idxs) for tb in time_bases]), (
        "time_bases and extract_frame_idxs must match in length")

    time_base = np.unique(np.concatenate(time_bases, axis=0))

    # cams x frames x landmarks x pix_coords
    landmark_imcoords = np.full((cams_n, len(time_base), len(extract_labels), 2), np.nan)
    for i_lm, lm in enumerate(extract_labels):
        if lm in labels["labels"]:
            for i_fr, fr_idx in enumerate(extract_frame_idxs):
                # Map identities
                if label_identity is not None and label_identity[i_lm] is not None:
                    lm_idx = extract_labels.index(label_identity[i_lm])
                else:
                    lm_idx = i_lm

                for i_cam in range(cams_n):
                    cam_time = time_bases[i_cam][i_fr]
                    time_base_idx = np.where(time_base == cam_time)[0][0]
                    if fr_idx in labels["labels"][lm]:
                        cam_coords = labels["labels"][lm][fr_idx]["coords"][i_cam]
                        if not np.any(np.isnan(cam_coords)):
                            # if (fr_idx == 7529 and i_cam == 1) or (fr_idx == 7572 and i_cam == 0):
                            #     print(f"Writing cam {i_cam}, frame {fr_idx} to {time_base_idx}")
                            landmark_imcoords[i_cam, time_base_idx, lm_idx] = (
                                np.nanmean(np.array(
                                    [cam_coords, landmark_imcoords[i_cam, time_base_idx, lm_idx]]
                                ), axis=0))

    if strip_nans:
        mask = ~np.all(np.isnan(landmark_imcoords), axis=(0, 2, 3))
        landmark_imcoords = landmark_imcoords[:, mask]
        time_base = time_base[mask]

    if scalar_label:
        landmark_imcoords = landmark_imcoords[..., 0, :]

    return landmark_imcoords, time_base


def to_pandas(labels):
    if legacy := (Version(labels["version"]) < Version("1.0")):
        labels = convert_v0_to_v1(labels)

    import pandas as pd  # We don't want this as a global dependency
    label_names = get_labels(labels)
    data, time_base = to_numpy(labels)
    data_shape = data.shape
    data = data.transpose([1, 0, 2, 3]).reshape((data_shape[1], -1))

    columns = list([f"{s}_{co}"
                    for s, co in itertools.product([f"c{c:02d}_{l}"
                                                    for c, l in itertools.product(range(data_shape[0]), label_names)],
                                                   ["x", "y"])])

    return pd.DataFrame(data, index=time_base, columns=columns)


def write_label_yaml(file_handle, labels):
    if Version(labels["version"]) >= Version("1.0"):
        write_label_yaml_v2(file_handle, labels)
    else:
        write_label_yaml_v1(file_handle, labels)


def read_label_yaml(file_handle):
    if Version(read_version(file_handle)) >= Version("1.0"):
        return read_label_yaml_v2(file_handle)
    else:
        return read_label_yaml_v1(file_handle)


# pyyaml is WAY too slow for large files
def write_label_yaml_v1(file_handle, labels):
    f = file_handle

    f.write("version: ")
    f.write(str(labels["version"]))
    f.write("\n")

    f.write("labeler_list: [")
    f.write(", ".join(labels["labeler_list"]))
    f.write("]\n")

    f.write("labeler:\n")
    for ln in labels["labeler"]:
        ln_dict = labels["labeler"][ln]
        f.write("  ")
        f.write(ln)
        if len(ln_dict) == 0:
            f.write(": {}\n")
        else:
            f.write(":\n")
            for fr_idx in ln_dict:
                fr_list = ln_dict[fr_idx]
                f.write("    ")
                f.write(str(fr_idx))
                f.write(": [")
                if isinstance(fr_list, np.ndarray):
                    f.write(", ".join([str(f) for f in fr_list]))
                else:
                    f.write(str(fr_list))
                f.write("]\n")

    f.write("labels:\n")
    for ln in labels["labels"]:
        ln_dict = labels["labels"][ln]
        f.write("  ")
        f.write(ln)
        if len(ln_dict) == 0:
            f.write(": {}\n")
        else:
            f.write(":\n")
            for fr_idx in ln_dict:
                fr_dict = ln_dict[fr_idx]
                f.write("    ")
                f.write(str(fr_idx))
                f.write(":\n")
                for row in fr_dict:
                    f.write("    - [")
                    if np.isnan(row[0]):
                        f.write(".nan")
                    else:
                        f.write(str(row[0]))
                    f.write(", ")
                    if np.isnan(row[0]):
                        f.write(".nan")
                    else:
                        f.write(str(row[1]))
                    f.write("]\n")

    f.write("point_times:\n")
    for ln in labels["point_times"]:
        ln_dict = labels["point_times"][ln]
        f.write("  ")
        f.write(ln)
        if len(ln_dict) == 0:
            f.write(": {}\n")
        else:
            f.write(":\n")
            for fr_idx in ln_dict:
                t_list = ln_dict[fr_idx]
                f.write("    ")
                f.write(str(fr_idx))
                f.write(": [")
                if isinstance(t_list, np.ndarray):
                    f.write(", ".join([str(t) for t in t_list]))
                else:
                    f.write(str(t_list))
                f.write("]\n")


def read_label_yaml_v1(file):
    labels = {}
    current_key = None
    current_label = None
    current_frame = None
    in_line = False
    full_line = ""

    pattern = r"\d+(?:\.\d+)?"
    re_key_numlist = re.compile(pattern)

    for line in file.readlines():
        line_parts = None
        try:
            if in_line:
                full_line += line.strip()

                if current_key == "labeler_list":
                    if full_line.endswith("]"):
                        labels[current_key] = [ln.strip() for ln in full_line[1:-1].split(",")]
                        in_line = False
                    continue
                else:
                    raise Exception("Lines should only overflow in labeler_list", line)

            if not line.startswith(" "):
                line_parts = line.strip().split(" ")
                current_key = line_parts[0][:-1]
                labels[current_key] = {}

                current_line = "".join(line_parts[1:]).strip()
                if current_key == "version":
                    labels[current_key] = float(current_line)
                elif current_key == "labeler_list":
                    if current_line.endswith("]"):
                        labels[current_key] = [ln.strip() for ln in current_line[1:-1].split(",")]
                    else:
                        full_line = current_line
                        in_line = True
            elif line.startswith("  ") and line[2] != " ":
                # This can only happen for labeler, point_times, labels and mean the same in all cases
                current_label = line.strip().split(":")[0]
                labels[current_key][current_label] = {}
            elif line.startswith("    ") and line[4] != " ":
                line_parts = line.strip().split(" ")
                if line_parts[0] != "-":
                    current_frame = int(line_parts[0][:-1])

                if current_key == "labeler":
                    list_part = " ".join(line_parts[1:])
                    labels[current_key][current_label][current_frame] = np.array(
                        [int(x) for x in re_key_numlist.findall(list_part)])
                elif current_key == "point_times":
                    list_part = " ".join(line_parts[1:])
                    labels[current_key][current_label][current_frame] = np.array(
                        [float(x) for x in re_key_numlist.findall(list_part)])
                elif current_key == "labels":
                    if len(line_parts) == 1:
                        labels[current_key][current_label][current_frame] = np.zeros((0, 2))
                    elif line_parts[0] == "-":
                        coords = [line_parts[1][1:-1], line_parts[2][0:-1]]
                        coords = [np.nan if c == ".nan" else float(c) for c in coords]
                        labels[current_key][current_label][current_frame] = np.vstack(
                            (labels[current_key][current_label][current_frame],
                             np.array([coords])))
                    else:
                        raise Exception("We should never get here", line)
        except Exception as e:
            logger.log(logging.ERROR, "Current key:", current_key)
            logger.log(logging.ERROR, "Current label:", current_label)
            logger.log(logging.ERROR, "Current frame:", current_frame)
            logger.log(logging.ERROR, "In line:", in_line)
            logger.log(logging.ERROR, line)
            logger.log(logging.ERROR, line_parts)
            raise e

    return labels


# pyyaml is WAY too slow for large files
def write_label_yaml_v2(file_handle, labels):

    f = file_handle

    if "labeler" in labels:
        labels = convert_v0_to_v1(labels)

    yi = 2 * " "  # yaml_indent

    f.write("version: ")
    f.write(str(labels["version"]))
    f.write("\n")

    f.write("labeler_list: [")
    f.write(", ".join(labels["labeler_list"]))
    f.write("]\n")

    f.write("action_list: [")
    f.write(", ".join(labels["action_list"]))
    f.write("]\n")

    f.write("labels:\n")
    for ln in labels["labels"]:
        ln_dict = labels["labels"][ln]
        f.write(yi)
        f.write(ln)
        if len(ln_dict) == 0:
            f.write(": {}\n")
        else:
            f.write(":\n")
            for fr_idx in ln_dict:
                fr_dict = ln_dict[fr_idx]
                f.write(2 * yi)
                f.write(str(fr_idx))
                f.write(":\n")
                # coords
                f.write(3 * yi)
                f.write("crds:\n")
                for row in fr_dict["coords"]:
                    f.write(3 * yi)
                    f.write("- [")
                    if np.isnan(row[0]):
                        f.write(".nan")
                    else:
                        f.write(str(row[0]))
                    f.write(", ")
                    if np.isnan(row[0]):
                        f.write(".nan")
                    else:
                        f.write(str(row[1]))
                    f.write("]\n")
                # labeler
                f.write(3 * yi)
                f.write("lr: [")
                if isinstance(fr_dict["labeler"], np.ndarray):
                    f.write(", ".join([str(f) for f in fr_dict["labeler"]]))
                else:
                    f.write(str(fr_dict["labeler"]))
                f.write("]\n")
                # point_times
                f.write(3 * yi)
                f.write("pts: [")
                if isinstance(fr_dict["point_times"], np.ndarray):
                    f.write(", ".join([str(t) for t in fr_dict["point_times"]]))
                else:
                    f.write(str(fr_dict["point_times"]))
                f.write("]\n")
                # action
                if "action" in fr_dict:
                    f.write(3 * yi)
                    f.write("a: [")
                    if isinstance(fr_dict["action"], np.ndarray):
                        f.write(", ".join([str(f) for f in fr_dict["action"]]))
                    else:
                        f.write(str(fr_dict["action"]))
                    f.write("]\n")


def read_label_yaml_v2(file):
    labels = {}
    current_key = None
    current_label = None
    current_frame = None
    current_prop = None
    in_line = False
    full_line = ""

    yi = 2 * " "  # yaml_indent

    pattern = r"\d+(?:\.\d+)?"
    re_key_numlist = re.compile(pattern)

    for line in file.readlines():
        line_parts = None
        try:
            if in_line:
                full_line += line.strip()

                if current_key == "labeler_list" or current_key == "action_list":
                    if full_line.endswith("]"):
                        labels[current_key] = [ln.strip() for ln in full_line[1:-1].split(",")]
                        in_line = False
                    continue
                else:
                    raise Exception("Lines should only overflow in labeler_list", line)

            if not line.startswith(" "):
                line_parts = line.strip().split(" ")
                current_key = line_parts[0][:-1]
                labels[current_key] = {}

                current_line = "".join(line_parts[1:]).strip()
                if current_key == "version":
                    labels[current_key] = current_line
                elif current_key == "labeler_list" or current_key == "action_list":
                    if current_line.endswith("]"):
                        labels[current_key] = [ln.strip() for ln in current_line[1:-1].split(",")]
                    else:
                        full_line = current_line
                        in_line = True
            elif line.startswith(yi) and line[2] != " ":  # label indent
                # first level indent, label level
                current_label = line.strip().split(":")[0]
                labels[current_key][current_label] = {}
            elif line.startswith(2 * yi) and line[4] != " ":
                # second level indent, frame level
                current_frame = int(line.strip().split(":")[0])
                labels[current_key][current_label][current_frame] = {}
            elif line.startswith(3 * yi) and line[6] != " ":
                # third level indent, coords, labeler, point_times or list items
                line_parts = line.strip().split(" ")
                if line_parts[0] != "-":
                    current_prop = line_parts[0][:-1]

                if current_prop == "lr":
                    list_part = " ".join(line_parts[1:])
                    labels[current_key][current_label][current_frame]["labeler"] = np.array(
                        [int(x) for x in re_key_numlist.findall(list_part)])
                elif current_prop == "pts":
                    list_part = " ".join(line_parts[1:])
                    labels[current_key][current_label][current_frame]["point_times"] = np.array(
                        [float(x) for x in re_key_numlist.findall(list_part)])
                elif current_prop == "a":
                    list_part = " ".join(line_parts[1:])
                    labels[current_key][current_label][current_frame]["action"] = np.array(
                        [int(x) for x in re_key_numlist.findall(list_part)])
                elif current_prop == "crds":
                    if len(line_parts) == 1:
                        labels[current_key][current_label][current_frame]["coords"] = np.zeros((0, 2))
                    elif line_parts[0] == "-":
                        coords = [line_parts[1][1:-1], line_parts[2][0:-1]]
                        coords = [np.nan if c == ".nan" else float(c) for c in coords]
                        labels[current_key][current_label][current_frame]["coords"] = np.vstack(
                            (labels[current_key][current_label][current_frame]["coords"],
                             np.array([coords])))
                    else:
                        raise Exception("We should never get here", line)
        except Exception as e:
            logger.log(logging.ERROR, "Current key:", current_key)
            logger.log(logging.ERROR, "Current label:", current_label)
            logger.log(logging.ERROR, "Current frame:", current_frame)
            logger.log(logging.ERROR, "Current frame:", current_prop)
            logger.log(logging.ERROR, "In line:", in_line)
            logger.log(logging.ERROR, line)
            logger.log(logging.ERROR, line_parts)
            raise e

    return labels


def read_version(file_handle):
    pos = file_handle.tell()
    file_handle.seek(0)
    # Newer label files have version first, so this should be fast
    while line := file_handle.readline():
        if line.startswith("version:"):
            break

    file_handle.seek(pos)

    if line:
        return line[8:].strip()
    else:  # only 0.1 does not have a version at all
        return "0.1"
