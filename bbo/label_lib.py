import os
from pathlib import Path
import numpy as np
import yaml

from bbo.exceptions import NoDataException


def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())


yaml.add_representer(np.ndarray, ndarray_representer)

version = 0.4


def update(labels, labeler="_unknown"):
    assert labels["version"] <= version, "Please update ACM traingui"

    # Before versioning
    if "version" not in labels or labels["version"] <= 0.2:
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
            if f_idx not in labels["fr_times"]:
                labels["fr_times"][f_idx] = 0

    if labels["version"] <= 0.3:
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
                point_times[ln][fr_idx] = np.ones(data_shape[0], dtype=np.uint64) * labels['fr_times'][fr_idx]

        labels["point_times"] = point_times
        labels["labeler"] = labeler
        labels.pop("fr_times")

    # Bring labeler list in shape (add specials etc)
    make_global_labeler_list([labels])

    labels["version"] = version
    return labels


def load(file_path):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.with_suffix(".yml").is_file():
        file_path = file_path.with_suffix(".yml")
        labels = load_raw_yaml(file_path)
    elif file_path.with_suffix(".npz").is_file():
        file_path = file_path.with_suffix(".npz")
        labels = np.load(file_path, allow_pickle=True)["arr_0"][()]
        print("WARNING: Loaded deprecated npz file!")
    else:
        raise FileNotFoundError(file_path.as_posix())

    labels = update(labels, labeler=file_path.parent.parent.stem)

    return labels


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


def save(file_path, labels, yml_only=False):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    os.makedirs(file_path.parent.as_posix(), exist_ok=True)

    if not yml_only:
        np.savez(file_path.with_suffix(".npz").as_posix(), labels)

    with open(file_path.with_suffix(".yml").as_posix(), 'w') as f:
        yaml.dump(labels, f, default_flow_style=None)


def get_labels(labels):
    return [lm for lm in labels["labels"] if len(labels["labels"][lm].keys()) > 0]


def get_labeled_frame_idxs(labels):
    frames = set()
    for ln in labels["labels"]:
        frames.update(set(labels["labels"][ln].keys()))

    return np.array(sorted(list(frames))).astype(int)


def get_labels_from_frame(labels, frame_idx):
    frame_labels = {}

    for lb in labels["labels"]:
        if frame_idx in labels["labels"][lb] and not np.all(np.isnan(labels["labels"][lb][frame_idx])):
            frame_labels[lb] = labels["labels"][lb][frame_idx]

    return frame_labels


def get_frame_labelers(labels, fr_idx, cam_idx=None):
    labeler_idxs = set()
    for ln in labels["labeler"]:
        if fr_idx in labels["labeler"][ln]:
            if cam_idx is None:
                labeler_idxs.update(labels["labeler"][ln][fr_idx])
            else:
                labeler_idxs.update(labels["labeler"][ln][fr_idx][(cam_idx,),])

    labelers = [labels["labeler_list"][i] for i in labeler_idxs]
    if "_unmarked" in labelers:
        labelers.pop(labelers.index("_unmarked"))
    return labelers


def merge(labels_list: list, target_file=None, overwrite=False, yml_only=False):
    # Load data from files
    labels_files = None
    if isinstance(labels_list[0], str):
        labels_files = [Path(lf).expanduser().resolve() for lf in labels_list]
    elif isinstance(labels_list[0], Path):
        labels_files = labels_list
    else:
        assert target_file is not None, "target_file is only supported if labels_list contains paths"

    # Normalize path of target_file
    if isinstance(target_file, str):
        target_file = Path(target_file).expanduser().resolve()

    # Add target files as first labels file if existing
    if target_file is not None and target_file.is_file():
        labels_files.insert(0, target_file)

    # Load data from labels files
    if labels_files is not None:
        labels_list = [load(lf.as_posix()) for lf in labels_files]

    make_global_labeler_list(labels_list)

    # Merge file-wise
    target_labels = labels_list[0]
    data_shape = get_data_shape(target_labels)

    for labels in labels_list[1:]:
        initialize_target(labels, target_labels, data_shape)

        for ln in labels["labels"]:
            for fr_idx in labels["labels"][ln]:
                target_cam_mask = target_labels["labels"][ln][fr_idx] != 0
                source_cam_mask = labels["labels"][ln][fr_idx] != 0
                source_newer_mask = target_labels["point_times"][ln][fr_idx] < labels["point_times"][ln][fr_idx]

                replace_mask = source_cam_mask & source_newer_mask

                if not overwrite:
                    replace_mask &= (~target_cam_mask)

                target_labels["labels"][ln][fr_idx][replace_mask] = labels["labels"][ln][fr_idx][replace_mask]
                target_labels["labeler"][ln][fr_idx][replace_mask] = labels["labeler"][ln][fr_idx][replace_mask]
                target_labels["point_times"][ln][fr_idx][replace_mask] = labels["point_times"][ln][fr_idx][replace_mask]

    sort_dictionaries(target_labels)

    if target_file is not None:
        save(target_file, target_labels, yml_only=yml_only)
    return target_labels


def combine_cams(labels_list: list, target_file=None, yml_only=False):
    # Normalize path of target_file
    if isinstance(target_file, str):
        target_file = Path(target_file).expanduser().resolve()

    # Load data from files
    for i_l, label in enumerate(labels_list):
        if isinstance(label, str):
            if label == "None":
                labels_list[i_l] = None
                continue
            labels_list[i_l] = Path(label)
        if isinstance(label, Path):
            labels_list[i_l] = labels_list[i_l].expanduser().resolve()
        labels_list[i_l] = load(labels_list[i_l].as_posix())

    make_global_labeler_list(labels_list)

    target_labels = get_empty_labels()
    target_labels['labeler_list'] = make_global_labeler_list(labels_list)

    data_shape = (len(labels_list), 2)

    for cam_idx, labels in enumerate(labels_list):
        if labels is None:
            continue

        # Walk through frames
        initialize_target(labels, target_labels, data_shape)

        for ln in labels["labels"]:
            for fr_idx in labels["labels"][ln]:
                target_labels["labels"][ln][fr_idx][cam_idx] = labels["labels"][ln][fr_idx]
                target_labels["point_times"][ln][fr_idx][cam_idx] = labels["point_times"][ln][fr_idx]
                target_labels["labeler"][ln][fr_idx][cam_idx] = labels["labeler"][ln][fr_idx]

    sort_dictionaries(target_labels)

    if target_file is not None:
        save(target_file, target_labels, yml_only=yml_only)
    return target_labels


def initialize_target(labels, target_labels, data_shape):
    # Walk through frames
    for ln in labels["labels"]:
        # Initialize label key
        if ln not in target_labels["labels"]:
            target_labels["labels"][ln] = {}
        if ln not in target_labels["labeler"]:
            target_labels["labeler"][ln] = {}
        if ln not in target_labels["point_times"]:
            target_labels["point_times"][ln] = {}
        for fr_idx in labels["labels"][ln]:
            # Initialize frame index
            if fr_idx not in target_labels["labels"][ln]:
                target_labels["labels"][ln][fr_idx] = \
                    np.full(data_shape, np.nan)
            if fr_idx not in target_labels["labeler"][ln]:
                target_labels["labeler"][ln][fr_idx] = \
                    np.full(data_shape[0], 0, dtype=np.uint16)
            if fr_idx not in target_labels["point_times"][ln]:
                target_labels["point_times"][ln][fr_idx] = \
                    np.full(data_shape[0], 0, dtype=np.uint64)


def sort_dictionaries(target_labels):
    # Sort dictionaries
    for label in target_labels["labeler"]:
        target_labels["labeler"][label] = dict(sorted(target_labels["labeler"][label].items()))
    target_labels["labeler"] = dict(sorted(target_labels["labeler"].items()))

    for label in target_labels["point_times"]:
        target_labels["point_times"][label] = dict(sorted(target_labels["point_times"][label].items()))
    target_labels["point_times"] = dict(sorted(target_labels["point_times"].items()))

    for label in target_labels["labels"]:
        target_labels["labels"][label] = dict(sorted(target_labels["labels"][label].items()))
    target_labels["labels"] = dict(sorted(target_labels["labels"].items()))


def make_global_labeler_list(labels_list):
    # This changes labeler_list in place!!!
    # Create a new global list of all labelers
    labeler_list_all = []
    for labels in labels_list:
        if labels is None:
            continue
        if "labeler_list" in labels:
            labeler_list_all += labels["labeler_list"]
    labeler_list_all += ["_unknown"]
    labeler_list_all += ["_unmarked"]

    # Make unique and sorted
    labeler_list_all = sorted(list(set(labeler_list_all)))
    # Get specials to the front
    labeler_list_all.pop(labeler_list_all.index("_unmarked"))
    labeler_list_all.pop(labeler_list_all.index("_unknown"))
    labeler_list_all.insert(0, "_unknown")
    labeler_list_all.insert(0, "_unmarked")

    # Rewrite to global index list
    for labels in labels_list:
        if labels is None:
            continue
        for ln in labels["labels"]:
            for fr_idx in labels['labels'][ln]:
                for i, labeler_idx in enumerate(labels['labeler'][ln][fr_idx]):
                    labeler = labels["labeler_list"][labeler_idx]
                    labels['labeler'][ln][fr_idx][i] = labeler_list_all.index(labeler)
        labels["labeler_list"] = labeler_list_all
    return labeler_list_all


def get_data_shape(labels):
    for ln in labels['labels']:
        for fr_idx in labels['labels'][ln]:
            return labels['labels'][ln][fr_idx].shape
    raise NoDataException("Did not find any data to determine data shape")


def get_n_cams(labels):
    return get_data_shape(labels)[0]


def get_empty_labels():
    return {
        'labels': {},
        'point_times': {},
        'labeler_list': ["_unmarked", "_unknown"],
        'labeler': {},
        'version': version,
    }


def to_array(labels,
             extract_frame_idxs=None, extract_labels=None,  # Extract only these parts of data
             time_bases=None,  # Rearrange after these time bases for cams
             label_identity=None  # Treat as identical labels (nanmean if both are labeled)
             ):
    cams_n = get_n_cams(labels)

    if extract_frame_idxs is None:
        extract_frame_idxs = get_labeled_frame_idxs(labels)

    if extract_labels is None:
        extract_labels = get_labels(labels)
    time_base = np.unique(np.concatenate(time_bases, axis=0))

    if time_bases is None:
        time_bases = [np.arange(len(extract_frame_idxs)) for _ in range(cams_n)]

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
                        cam_coords = labels["labels"][lm][fr_idx][i_cam]
                        if not np.any(np.isnan(cam_coords)):
                            # if (fr_idx == 7529 and i_cam == 1) or (fr_idx == 7572 and i_cam == 0):
                            #     print(f"Writing cam {i_cam}, frame {fr_idx} to {time_base_idx}")
                            landmark_imcoords[i_cam, time_base_idx, lm_idx] = (
                                np.nanmean(np.array(
                                    [cam_coords, landmark_imcoords[i_cam, time_base_idx, lm_idx]]
                                ), axis=0))
    return landmark_imcoords, time_base


