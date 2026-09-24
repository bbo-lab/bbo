import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bbo import label_lib


class TestPandasConversion(unittest.TestCase):
    def setUp(self):
        self.labels = {
            "version": "1.0",
            "labeler_list": ["_unmarked", "_unknown", "Alice #1"],
            "action_list": ["create", "delete"],
            "labels": {
                "nose_point_time": {
                    2: {"coords": np.array([[1., 2.], [3., 4.]]),
                        "labeler": np.array([2, 1]), "point_times": np.array([10., 11.])},
                    9: {"coords": np.full((2, 2), np.nan),
                        "labeler": np.array([2, 2]), "point_times": np.array([12., 13.]),
                        "action": np.array([1, 1])},
                },
                "tail": {
                    5: {"coords": np.array([[5., 6.], [np.nan, np.nan]]),
                        "labeler": np.array([2, 0]), "point_times": np.array([14., 0.]),
                        "action": np.array([0, 0])},
                },
                "empty": {},
            },
        }

    def assertLabelsEqual(self, actual, expected):
        for key in ("version", "labeler_list", "action_list"):
            self.assertEqual(actual[key], expected[key])
        self.assertEqual(actual["labels"].keys(), expected["labels"].keys())
        for name, frames in expected["labels"].items():
            self.assertEqual(actual["labels"][name].keys(), frames.keys())
            for frame, entry in frames.items():
                result = actual["labels"][name][frame]
                self.assertEqual(result.keys(), entry.keys())
                for field, value in entry.items():
                    np.testing.assert_equal(result[field], value)

    def test_layout_and_roundtrip(self):
        original = copy.deepcopy(self.labels)
        table = label_lib.to_pandas(self.labels)
        self.assertEqual(table.index.tolist(), [2, 5, 9])
        self.assertEqual(table.columns[:10].tolist(), [
            f"c{cam:02d}_nose_point_time_{field}" for cam in range(2)
            for field in ("x", "y", "labeler", "point_time", "action")])
        self.assertTrue(np.isnan(table.loc[2, "c00_nose_point_time_action"]))
        self.assertEqual(table.loc[9, "c01_nose_point_time_action"], 1)
        self.assertEqual(table.loc[2, "c00_nose_point_time_labeler"], 2)
        self.assertLabelsEqual(label_lib.from_pandas(table), self.labels)
        table.attrs["labeler_list"].append("Bob")
        self.assertLabelsEqual(self.labels, original)

    def test_per_camera_roundtrip(self):
        tables = label_lib.to_pandas(self.labels, per_cam=True)
        self.assertEqual(len(tables), 2)
        self.assertIn("nose_point_time_x", tables[0])
        self.assertLabelsEqual(label_lib.from_pandas(tables), self.labels)
        self.assertLabelsEqual(label_lib.from_pandas(tables, per_cam=True), self.labels)
        tables[0].attrs["labeler_list"].append("Bob")
        self.assertNotIn("Bob", tables[1].attrs["labeler_list"])

    def test_defaults_and_separately_missing_columns(self):
        table = pd.DataFrame({"c00_eye_x": [1., np.nan], "c00_eye_y": [2., np.nan]}, index=[3, 7])
        labels = label_lib.from_pandas(table)
        entry = labels["labels"]["eye"][3]
        self.assertEqual(labels["labeler_list"][entry["labeler"][0]], "_unknown")
        self.assertEqual(entry["point_times"][0], 0)
        self.assertNotIn("action", entry)
        self.assertNotIn(7, labels["labels"]["eye"])
        table = table.iloc[:1].copy()
        table["c00_eye_point_time"] = 123
        labels = label_lib.from_pandas(table, labeler="Bob", point_time=42)
        entry = labels["labels"]["eye"][3]
        self.assertEqual(labels["labeler_list"][entry["labeler"][0]], "Bob")
        self.assertEqual(entry["point_times"][0], 123)
        table = table.drop(columns="c00_eye_point_time")
        table["c00_eye_labeler"] = 0
        entry = label_lib.from_pandas(table, point_time=42)["labels"]["eye"][3]
        self.assertEqual(entry["labeler"][0], 0)
        self.assertEqual(entry["point_times"][0], 42)

    def test_independent_camera_indices_and_partial_actions(self):
        cameras = [pd.DataFrame({"eye_x": [1.], "eye_y": [2.], "eye_action": [1]}, index=[3]),
                   pd.DataFrame({"eye_x": [4.], "eye_y": [5.]}, index=[7])]
        labels = label_lib.from_pandas(cameras)
        frames = labels["labels"]["eye"]
        self.assertEqual(set(frames), {3, 7})
        np.testing.assert_equal(frames[3]["coords"], [[1., 2.], [np.nan, np.nan]])
        np.testing.assert_equal(frames[3]["action"], [1, 0])
        self.assertNotIn("action", frames[7])
        self.assertEqual(labels["labeler_list"][frames[3]["labeler"][1]], "_unmarked")

    def test_tsv_roundtrip(self):
        for per_cam in (False, True):
            with self.subTest(per_cam=per_cam):
                streams = [io.StringIO(), io.StringIO()] if per_cam else io.StringIO()
                label_lib.to_csv(self.labels, streams, per_cam=per_cam)
                for stream in streams if per_cam else [streams]:
                    text = stream.getvalue()
                    self.assertTrue(text.startswith('# version: "1.0"\n'))
                    self.assertIn('# labeler_list:', text)
                    self.assertIn('# action_list:', text)
                    self.assertIn('frame\t', text)
                    stream.seek(0)
                self.assertLabelsEqual(label_lib.from_csv(streams), self.labels)

    def test_tsv_paths_and_hash_in_label(self):
        self.labels["labels"]["#tail"] = self.labels["labels"].pop("tail")
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"camera{cam}.tsv" for cam in range(2)]
            label_lib.to_csv(self.labels, paths, per_cam=True)
            self.assertLabelsEqual(label_lib.from_csv(paths), self.labels)
            path = Path(directory) / "labels.tsv"
            label_lib.to_csv(self.labels, path)
            self.assertLabelsEqual(label_lib.from_csv(path), self.labels)

    def test_metadata_free_tsv_and_single_camera(self):
        labels = label_lib.from_csv(io.StringIO("frame\teye_x\teye_y\n4\t1\t2\n"),
                                    per_cam=True, labeler="Bob", point_time=12)
        entry = labels["labels"]["eye"][4]
        self.assertEqual(entry["coords"].shape, (1, 2))
        self.assertEqual(labels["labeler_list"][entry["labeler"][0]], "Bob")
        self.assertEqual(entry["point_times"][0], 12)

    def test_empty_labels(self):
        labels = label_lib.get_empty_labels()
        self.assertLabelsEqual(label_lib.from_pandas(label_lib.to_pandas(labels)), labels)
        self.assertLabelsEqual(label_lib.from_pandas([]), labels)
        stream = io.StringIO()
        label_lib.to_csv(labels, stream)
        stream.seek(0)
        self.assertLabelsEqual(label_lib.from_csv(stream), labels)

    def test_invalid_input(self):
        table = label_lib.to_pandas(self.labels)
        for invalid in (table.drop(columns="c00_tail_y"),
                        table.rename(index={2: 2.5}),
                        pd.concat([table, table])):
            with self.subTest(columns=invalid.columns.tolist()):
                with self.assertRaises(ValueError):
                    label_lib.from_pandas(invalid)
        for value in (-1, 0.5, 99):
            invalid = table.copy()
            invalid.loc[2, "c00_nose_point_time_labeler"] = value
            with self.assertRaises(ValueError):
                label_lib.from_pandas(invalid)
        cameras = label_lib.to_pandas(self.labels, per_cam=True)
        cameras[1].attrs["action_list"] = ["delete", "create"]
        with self.assertRaises(ValueError):
            label_lib.from_pandas(cameras)


if __name__ == "__main__":
    unittest.main()
