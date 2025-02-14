import unittest
import os
from bbo.yaml import load as yaml_load

from deep_eq import deep_eq


class TestSimpleRead(unittest.TestCase):
    def test_read_yaml(self):
        yaml_load(os.path.dirname(__file__) + "/../test/empty_file.yaml")

    def test_replace_keys(self):
        orig = yaml_load(os.path.dirname(__file__) + "/../test/variable.yaml")
        replaced = yaml_load(os.path.dirname(__file__) + "/../test/variable.yaml",
                  replace_dict={'foo': 'foo', 'batz': 'batz', 'bar': 'bar'}, exist_required=False)
        expected = yaml_load(os.path.dirname(__file__) + "/../test/variable_expected.yaml")
        assert deep_eq(orig, orig)
        assert not deep_eq(orig, replaced)
        assert deep_eq(replaced, expected)

if __name__ == '__main__':
    unittest.main()
