import unittest
import bbo.geometry as geometry
import numpy as np
from scipy.spatial.transform import Rotation as R
import numpy.testing as testing


class TestGeometryObjects(unittest.TestCase):
    def test_reflection(self):
        gen = np.random.default_rng(1)
        for i in range(10):
            reflection_0 = geometry.Reflection(normal=gen.normal(loc=0.0, scale=3.0, size=3))
            reflection_1 = geometry.Reflection(normal=gen.normal(loc=0.0, scale=3.0, size=3))
            testvec = gen.normal(loc=0.0, scale=3.0, size=3)
            result = reflection_0.apply(reflection_1.apply(testvec))
            chained = reflection_0 * reflection_1
            testing.assert_allclose(reflection_0.as_matrix() @ testvec, reflection_0.apply(testvec), atol=0.00001)
            testing.assert_allclose(chained.apply(testvec), result, atol=0.00001)
            testing.assert_allclose(reflection_0.as_matrix() @ reflection_1.as_matrix(), chained.as_matrix(), atol=0.00001)
        reflection_0 = geometry.Reflection(normal=(1, 0, 0))
        reflection_1 = geometry.Reflection(normal=(0, 1, 0))
        testvec = gen.normal(loc=0.0, scale=3.0, size=3)
        testing.assert_allclose((reflection_0 * reflection_1).apply(testvec),
                                reflection_0.apply(reflection_1.apply(testvec)), atol=0.00001)

    def test_mirror_rotation(self):
        m_orig = geometry.Mirror(normal=(1, 1, 1), point_on_mirror=(3, 4, 5))
        point = np.asarray((5, 3, 2))
        rotation = R.from_euler('xyz', (3, 132, 5))
        m_rotated = m_orig.rotate(rotation)
        testing.assert_allclose(m_orig.apply_on_point(point),
                                rotation.inv().apply(m_rotated.apply_on_point(rotation.apply(point))))

    def test_mirror_translation(self):
        m_orig = geometry.Mirror(normal=(1, 1, 1), point_on_mirror=(3, 4, 5))
        point = np.asarray((5, 3, 2))
        translation = np.asarray((3, 132, 5))
        m_translated = m_orig.translate(translation)
        testing.assert_allclose(m_orig.apply_on_point(point),
                                m_translated.apply_on_point(point + translation) - translation)

    def test_create_line(self):
        l0 = geometry.Line((0, 0, 0), (0, 0, 1))
        raveled = l0.ravel()
        raveled[1] = 5
        self.assertEqual(l0, geometry.Line((0, 5, 0), (0, 0, 1)))


if __name__ == '__main__':
    unittest.main()
