import unittest

import numpy as np
import numpy.testing as testing
from scipy.spatial.transform import Rotation as R

import bbo.geometry as geometry


class TestRotations(unittest.TestCase):
    def test_rotation_insert(self):
        data = np.arange(4)
        insertion_indices = np.arange(5)
        insertion_data = -np.arange(5)
        np.testing.assert_equal(np.insert(data, insertion_indices, insertion_data),
                                geometry.rot_insert(data, insertion_indices, insertion_data))

class TestSimpleFunctions(unittest.TestCase):
    @staticmethod
    def run_on_vectors(v0, v1):
        v0, v1 = np.asarray(v0), np.asarray(v1)
        v0, v1 = v0 / np.linalg.norm(v0, axis=v0.ndim - 1, keepdims=True), \
                 v1 / np.linalg.norm(v1, axis=v1.ndim - 1, keepdims=True)
        rot = geometry.get_perpendicular_rotation(v0, v1)
        testing.assert_allclose(rot.apply(v0), v1, atol=1e-7)

    def test_get_perpendicular_rotation(self):
        gen = np.random.default_rng(1)

        TestSimpleFunctions.run_on_vectors((1, 0, 0), (1, 0, 0))
        TestSimpleFunctions.run_on_vectors((1, 0, 0), (0, 1, 0))
        TestSimpleFunctions.run_on_vectors((1, 0, 0), (-1, 1e-3, 0))

        for i in range(10):
            v0 = gen.normal(loc=0.0, scale=3.0, size=3)
            v1 = gen.normal(loc=0.0, scale=3.0, size=3)
            TestSimpleFunctions.run_on_vectors(v0, v1)

        TestSimpleFunctions.run_on_vectors(gen.normal(loc=0.0, scale=3.0, size=(2, 3)), gen.normal(loc=0.0, scale=3.0, size=(2, 3)))
        TestSimpleFunctions.run_on_vectors(gen.normal(loc=0.0, scale=3.0, size=(1, 3)), gen.normal(loc=0.0, scale=3.0, size=(2, 3)))


class TestCoordinateTransformations(unittest.TestCase):
    def test_coordinate_inverse(self):
        rnd = np.random.default_rng(1)
        for i in range(10):
            cart = [rnd.normal(loc=0, size=3)]
            sph = geometry.cart2spherical(cart)
            np.testing.assert_allclose(geometry.spherical2cart(sph), cart)

    def test_chain(self):
        rnd = np.random.default_rng(1)
        mirror0 = geometry.AffineTransformation(geometry.Mirror(normal=rnd.normal(size=3), tr=rnd.normal()))
        rigid0 = geometry.RigidTransform(rotation=R.from_rotvec(rnd.normal(size=3)), translation=rnd.normal(size=3))
        mirror1 = geometry.AffineTransformation(geometry.Reflection(normal=np.array([1, 0, 0])))
        point = np.array([1, 2, 3])
        ppoint = np.asarray([1, 2, 3, 1])
        np.testing.assert_allclose(mirror0.apply(point), (mirror0.as_matrix(shape=(3, 4)) @ ppoint))
        np.testing.assert_allclose(rigid0.apply(point), (rigid0.as_matrix(shape=(3, 4)) @ ppoint))
        np.testing.assert_allclose(mirror1.apply(point), (mirror1.as_matrix(shape=(3, 4)) @ ppoint))
        combined = geometry.RigidTransform.from_matrix((mirror0 * rigid0 * mirror1).as_matrix())

        np.testing.assert_allclose(combined.apply(point), mirror0.apply(rigid0.apply(mirror1.apply(point))))

    def test_chain2(self):
        def run_on_data(c2t, pom, normal, p):
            m = geometry.Mirror(point_on_mirror=pom, normal=normal)
            np.testing.assert_allclose((c2t * m).apply(p), c2t.apply(m.apply(p)))

        #Rigid Transformation
        c2mat = np.array([[0, 1, 0, 2],
                          [-1, 0, 0, 3],
                          [0, 0, 1, 4],
                          [0, 0, 0, 1]])
        run_on_data(
            c2t=geometry.RigidTransform.from_matrix(c2mat),
            pom=np.array([1, 2, 3]),
            normal=np.array([0, 1, 0]),
            p=np.array([1, 5, 8]))

        rnd = np.random.default_rng(1)
        run_on_data(
            c2t=geometry.RigidTransform(rotation=R.random(random_state=rnd),translation=rnd.normal(size=3)),
            pom=rnd.normal(size=3),
            normal=rnd.normal(size=3),
            p=rnd.normal(size=3))

class TestAffineTransformation(unittest.TestCase):
    def test_concatenation(self):
        rnd = np.random.default_rng(1)
        af0 = geometry.AffineTransformation(rnd.normal(size=(4,4)))
        af1 = geometry.AffineTransformation(rnd.normal(size=(4,4)))
        randvec = rnd.normal(size=3)
        np.testing.assert_allclose((af1 * af0).apply(randvec), af1.apply(af0.apply(randvec)))


    def test_constructor(self):
        rnd = np.random.default_rng(1)
        r = R.random()
        randvec = rnd.normal(size=3)
        np.testing.assert_allclose(geometry.AffineTransformation(r).apply(randvec), r.apply(randvec))


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
            testing.assert_allclose(reflection_0.as_matrix() @ reflection_1.as_matrix(), chained.as_matrix(),
                                    atol=0.00001)
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

    def test_line_mindist(self):
        # 2D
        line = geometry.Line((0, 1), (2, 1))
        self.assertEqual(line.calc_min_point_dist((2, 2)), 0)
        line = geometry.Line((0, 0), (1, 1))
        self.assertEqual(line.calc_min_point_dist((-1, 1)), np.sqrt(2))
        # 3D a line and a point
        line = geometry.Line((0, 0, 0), (1, 1, 1))
        self.assertAlmostEqual(line.calc_min_point_dist((0, 1, 1)), np.sqrt(2)/np.sqrt(3))
        # 3D a line and 2 points
        np.testing.assert_allclose(line.calc_min_point_dist([(0, 1, 1), (1, 1, -2)]),
                                   [np.sqrt(2/3), np.sqrt(6)])
        # 3D 2 lines and a point
        line = geometry.Line(np.zeros((2, 3)), np.arange(6).reshape((2, 3)))
        np.testing.assert_allclose(line.calc_min_point_dist((0, 1, 1)), [1/np.sqrt(5), np.sqrt(19/50)])
        # 3D 2 lines and 2 points
        np.testing.assert_allclose(line.calc_min_point_dist([(0, 1, 1), (1, 1, -2)]), [1/np.sqrt(5), np.sqrt(291/50)])
        # 3D 2 lines and 3 points
        line = geometry.Line([np.ones(3), np.zeros(3)], np.arange(6).reshape((2, 3)))
        np.testing.assert_allclose(line.calc_min_point_dist([(0, 1, 1), (1, 1, -2), (0, 1, 2)], outer=True),
                                   [[1, np.sqrt(9/5), np.sqrt(6/5)],
                                    [np.sqrt(19/50), np.sqrt(291/50), np.sqrt(54/50)]],
                                   atol=1e-5)

    def test_rigid_transform_concatenation(self):
        gen = np.random.default_rng(1)
        for i in range(10):
            tr0 = geometry.RigidTransform(rotation=R.from_rotvec(gen.normal(loc=0.0, scale=3.0, size=3)),
                                          translation=gen.normal(loc=0.0, scale=3.0, size=3))
            tr1 = geometry.RigidTransform(rotation=R.from_rotvec(gen.normal(loc=0.0, scale=3.0, size=3)),
                                          translation=gen.normal(loc=0.0, scale=3.0, size=3))
            testvec = gen.normal(loc=0.0, scale=3.0, size=3)
            testing.assert_allclose((tr0 * tr1).apply(testvec), tr0.apply(tr1.apply(testvec)))

    def test_rigid_transform_inverse(self):
        gen = np.random.default_rng(1)
        for i in range(10):
            tr0 = geometry.RigidTransform(rotation=R.from_rotvec(gen.normal(loc=0.0, scale=3.0, size=3)),
                                          translation=gen.normal(loc=0.0, scale=3.0, size=3))
            testvec = gen.normal(loc=0.0, scale=3.0, size=3)
            testing.assert_allclose(tr0.apply(tr0.inv().apply(testvec)), testvec)


if __name__ == '__main__':
    unittest.main()
