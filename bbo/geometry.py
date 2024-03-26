import numpy as np
from scipy.spatial.transform import Rotation


def get_mirror_matrix(normal):
    normal = np.asarray(normal)
    H = np.eye(len(normal)) - 2 * np.outer(normal, normal.T) / np.linalg.norm(normal) ** 2
    return H


class RaveledLine:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction

    def __getitem__(self, index):
        if index % 6 < 3:
            array = self.position
        else:
            array = self.direction
        return array.ravel()[index - (index // 6) * 3]

    def __setitem__(self, index, value):
        if index % 6 < 3:
            array = self.position
        else:
            array = self.direction
        array.ravel()[index - (index // 6) * 3] = value


class ElementRef(object):
    def __init__(self, obj, key):
        self.obj = obj
        self.key = key

    @property
    def value(self): return self.obj[self.key]

    @value.setter
    def value(self, value): self.obj[self.key] = value


def from_quat_rot(quats):
    mask = ~np.any(np.isnan(quats), axis=1)
    res = Rotation.from_rotvec(np.full(shape=(len(mask), 3), fill_value=np.nan))
    res[mask] = Rotation.from_quat(quats[mask])
    return res


def from_rotvec_rot(rotvecs):
    mask = ~np.any(np.isnan(rotvecs), axis=1)
    res = Rotation.from_rotvec(np.full(shape=(len(mask), 3), fill_value=np.nan))
    res[mask] = Rotation.from_rotvec(rotvecs[mask])
    return res


@staticmethod
def smoothrot(r, kernel=[1, 2, 1]):
    kernel_len = len(kernel)
    res = Rotation.identity(len(r))
    for i in range(len(r) - kernel_len):
        res[i] = mean_rot(r[i:i + kernel_len], weights=kernel)
    return res


@staticmethod
def isnan_rot(r):
    return np.isnan(r.magnitude())


@staticmethod
def inverse_rot(r):
    mask = np.logical_not(isnan_rot(r))
    if isinstance(mask, np.ndarray):
        res = Rotation.from_rotvec(np.full(fill_value=np.nan, shape=(len(r), 3)))
        res[mask] = r[mask].inv()
        return res
    if mask:
        return r.inv()
    if not mask:
        return r
    raise Exception(f'Input type not supported {type(r)}')


@staticmethod
def rotation_gradient(rot):
    inv_rot = inverse_rot(rot)
    res = multiply_rot(rot[:-2], inv_rot[2:])
    res = np.asarray(
        [(multiply_rot(rot[0], inv_rot[1])).as_rotvec(), *res.as_rotvec() / 2,
         (multiply_rot(rot[-1], inv_rot[-2])).as_rotvec()])
    return Rotation.from_rotvec(res)


@staticmethod
def mean_rot(r, weights=None, ignore_nan=True):
    mask = np.logical_not(isnan_rot(r))
    if np.any(mask) if ignore_nan else np.all(mask):
        return r[mask].mean(weights=None if weights is None else np.asarray(weights)[mask])
    return Rotation.from_rotvec(np.full(fill_value=np.nan, shape=3))


@staticmethod
def multiply_rot(lhs, rhs):
    lmask, rmask = isnan_rot(lhs), isnan_rot(rhs)
    mask = ~np.logical_or(lmask, rmask)
    if isinstance(mask, np.ndarray):
        res = Rotation.from_rotvec(np.full(fill_value=np.nan, shape=(len(mask), 3)))
        if not np.any(mask):
            return res
        if isinstance(lmask, np.ndarray):
            lhs = lhs[mask]
        if isinstance(rmask, np.ndarray):
            rhs = rhs[mask]
        res[mask] = lhs * rhs
        return res
    if mask:
        return lhs * rhs
    if not mask:
        return Rotation.from_rotvec(np.full(fill_value=np.nan, shape=3))
    raise Exception(f'Input type not supported {type(lhs)} {type(rhs)}')


@staticmethod
def apply_rot(r, vec):
    mask = ~isnan_rot(r)
    if isinstance(mask, np.ndarray):
        res = np.full(shape=(len(mask), 3), fill_value=np.nan)
        if len(vec.shape) == 2:
            vec = vec[mask]
        res[mask] = r[mask].apply(vec)
        return res
    if mask:
        return r.apply(vec)
    return np.full(shape=vec.shape, fill_value=np.nan)


def divlim(divident, divisor):
    return np.divide(divident, divisor, np.zeros_like(divident), where=np.logical_or(divident!=0,divisor!=0))


def cart2equidistant(vec, cart="xyz", equidist="rxy", invertaxis="", degrees=False):
    """
    Converts cartesian to equidistant coordinates
    """
    vec = np.asarray(vec).T[([cart.index(a) for a in "xyz"],)]
    for a in invertaxis:
        idx = "xyz".index(a)
        vec[idx] = -vec[idx]
    x,y,z = vec
    length = np.square(x) + np.square(y)
    norm = np.sqrt(length + np.square(z))
    length = np.sqrt(length)
    length = np.arctan2(length, -z) / length;
    res = [norm, x * length, y * length]
    if degrees:
        res[1] = np.rad2deg(res[1])
        res[2] = np.rad2deg(res[2])
    return np.asarray([res["rxy".index(a)] for a in equidist]).T


def equidist2cart(vec, cart="xyz", equidist="rxy", invertaxis="", degrees=False):
    """
    Converts equidistant to cartesian coordinates
    """
    vec = np.asarray(vec).T
    indices = [equidist.find(a) for a in "rxy"]
    radius = vec[indices[0]] if indices[0] > -1 else 1
    xequidist, yequidist = vec[indices[1]], vec[indices[2]]
    if degrees:
        xequidist = np.deg2rad(xequidist)
        yequidist = np.deg2rad(yequidist)
    radius2d = np.sqrt(np.square(xequidist) + np.square(yequidist))
    div = divlim(np.sin(radius2d) * radius, radius2d)
    res = [div * xequidist, div * yequidist, -np.cos(radius2d) * radius]
    for a in invertaxis:
        idx = "xyz".index(a)
        res[idx] = -res[idx]
    return np.asarray([res["xyz".index(a)] for a in cart]).T


def spherical2cart(vec, cart="xyz", sph="ria", invertaxis="", center_inclination=False, degrees=False):
    """
    Converts spherical to cartesian coordinates according to physical definition
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions.
    Norm gives the length of the incoming vector
    Inclination gives the rotaton for x-y-plane to z
    Azimuth rotates around the Longitude on the x-y-plane
    Parameters
    ----------
    vec
    cart
    sph
    degrees
    """
    vec = np.asarray(vec).T
    indices = [sph.find(a) for a in "ria"]
    radius = vec[indices[0]] if indices[0] > -1 else 1
    inclination, azimuth = vec[indices[1]], vec[indices[2]]
    if degrees:
        inclination = np.deg2rad(inclination)
        azimuth = np.deg2rad(inclination)
    if center_inclination:
        inclination += np.pi / 2
    rsin = radius * np.sin(inclination)
    res = [rsin * np.cos(azimuth), rsin * np.sin(azimuth), radius * np.cos(inclination)]
    for a in invertaxis:
        idx = "xyz".index(a)
        res[idx] = -res[idx]
    return np.asarray([res["xyz".index(a)] for a in cart]).T


def cart2spherical(vec, cart="xyz", sph="ria", invertaxis="", center_inclination=False, degrees=False):
    """
    Converts cartesian to spherical coordinates according to physical definition
    https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions.
    Norm gives the length of the incoming vector
    Inclination rotates around the Longitude on the x-y-plane
    Elevation gives the rotaton for x-y-plane to z
    Parameters
    ----------
    vec
    cart
    sph
    degrees

    Returns

    """
    vec = np.asarray(vec).T[([cart.index(a) for a in "xyz"],)]
    for a in invertaxis:
        idx = "xyz".index(a)
        vec[idx] = -vec[idx]
    x, y, z = vec
    sum = np.square(x) + np.square(y)
    azimuth = np.arctan2(y, x)
    inclination = np.arctan2(np.sqrt(sum), z)
    if center_inclination:
        inclination -= np.pi / 2
    if degrees:
        azimuth = np.rad2deg(azimuth)
        inclination = np.rad2deg(inclination)
    res = [np.sqrt(sum + np.square(z)), inclination, azimuth]
    return np.asarray([res["ria".index(a)] for a in sph]).T


class RigidTransform:
    def __init__(self, rotation=Rotation.identity(), translation=np.zeros(shape=(1, 3))):
        self.rotation = rotation
        self.translation = translation

    def apply(self, vec):
        return self.rotation.apply(vec) + self.translation

    def __getitem(self, key):
        return RigidTransform(rotation=self.rotation[key], translation=self.translation[key])

    def __setitem__(self, index, value):
        self.rotation[index] = value.rotation
        self.translation[index] = value.translation

    def __len__(self):
        return len(self.rotation)

    def __eq__(self, other):
        if isinstance(other, RigidTransform):
            return np.array_equal(self.rotation, other.rotation) and np.array_equal(self.translation, other.translation)
        return False

    def __mul__(self, other):
        if isinstance(other, RigidTransform):
            return RigidTransform(rotation=multiply_rot(self.rotation, other.rotation),
                                  translation=self.rotation.apply(other.translation) + self.translation)

    def inv(self):
        rotinv = inverse_rot(self.rotation)
        return RigidTransform(rotation=rotinv, translation=-rotinv.apply(self.translation))

    def get_rotation(self):
        return self.rotation

    def get_translation(self):
        return self.translation


class Line:
    def __init__(self, position, direction):
        self.position = np.asarray(position, dtype=np.float64)
        self.direction = np.asarray(direction, dtype=np.float64)
        if not np.array_equal(self.position.shape, self.direction.shape):
            raise Exception("Shapes have to be equal")
        self.shape = self.position.shape

    def transform_internal_array(self, functional):
        self.position = functional(self.position)
        self.direction = functional(self.direction)

    def __repr__(self):
        return f"{self.position} + t * {self.direction}"

    def __str__(self):
        return f"{self.position} + t * {self.direction}"

    def __getitem__(self, key):
        return Line(position=self.position[key], direction=self.direction[key])

    def __setitem__(self, index, value):
        self.position[index] = value.position
        self.direction[index] = value.direction

    def is_single(self):
        return self.position.ndim == 1

    def __len__(self):
        if self.is_single():
            # TODO: retun 1?
            raise Exception('Not an array')
        return self.position.shape[0]

    def __eq__(self, other):
        if isinstance(other, Line):
            return np.array_equal(self.position, other.position) and np.array_equal(self.direction, other.direction)
        return False

    def calc_min_point_dist(self, x, outer=False) -> np.ndarray or float:
        """Return the minimum distances of points from line(s)
        :param x: Nx3 array of points
        :param outer: if True, return the distance matrix
        :return: (N,) array of distances if N matches the number of lines and outer is False
        :return: (N, M) array of distances if N is different from M (number of lines) and outer is True
        """

        self.normalize()
        p = self.position
        v = self.direction
        x = np.asarray(x, dtype=np.float64)
        if self.is_single() and x.ndim > 1:
            p = p[np.newaxis, :]
            v = v[np.newaxis, :]
        if outer:
            p = p[:, np.newaxis, :]
            v = v[:, np.newaxis, :]
        d = x - p
        proj_vecs = d - np.sum(d * v, axis=-1, keepdims=True) * v
        dists = np.linalg.norm(proj_vecs, axis=proj_vecs.ndim - 1)
        return dists

    def normalize(self):
        self.direction /= np.linalg.norm(self.direction, axis=-1, keepdims=True)

    def copy(self):
        return Line(position=np.copy(self.position), direction=np.copy(self.direction))

    def ravel(self, index=None):
        if index is not None:
            if index % 6 < 3:
                array = self.position
            else:
                array = self.direction
            return ElementRef(obj=array.ravel(), key=index - (index // 6) * 3)
        return RaveledLine(position=self.position, direction=self.direction)


def get_homogenuous_transformation_from_mirror(normal, point_on_mirror):
    dim = len(normal)
    H = get_mirror_matrix(normal)
    A = np.hstack((H, 2 * np.asarray(point_on_mirror).reshape(-1, 1)))
    T = np.zeros(shape=(dim + 1,))
    T[-1] = 1
    return np.vstack((A, T.reshape(1, -1)))


def get_orthogonal(v):
    v = np.asarray(v)
    x = np.random.randn(3)  # take a random vector
    x -= x.dot(v) * v  # make it orthogonal to k
    x /= np.linalg.norm(x)  # normalize it
    y = np.cross(v, x)
    return np.vstack((v, x, y))


def get_perpendicular_rotation(source, dest, normalize=False):
    """
    Number of source and dest have to match
    Parameters
    ----------
    source
    dest
    normalize

    Returns
    -------

    """
    sounce = np.asarray(source)
    dest = np.asarray(dest)
    if normalize:
        source = source / np.linalg.norm(source, axis=-1, keepdims=True)
        dest = dest / np.linalg.norm(dest, axis=-1, keepdims=True)
    rotvec = np.cross(source, dest)
    norm = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    dot = np.sum(source * dest,axis=-1, keepdims=True)
    rotvec *= np.divide(np.arccos(dot), norm, where=norm > 1e-10)
    return Rotation.from_rotvec(rotvec, degrees=False)


def get_distances(points, v0, v1, tr):
    t = np.zeros(len(points))
    points = points - tr[np.newaxis, :]


class LinearTransformation:
    def __init__(self, mat):
        self.mat = mat

    def inv(self):
        return LinearTransformation(np.linalg.inv(self.mat))

    def apply(self, points):
        return points @ self.mat.T

    def __mul__(self, other):
        if isinstance(other, AffineTransformation):
            return AffineTransformation(self) * other
        return LinearTransformation(self.mat @ other.mat)


class AffineTransformation:
    def __init__(self, mat):
        if isinstance(mat, Rotation):
            self.mat = np.eye(4, dtype=float)
            self.mat[0:3, 0:3] = mat.as_matrix()
        elif isinstance(mat, LinearTransformation):
            self.mat = np.eye(4, dtype=float)
            self.mat[0:3, 0:3] = mat.mat
        elif isinstance(mat, AffineTransformation):
            self.mat = mat.mat
        else:
            self.mat = np.copy(mat)
        if not isinstance(self.mat, np.ndarray):
            raise Exception(f'Wrong matrix type {type(self.mat)}')
        self.mat[3, 0:3] = 0
        self.mat[3, 3] = 1

    def inv(self):
        return AffineTransformation(np.linalg.inv(self.mat))

    def apply(self, points):
        return (points @ self.mat[0:3, 0:3].T) + self.mat[0:3, 3]

    def linear(self):
        return LinearTransformation(self.mat[0:3, 0:3])

    def __mul__(self, other):
        if not isinstance(other, AffineTransformation):
            other = AffineTransformation(other)
        return AffineTransformation(self.mat @ other.mat)


class Reflection:
    def __init__(self, normal):
        normal = np.array(normal, copy=True)
        self.normal = normal / np.linalg.norm(normal, axis=0)

    def __repr__(self):
        return f"{self.normal} * x = 0"

    def __str__(self):
        return f"{self.normal} * x = 0"

    def apply(self, vec):
        return vec - np.inner(self.normal, vec) * self.normal * 2

    def as_matrix(self):
        return np.eye(len(self.normal)) - 2 * np.outer(self.normal, self.normal)

    def __mul__(self, other):
        if isinstance(other, Reflection):
            rotvec = np.cross(other.normal, self.normal)
            norm = np.linalg.norm(rotvec)
            rotvec *= np.where(np.inner(other.normal, self.normal) > 0, 1, -1) * 2 * np.arcsin(norm) / norm
            return Rotation.from_rotvec(rotvec, degrees=False)
        raise Exception(F'Type {type(other)} not supported')


class Mirror:
    def __init__(self, normal, point_on_mirror=None, tr=None):
        normal = np.asarray(normal)
        normal = normal / np.linalg.norm(normal)
        self.normal = normal

        if point_on_mirror is not None:
            self.point_on_mirror = point_on_mirror
            self.tr = np.dot(self.normal, self.point_on_mirror)

        if tr is not None:
            self.tr = tr
            self.point_on_mirror = self.normal * self.tr

        self.M = get_mirror_matrix(self.normal)
        self.translation = np.asarray(self.point_on_mirror) * 2
        self.HomTr = get_homogenuous_transformation_from_mirror(self.normal, self.point_on_mirror)

    def __mul__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        A new mirror M'=MR^-1
        """
        if isinstance(other, Rotation):
            other = Rotation.from_rotvec(other.as_rotvec() * 0.5)
            return Mirror(normal=other.apply(self.normal), point_on_mirror=other.apply(self.point_on_mirror))
        if isinstance(other, Mirror):
            return Rotation.from_rotvec(np.cross(self.normal, other.normal))
        raise Exception(F'Type {type(other)} not supported')

    def rotate(self, rot):
        """
        Parameters
        rot

        Returns
        A mirror in another coordinate system, M'=RMR^-1
        """
        return Mirror(normal=rot.apply(self.normal), point_on_mirror=rot.apply(self.point_on_mirror))

    def translate(self, tr):
        return Mirror(normal=np.copy(self.normal), point_on_mirror=self.point_on_mirror + tr)

    def set_tr(self, tr):
        self.__init__(self.normal, tr=tr)

    def apply_on_point(self, points):
        res = np.dot(self.M, points.T).T + (self.normal * (self.tr * 2))[np.newaxis, :]
        return res

    def apply_on_vector(self, vectors):
        return np.dot(self.M, vectors.T).T

    def __len__(self):
        if len(self.normal.shape) == 1:
            raise Exception('Not an array')
        return self.normal.shape[0]

    def apply_on_line(self, line):
        return self.apply_on_vector(line[0]), self.apply_on_point(line[1])

    def apply_on_line(self, line: Line):
        return Line(direction=self.apply_on_vector(line.direction), position=self.apply_on_point(line.position))

    def copy(self):
        return Mirror(normal=np.copy(self.normal), tr=np.copy(self.tr))

    def line_distance(self, line: Line):
        return (self.tr - np.dot(self.normal, line.position)) / np.dot(self.normal, line.direction.T)

    def distance_to_mirror_plane(self, vectors):
        return np.dot(self.normal, vectors) - self.tr

    def find_intersection(self, directions, translation):
        directions = np.asarray(directions)
        translation = np.asarray(translation)
        return translation[:, np.newaxis] - directions.T * self.distance_to_mirror_plane(translation) / np.dot(
            self.normal, directions.T)

    #    def find_intersection(self, line:Line):
    #        return self.find_intersection(line.directions, line.translations)

    def get_distmat(self, lines):
        mirror_3d_vertices = self.find_intersection(lines[0], lines[1])
        return np.linalg.norm(mirror_3d_vertices[:, np.newaxis, :] - mirror_3d_vertices[:, :, np.newaxis], axis=0)

    def optimize_from_lines(self, lines, distances):
        distances = np.asarray(distances)
        from scipy.optimize import minimize

        def mirror_sqerror(translation):
            self.set_tr(translation)
            error = np.nansum(np.square(self.get_distmat(lines) - distances))
            return error

        res = minimize(mirror_sqerror, x0=self.tr)
        self.set_tr(res.x[0])

    def __repr__(self):
        return f"{self.normal} * x = {self.tr}"

    def __str__(self):
        return f"{self.normal} * x = {self.tr}"
