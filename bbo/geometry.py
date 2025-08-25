from __future__ import annotations

import numpy as np
import scipy.spatial.transform
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import RotationSpline

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
    if np.any(mask):
        res[mask] = Rotation.from_rotvec(rotvecs[mask])
    return res


def angle_between(u, v, axis=-1, normalize=True):
    """
    Return the angle(s) between two n-dimensional vectors in radians.

    Uses a numerically stable arctan2 formulation:
        θ = arctan2(‖u - v‖ · ‖u + v‖, 2 · (u·v))

    Parameters
    ----------
    u : array_like
        First input vector(s).
    v : array_like
        Second input vector(s). Must be broadcast-compatible with `u`.
    axis : int, optional
        Axis along which the vector components are stored (default -1).

    Returns
    -------
    angles : ndarray or float
        Angle(s) in radians between `u` and `v`, in [0, π].
        :param normalize: If True, normalize the input vectors before calculating the angle.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    if normalize:
        u = u / np.linalg.norm(u, axis=axis, keepdims=True)
        v = v / np.linalg.norm(v, axis=axis, keepdims=True)

    dot = np.sum(u * v, axis=axis)
    #Stable way to compute sqrt(||u||²||v||² - (u·v)²)
    num = np.linalg.norm(u - v, axis=axis) * np.linalg.norm(u + v, axis=axis)
    return np.arctan2(num, 2 * dot)


def smoothrot(r, kernel=(1, 2, 1)):
    kernel_len = len(kernel)
    res = Rotation.identity(len(r))
    for i in range(len(r) - kernel_len):
        res[i] = mean_rot(r[i:i + kernel_len], weights=kernel)
    return res


#Matrix shape is (...,3,3)
def rot_from_nanmatrix(matrices):
    if matrices.ndim == 2:
        if np.any(np.isnan(matrices)):
            return get_nan_rot()
        return Rotation.from_matrix(matrices)
    if matrices.ndim == 3:
        mask = ~np.any(np.isnan(matrices), axis=(-2, -1))
        res = get_nan_rot(len(matrices))
        if np.any(mask):
            res[mask] = Rotation.from_matrix(matrices[mask])
        return res

def isnan_rot(r, inverted=False):
    result = np.isnan(r.magnitude())
    if inverted:
        np.logical_not(result, out=result)
    return result

def rot_insert(arr, obj, values):
    if isinstance(arr, Rotation):
        result = get_nan_rot(len(arr) + len(obj))
    else:
        #Only for pytest
        result = np.empty(shape=(len(arr) + len(obj)))
    mask = np.ones(shape=len(arr), dtype=bool)
    mask = np.insert(mask, obj, False)
    result[mask] = arr
    result[~mask] = values
    return result


def nanmean_rot(r):
    return r[isnan_rot(r, inverted=True)].mean()

def from_euler_rot(seq, angles, degrees=False):
    res = get_nan_rot(len(angles))
    mask = ~np.isnan(angles)
    if len(mask.shape) == 2:
        mask = np.all(mask, axis=-1)
    res[mask] = Rotation.from_euler(seq, angles[mask], degrees=degrees)
    return res

def inverse_rot(r):
    mask = np.logical_not(isnan_rot(r))
    if isinstance(mask, np.ndarray):
        res = get_nan_rot(len(r))
        if np.count_nonzero(mask) != 0:
            res[mask] = r[mask].inv()
        return res
    if mask:
        return r.inv()
    if not mask:
        return r
    raise Exception(f'Input type not supported {type(r)}')


def rotation_gradient(rot):
    inv_rot = inverse_rot(rot)
    res = multiply_rot(rot[:-2], inv_rot[2:])
    res = np.asarray(
        [(multiply_rot(rot[0], inv_rot[1])).as_rotvec(), *res.as_rotvec() / 2,
         (multiply_rot(rot[-1], inv_rot[-2])).as_rotvec()])
    return Rotation.from_rotvec(res)


def mean_rot(r, weights=None, ignore_nan=True):
    mask = ~isnan_rot(r)
    if np.any(mask) if ignore_nan else np.all(mask):
        return r[mask].mean(weights=None if weights is None else np.asarray(weights)[mask])
    return Rotation.from_rotvec(np.full(fill_value=np.nan, shape=3))


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
    return np.divide(divident, divisor, np.zeros_like(divident), where=np.logical_or(divident != 0, divisor != 0))


def cart2equidistant(vec, cart="xyz", equidist="rxy", invertaxis="", degrees=False):
    """
    Converts cartesian to equidistant coordinates
    """
    vec = np.moveaxis(np.asarray(vec),-1,0)
    vec = vec[([cart.index(a) for a in "xyz"],)]
    for a in invertaxis:
        idx = "xyz".index(a)
        vec[idx] = -vec[idx]
    x, y, z = vec
    length = np.square(x) + np.square(y)
    norm = np.sqrt(length + np.square(z))
    length = np.sqrt(length)
    length = np.divide(np.arctan2(length, -z), length, where=length!=0)
    res = [norm, x * length, y * length]
    if degrees:
        res[1] = np.rad2deg(res[1])
        res[2] = np.rad2deg(res[2])
    return np.stack([res["rxy".index(a)] for a in equidist], axis=-1)


def smooth(data, times, sigma, num_neighbors):
    if sigma is None or sigma == 0:
        return data
    if isinstance(data, Rotation):
        num_elements = len(data)
        #Equivalent but slower path for Rotations
        res = Rotation.identity(len(times))
        for i in range(num_elements):
            lhs = max(0, i - num_neighbors // 2)
            rhs = min(num_elements, i + num_neighbors // 2)
            kernel = np.exp(-np.square(times[lhs:rhs] - times[i]) / (2 * sigma ** 2))
            res[i] = mean_rot(data[lhs: rhs], weights=kernel)
        return res
    num_elements = data.shape[0]

    accumulated_weights = np.full(shape = num_elements, dtype=float, fill_value=1)
    result = np.copy(data)
    factor = 1 / (2 * sigma ** 2)
    expand_dims = (...,) + (np.newaxis,) * (data.ndim - 1)

    for shift in range(1, min(num_neighbors // 2, len(data)-1)):
        weights = np.exp(-np.square(times[:-shift] - times[shift:]) * factor)
        accumulated_weights[:-shift] += weights
        accumulated_weights[shift:] += weights
        result[:-shift] += weights[expand_dims] * data[shift:]
        result[shift:] += weights[expand_dims] * data[:-shift]
    result /= accumulated_weights[expand_dims]
    return result


def equidist2cart(vec, cart="xyz", equidist="rxy", invertaxis="", degrees=False):
    """
    Converts equidistant to cartesian coordinates
    """
    vec = np.moveaxis(np.asarray(vec), -1,0)
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
    return np.stack([res["xyz".index(a)] for a in cart], axis=-1)


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
    vec = np.moveaxis(np.asarray(vec), -1,0)
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
    return np.stack([res["xyz".index(a)] for a in cart], axis=-1)


def cart2spherical(vec, cart:str="xyz", sph:str="ria", invertaxis:str="", center_inclination:bool=False, degrees:bool=False, xp=np):
    """
    Converts cartesian to spherical coordinates according to physical definition
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

    Returns

    """
    vec = xp.moveaxis(xp.asarray(vec), -1, 0)
    vec = vec[([cart.index(a) for a in "xyz"],)]
    for a in invertaxis:
        idx = "xyz".index(a)
        vec[idx] = -vec[idx]
    x, y, z = vec
    sum = xp.square(x) + xp.square(y)
    azimuth = xp.arctan2(y, x)
    inclination = xp.arctan2(xp.sqrt(sum), z)
    if center_inclination:
        inclination -= xp.pi / 2
    if degrees:
        azimuth = xp.rad2deg(azimuth)
        inclination = xp.rad2deg(inclination)
    res = [xp.sqrt(sum + xp.square(z)), inclination, azimuth]
    return xp.stack([res["ria".index(a)] for a in sph], axis=-1)


def flip_quaternions(rot):
    flips = (np.sum(rot[:-1] * rot[1:], axis=-1) < 0).astype(int)
    flips = np.cumsum(flips) % 2
    flips = np.insert(flips, 0, 0)
    flips = np.round(flips * 2 - 1)
    return rot * flips[:, np.newaxis]


class RigidTransform:
    def __init__(self, rotation: Rotation|np.ndarray = Rotation.identity(), translation=None, rotation_type=None):
        if translation is None:
            translation = np.zeros(shape=(1, 3))
        if isinstance(rotation, np.ndarray):
            rs = rotation.shape
            # TODO: Use translation length for further identification if possible
            if rotation_type == "quaternion" or rs[-1] == 4:
                rotation = Rotation.from_quat(rotation)
            elif rotation_type == "matrix" or len(rs) == 3 and rs[-1] == 3 and rs[-2] == 3:
                rotation = Rotation.from_matrix(rotation)
            elif rotation_type == "rotvec" or len(rs) == 2 and rs[-1] == 3 and not rs[-2] == 3:
                # Note that this indistinguishable form euler, which we generally do not support
                rotation = Rotation.from_rotvec(rotation)
            elif len(rs) == 1 and rs[-1] == 3:
                # Note that this indistinguishable form euler, which we generally do not support
                rotation = Rotation.from_rotvec(rotation)
            else:
                raise ValueError(f"Could not reliably determine type of rotation: {rs}")

        self.rotation = rotation
        self.translation = translation
        self.single = rotation.single

    @staticmethod
    def identity(len=None):
        return RigidTransform(rotation=Rotation.identity(len), translation=np.zeros(shape=(len,3)))

    @staticmethod
    def align_points(a, b, weights=None):
        """
        Aligns two sets of points a and b using the Kabsch algorithm.
        :param a:
        :param b:
        :param weights:
        :return: RigidTransform object representing the rotation and translation that aligns b to a, ie RT(b) ≈ a.
        """
        amean, bmean = np.average(a, weights=weights, axis=0), np.average(b, weights=weights, axis=0)
        rotation, _ = Rotation.align_vectors(a - amean[np.newaxis,...], b - bmean[np.newaxis,...], weights=weights)
        return RigidTransform(rotation=rotation, translation=amean - rotation.apply(bmean))

    @staticmethod
    def from_matrix(matrix):
        np.testing.assert_allclose(matrix[..., 3, 0:4], np.asarray([0, 0, 0, 1]))
        return RigidTransform(rotation=Rotation.from_matrix(matrix[..., 0:3, 0:3]), translation=matrix[..., 0:3, 3])

    def apply(self, vec):
        vec_shape = vec.shape
        if len(vec_shape) > 2:
            vec = vec.reshape((-1, vec_shape[-1]))

        new_vec = self.rotation.apply(vec) + self.translation

        if len(vec_shape) > 2:
            new_vec = new_vec.reshape(vec_shape)
        return new_vec

    def apply_broadcast(self, vec):
        # Return a [shape transforms] x [shape vecs] array of vectors
        new_vec = self.rotate_broadcast(vec)

        #Fill in singular dimensions
        translations = np.expand_dims(self.translation,
                                      [self.translation.ndim - 1 + n for n in range(vec.ndim - 1)])
        new_vec += translations

        return new_vec

    def rotate(self, vec):
        vec_shape = vec.shape
        if len(vec_shape) > 2:
            vec = vec.reshape((-1, vec_shape[-1]))

        new_vec = self.rotation.apply(vec)

        if len(vec_shape) > 2:
            new_vec = new_vec.reshape(vec_shape)
        return new_vec


    def rotate_broadcast(self, vec):
        rot_mats = self.rotation.as_matrix()
        # Somehow was unsuccessful with ellipsis ...
        matrices_subscript = ''.join(chr(ord('a') + i) for i in range(rot_mats.ndim - 2))
        vectors_subscript = ''.join(chr(ord('a') + rot_mats.ndim - 2 + i) for i in range(vec.ndim - 1))
        einsum = f'{matrices_subscript}ij,{vectors_subscript}j->{matrices_subscript}{vectors_subscript}i'
        new_vec = np.einsum(einsum, rot_mats, vec)
        return new_vec

    def apply_on_vector(self, vec):
        vec_shape = vec.shape
        if len(vec_shape) > 2:
            vec = vec.reshape((-1, vec_shape[-1]))

        vec = self.rotation.apply(vec)

        if len(vec_shape) > 2:
            vec = vec.reshape(vec_shape)

        return vec

    def __getitem__(self, key):
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
        if isinstance(other, (Mirror, AffineTransformation)):
            return AffineTransformation(self.as_matrix() @ other.as_matrix())
        raise Exception(f"Type of {other} not supported")

    def as_matrix(self, shape=None):
        if shape is None:
            shape = (4, 4)
        if not self.single and len(shape) == 2:
            shape = (len(self), *shape)

        res = np.zeros(shape=shape, dtype=float)
        res[...,0:3, 0:3] = self.rotation.as_matrix()
        res[...,0:3, 3] = self.translation
        rg = np.arange(3, np.maximum(3, np.min(shape)))
        res[(rg, rg)] = 1
        return res

    def inv(self) -> RigidTransform:
        rotinv = inverse_rot(self.rotation)
        return RigidTransform(rotation=rotinv, translation=-rotinv.apply(self.translation))

    def get_rotation(self):
        return self.rotation

    def get_translation(self):
        return self.translation

    def isnan(self):
        return np.logical_or(isnan_rot(self.rotation), np.any(np.isnan(self.translation), axis=-1))

    @staticmethod
    def concatenate(list):
        return RigidTransform(rotation=Rotation.concatenate([tr.rotation for tr in list]),
                              translation=np.asarray([tr.translation for tr in list]))

    def as_map(self, prefix="", suffix="", flip_quats=False):
        rot = self.rotation.as_quat()
        if flip_quats:
            rot = flip_quaternions(rot)
        return {
            f"{prefix}x{suffix}": self.translation[..., 0],
            f"{prefix}y{suffix}": self.translation[..., 1],
            f"{prefix}z{suffix}": self.translation[..., 2],
            f"{prefix}rx{suffix}": rot[..., 0],
            f"{prefix}ry{suffix}": rot[..., 1],
            f"{prefix}rz{suffix}": rot[..., 2],
            f"{prefix}rw{suffix}": rot[..., 3]}

    @staticmethod
    def from_map(map, suffix=""):
        return RigidTransform(rotation=Rotation.from_quat((map[np.char.add(['rx', 'ry', 'rz', 'rw'], suffix)])),
                              translation=np.asarray((map[np.char.add(['x', 'y', 'z'], suffix)])))

    def interpolate(self, times, fill_boundary="nan", interpolation_method="linear"):
        rotation_interpolation = slerp(times=times, rots=self.rotation, fill_boundary=fill_boundary, interpolation_method=interpolation_method)

        # TODO: interp1d is legacy API. This should be reimplemented, np.interp would be an option
        if fill_boundary == "constant":
            fill_value = self.get_translation()[(0, -1),]
            translation_interpolation = [
                scipy.interpolate.interp1d(times, traj, kind=interpolation_method, bounds_error=False,
                                           fill_value=(fill_value[0,i], fill_value[1,i])) for i, traj in enumerate(np.moveaxis(self.translation, -1, 0))]
        elif fill_boundary == "nan":
            translation_interpolation = [
                scipy.interpolate.interp1d(times, traj, kind=interpolation_method, bounds_error=False,
                                           fill_value=fill_boundary) for traj in np.moveaxis(self.translation, -1, 0)]
        else:
            raise Exception(f"Boundary {fill_boundary} not known")

        return lambda interptimes: RigidTransform(rotation=rotation_interpolation(interptimes), translation=np.stack([ti(interptimes) for ti in translation_interpolation], axis=-1))

    def mean(self, weights):
        return RigidTransform(rotation=mean_rot(self.rotation, weights=weights), translation=np.average(self.translation, axis=0, weights=weights))

    def nanmean(self, keepdims=False):
        valid = np.logical_not(self.isnan())
        return RigidTransform(rotation=self.rotation[valid], translation=self.translation[valid])


def interp_nd(x, xp, fp, **kwargs):
    """
    Multidimensional wrapper for np.interp.


    Parameters
    ----------
    x : (...,) array_like
    The x-coordinates at which to evaluate the interpolated values.
    xp : (M,) array_like
    The x-coordinates of the data points, must be increasing.
    fp : (M, *shape) array_like
    The y-coordinates of the data points, same length as xp,
    followed by arbitrary trailing dimensions.
    **kwargs : dict, optional
    Additional keyword arguments passed to np.interp (e.g., left, right, period).


    Returns
    -------
    out : (..., *shape) ndarray
    Interpolated values, with `x` broadcast to leading dimension(s)
    and trailing dimensions from `fp`.
    """
    x = np.asarray(x)
    xp = np.asarray(xp)
    fp = np.asarray(fp)

    if xp.ndim != 1:
        raise ValueError(f"xp must be 1D and not {xp.ndim}D")
    if fp.shape[0] != xp.shape[0]:
        raise ValueError(f"First dimension of fp {fp.shape[0]} must match xp {xp.shape[0]}")

    orig_shape = fp.shape[1:]
    fp_flat = fp.reshape(fp.shape[0], -1)

    out = np.stack([
        np.interp(x, xp, fp_flat[:, j], **kwargs)
        for j in range(fp_flat.shape[1])
    ], axis=-1)

    return out.reshape(x.shape + orig_shape)


def get_nan_rot(size=None):
    if size == 0:
        return Rotation.identity(0)
    if size is None:
        return Rotation.from_rotvec(np.full(shape=3, fill_value=np.nan))
    array = np.full(fill_value=np.nan, shape=(size, 3)) if size is None else np.full(fill_value=np.nan, shape=(size, 3))
    return Rotation.from_rotvec(array)


@staticmethod
def slerp(times, rots, fill_boundary="nan", interpolation_method="linear", sort=True):
    assert not np.any(np.isnan(times))
    if sort:
        sorted_indices = np.argsort(times)
        times = times[sorted_indices]
        rots = rots[sorted_indices]
    assert np.all(times[:-1] <= times[1:]), f'times must be sorted {np.nonzero(~(times[:-1] <= times[1:]))}'

    if len(times) == 0:
        return lambda etimes: get_nan_rot(len(etimes))
    else:
        match interpolation_method:
            case "nearest":
                mtimes = times
                ctimes = np.convolve(times, (0.5, 0.5), mode="valid") if len(times) != 0 else times

                def find_nearest(etimes):
                    idx = np.searchsorted(ctimes, etimes, side="left")
                    return rots[idx]

                interp = find_nearest
            case "linear":
                mask = isnan_rot(rots, inverted=True)
                mtimes = times[mask]
                interp = Slerp(mtimes, rots[mask])
            case "quadratic":
                mask = isnan_rot(rots, inverted=True)
                mtimes = times[mask]
                interp = RotationSpline(mtimes, rots[mask])
            case _:
                raise Exception(f"Interpolation method {interpolation_method} not known")

    tmin, tmax = (mtimes[0], mtimes[-1]) if len(mtimes) != 0 else (np.inf, -np.inf)
    # Open bins to the left and right of the original time bins should not be considered
    # note that this dumps the last time value when "interpolating" identical,
    # as it belongs to the last bin that is open to the right
    nan_mask = np.zeros(len(rots)+2, dtype=bool)
    nan_mask[1:-1] = isnan_rot(rots)

    def funct(interptimes):
        interptimes = np.copy(interptimes)
        low = interptimes < tmin
        high = interptimes > tmax
        interptimes[low] = tmin
        interptimes[high] = tmax
        res = interp(interptimes)
        match fill_boundary:
            case "nan":
                replace_mask = np.logical_or(low, high)
                res[replace_mask] = Rotation.from_rotvec(np.full(fill_value=np.nan, shape=3))
            case "constant":
                pass
            case _:
                raise Exception(f"Boundary {fill_boundary} not known")
        if interpolation_method != "nearest":
            indices = np.digitize(interptimes, times)
            to_nan_mask = nan_mask[indices] | (nan_mask[indices+1] & (times[indices-1]!=interptimes))
            count = np.count_nonzero(to_nan_mask)
            if count > 0:
                res[to_nan_mask] = get_nan_rot(np.count_nonzero(to_nan_mask))
        return res
    return funct


def intersect(bases, vecs):
    """
    Compute the intersection point of a set of lines in an arbitrary-dimensional space.

    Each line is defined by a base point and a direction vector.
    The function calculates the intersection point by projecting the base points onto
    the null spaces of the direction vectors, then solving the resulting linear system.

    Parameters:
    -----------
    bases : numpy.ndarray
        An array of shape (n, d) where each row represents the base point of a line in d-dimensional space.
    vecs : numpy.ndarray
        An array of shape (n, d) where each row represents the direction vector of a line in d-dimensional space.

    Returns:
    --------
    numpy.ndarray
        A 1D array of shape (d,) representing the intersection point of the lines.
        If the input data is invalid, or if the lines do not intersect uniquely,
        the function returns an array filled with NaN.
    """
    bases = np.asarray(bases)
    vecs = np.asarray(vecs)
    ray_ok = ~np.any(np.isnan(bases) | np.isnan(vecs), axis=-1)
    if len(bases.shape) == 2:
        bases = bases[ray_ok]
        vecs = vecs[ray_ok]
        n = bases.shape[0]
        d = bases.shape[1]
        if n < 2:
            return np.full(d, fill_value=np.nan)

        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # Normalize vecs

        M_sum = np.eye(vecs.shape[1]) * n - np.einsum('ki,kj->ij', vecs, vecs)  # Shape: (d, d)
        # Check rank
        if np.linalg.det(M_sum) < 1e-10:
            return np.full(d, fill_value=np.nan)

        Mbase_sum = np.sum(bases,axis=0) - np.dot(np.einsum('ij,ij->i',vecs, bases),vecs)  # Shape: (d)
        return np.linalg.solve(M_sum, Mbase_sum)
    else:
        bases = np.where(ray_ok[..., None], bases, 0)
        vecs = np.where(ray_ok[..., None], vecs, 0)
        vecs_norm = np.linalg.norm(vecs, axis=-1, keepdims=True)
        vecs = np.divide(vecs, vecs_norm, where=vecs_norm > 0)

        valid_counts = np.count_nonzero(ray_ok, axis=-1, keepdims=True)
        identity = np.eye(vecs.shape[-1])[None, :, :]  # Shape (1, d, d)
        M_sum = valid_counts[:, None] * identity - np.einsum('mij,mik->mjk', vecs, vecs)

        non_singular_mask = ~(np.linalg.det(M_sum) < 1e-10)

        proj = np.einsum('mij,mij->mi', vecs, bases)[..., None] * vecs
        Mbase_sum = np.sum(bases - proj, axis=1)

        intersections = np.full((M_sum.shape[0], vecs.shape[-1]), np.nan)
        intersections[non_singular_mask] = np.linalg.solve(M_sum[non_singular_mask], Mbase_sum[non_singular_mask])
        return intersections


def point_line_distance(bases, vecs, points, outer=False):
    points = np.asarray(points, dtype=np.float64)
    if bases.ndim == 1 and points.ndim > 1:
        bases = bases[np.newaxis, :]
        vecs = vecs[np.newaxis, :]
    if outer:
        bases = bases[:, np.newaxis, :]
        vecs = vecs[:, np.newaxis, :]
    d = points - bases
    proj_vecs = d - np.sum(d * vecs, axis=-1, keepdims=True) * vecs
    dists = np.linalg.norm(proj_vecs, axis=-1)
    return dists

class Line:
    def __init__(self, position=None, direction=None, lines=None, dtype=np.float64):
        if lines is not None:
            assert position is None
            assert direction is None
            position = [l.position for l in lines]
            direction = [l.direction for l in lines]
        self.position = np.asarray(position, dtype=dtype)
        self.direction = np.asarray(direction, dtype=dtype)
        if not np.array_equal(self.position.shape, self.direction.shape):
            raise Exception("Shapes have to be equal")
        self.shape = self.position.shape
        self.dtype = dtype

    def transform_internal_array(self, functional):
        self.position = functional(self.position)
        self.direction = functional(self.direction)

    def intersect(self):
        return intersect(self.position, self.direction)

    def get_endpoints(self):
        return np.stack((self.position, self.position + self.direction), axis=-2)

    def __repr__(self):
        return f"{self.position} + t * {self.direction}"

    def __str__(self):
        return f"{self.position} + t * {self.direction}"

    def __getitem__(self, key):
        return Line(position=self.position[key], direction=self.direction[key])

    def __setitem__(self, index, value):
        self.position[index] = value.position
        self.direction[index] = value.direction

    def get_projection_t(self, position):
        position = np.asarray(position) - self.position
        return np.sum(position * self.direction, axis=-1) / np.sum(self.direction * self.direction, axis=-1)

    def project(self, position):
        t = self.get_projection_t(position)
        return self.position + t[..., np.newaxis] * self.direction

    def is_single(self):
        return self.position.ndim == 1

    def __len__(self):
        #Show same behavior as numpy arrays
        if self.is_single():
            raise Exception('len() of unsized object')
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
        return point_line_distance(self.position, self.direction / np.linalg.norm(self.direction, axis=-1, keepdims=True), x, outer=outer)

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


def get_homogenuous_transformation_from_mirror(normal, translation):
    dim = len(normal)
    res = np.zeros((dim + 1, dim + 1), dtype=float)
    res[0:3, 0:3] = get_mirror_matrix(normal)
    res[:3, 3] = 2 * np.asarray(normal) * translation
    res[3, 3] = 1
    return res


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
    source = np.asarray(source)
    dest = np.asarray(dest)
    if normalize:
        source = source / np.linalg.norm(source, axis=-1, keepdims=True)
        dest = dest / np.linalg.norm(dest, axis=-1, keepdims=True)
    rotvec = np.cross(source, dest)
    norm = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    dot = np.sum(source * dest, axis=-1, keepdims=True)
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
        return points @ np.swapaxes(self.mat, -1, -2)

    def as_matrix(self):
        return self.mat

    def __mul__(self, other):
        if isinstance(other, AffineTransformation):
            return AffineTransformation(self) * other
        return LinearTransformation(self.mat @ other.mat)


class AffineTransformation:
    def __init__(self, mat):
        if isinstance(mat, Rotation):
            self.mat = np.eye(4, dtype=float)
            self.mat[0:3, 0:3] = mat.as_matrix()
        elif isinstance(mat, tuple) and len(mat) == 2:
            self.mat = np.eye(4, dtype=float)
            if isinstance(mat[0], Rotation):
                self.mat[0:3, 0:3] = mat[0].as_matrix()
            elif mat[0] is not None:
                self.mat[0:3, 0:3] = mat[0]
            if mat[1] is not None:
                 self.mat[0:3, 3] = mat[1]
        elif isinstance(mat, LinearTransformation):
            self.mat = np.eye(4, dtype=float)
            self.mat[0:3, 0:3] = mat.mat
        elif isinstance(mat, AffineTransformation):
            self.mat = mat.mat
        elif isinstance(mat, (RigidTransform, Reflection, Mirror)):
            self.mat = mat.as_matrix(shape=(4, 4))
        elif isinstance(mat, np.ndarray):
            if mat.shape[-1] == 4 and mat.shape[-2] == 4:
                self.mat = np.copy(mat)
            else:
                self.mat = np.zeros_like(mat, shape=(*mat.shape[0:-2], 4, 4))
                self.mat[...,0:mat.shape[-2], 0:mat.shape[-1]] = mat
        else:
            raise Exception(f'Wrong matrix type {type(mat)}')
        self.mat[...,3, 0:3] = 0
        self.mat[...,3, 3] = 1

    @staticmethod
    def from_matrix(mat):
        return AffineTransformation(mat)

    @staticmethod
    def identity():
        return AffineTransformation(np.eye(4, dtype=float))

    def to_dict(self):
        return {
            "mat": self.mat,
            "type": "AffineTransformation"
        }

    @staticmethod
    def from_dict(data):
        if data["type"] != "AffineTransformation":
            raise ValueError("Invalid type for AffineTransformation")
        return AffineTransformation(data["mat"])

    def get_scaling(self, keepdims=False):
        return np.linalg.norm(self.mat[...,0:3, 0:3], axis=-2, keepdims=keepdims)

    def get_translation(self):
        return self.mat[..., 0:3, 3]

    def inv(self):
        return AffineTransformation(np.linalg.inv(self.mat))

    def apply(self, points):
        return (points @ np.swapaxes(self.mat[...,0:3, 0:3],-1,-2)) + self.mat[...,0:3, 3]

    def apply_on_vector(self, vectors):
        return vectors @ np.swapaxes(self.mat[...,0:3, 0:3],-1,-2)

    def linear(self):
        return LinearTransformation(self.mat[0:3, 0:3])

    def as_matrix(self, shape=None):
        return self.mat if shape is None else self.mat[..., 0:shape[0], 0:shape[1]]

    def __getitem__(self, key):
        return AffineTransformation(self.mat[key])

    def __setitem__(self, index, value):
        if isinstance(value, AffineTransformation):
            self.mat[index] = value.mat
        elif isinstance(value, np.ndarray) and value.shape == (4, 4):
            self.mat[index] = value
        else:
            raise ValueError(f"Cannot set item with type {type(value)}")

    def __len__(self):
        if self.mat.ndim < 2:
            raise Exception('Not an array')
        return len(self.mat)

    def __mul__(self, other):
        if not isinstance(other, AffineTransformation):
            other = AffineTransformation(other)
        return AffineTransformation(self.mat @ other.mat)

    def __repr__(self):
        return f"AffineTransformation({self.mat})"

    def __str__(self):
        return f"AffineTransformation({self.mat})"


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

    def as_matrix(self, shape=None):
        res = np.eye(len(self.normal)) - 2 * np.outer(self.normal, self.normal)
        if shape is not None:
            tmp = np.zeros(shape=shape, dtype=float)
            tmp[0:3, 0:3] = res
            res = tmp
            rg = np.arange(3, np.maximum(3, np.min(shape)))
            res[(rg, rg)] = 1
        return res

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
            self.point_on_mirror:np.ndarray = point_on_mirror
            self.tr:float = np.dot(self.normal, self.point_on_mirror)

        if tr is not None:
            self.tr:float = tr
            self.point_on_mirror:np.ndarray = self.normal * self.tr

        self.M = get_mirror_matrix(self.normal)
        self.HomTr = get_homogenuous_transformation_from_mirror(self.normal, self.tr)

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

    def as_matrix(self, shape=None):
        return np.copy(self.HomTr)

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

    def transform(self, rot_traf, tr=None):
        if tr is None:
            tr = np.zeros((1, 3))
        if isinstance(rot_traf, Rotation):
            rot_traf = RigidTransform(rotation=rot_traf, translation=tr)
        return Mirror(normal=rot_traf.apply_on_vector(self.normal),
                      point_on_mirror=rot_traf.apply(self.point_on_mirror).reshape((3,)))

    def set_tr(self, tr):
        self.__init__(self.normal, tr=tr)

    def apply_on_point(self, points):
        res = np.dot(self.M, points.T).T + (self.normal * (self.tr * 2))[np.newaxis, :]
        return res

    def apply(self, points):
        mirrored = np.dot(self.M, points.T).T
        translation = (self.normal * (self.tr * 2))
        return mirrored + translation[tuple([slice(np.newaxis)] * (mirrored.ndim - translation.ndim))]

    def apply_on_vector(self, vectors):
        return np.dot(self.M, vectors.T).T

    def __len__(self):
        if len(self.normal.shape) == 1:
            raise Exception('Not an array')
        return self.normal.shape[0]

    def apply_on_line(self, line: Line|np.ndarray):
        if isinstance(line, Line):
            return Line(direction=self.apply_on_vector(line.direction), position=self.apply_on_point(line.position))
        return self.apply_on_vector(line[0]), self.apply_on_point(line[1])

    def copy(self):
        return Mirror(normal=np.copy(self.normal), tr=np.copy(self.tr))

    def line_distance(self, line: Line):
        return (self.tr - np.dot(self.normal, line.position)) / np.dot(self.normal, line.direction.T)

    def distance_to_mirror_plane(self, vectors):
        return vectors @ self.normal - self.tr

    def find_intersection(self, directions, translation):
        directions = np.asarray(directions)
        translation = np.asarray(translation)
        ndim = max(directions.ndim, translation.ndim)
        if directions.ndim < ndim:
            directions = np.expand_dims(directions, axis=tuple(np.arange(ndim - directions.ndim)))
        elif translation.ndim < ndim:
            translation = np.expand_dims(translation, axis=tuple(np.arange(ndim - translation.ndim)))
        scalar = ((translation @ self.normal - self.tr) / (directions @ self.normal))
        return translation - directions * scalar[..., np.newaxis]

    #    def find_intersection(self, line:Line):
    #        return self.find_intersection(line.directions, line.translations)

    def get_distmat(self, lines):
        mirror_3d_vertices = self.find_intersection(lines[0], lines[1])
        return np.linalg.norm(mirror_3d_vertices[np.newaxis, :, :] - mirror_3d_vertices[:, np.newaxis, :], axis=-1)

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
