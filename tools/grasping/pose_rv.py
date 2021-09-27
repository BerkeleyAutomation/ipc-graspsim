import numpy as np
import scipy.stats

from autolab_core import transformations, RigidTransform
from autolab_core.utils import sph2cart

import abc
import six

@six.add_metaclass(abc.ABCMeta)
class RV(object):
    """Abstract base class for all random variables, which are classes that
    produce some object or primitive type (usually via sampling).
    Every factory has a method, `generate()`, which is responsible
    for providing a concrete object or primitive type to the caller.
    """

    @abc.abstractmethod
    def generate(self, mean=False):
        pass

    def mean(self):
        return self.generate(mean=True)


class PoseRV(RV):
    """Base class for all random variables that produce a pose.
    """
    pass

class GaussianPoseRV(PoseRV):
    """A normally-distributed pose random variable.

    This samples pose perturbations about the origin.

    Attributes
    ----------
    sigma_trans : float
        Standard deviation of the translation.
    sigma_rot : float
        Standard deviation of the rotation.
    size : int
        The number of instances of the RV to generate on each call.
    from_frame : str
        The frame of reference the RV's transforms are from.
    to_frame : str
        The frame of reference the RV's transforms are to.
    """

    def __init__(self, sigma_trans, sigma_rot, size=1, from_frame='world', to_frame='world'):
        self.sigma_trans = sigma_trans
        self.sigma_rot = sigma_rot
        self.size = size
        self.from_frame = from_frame
        self.to_frame = to_frame

        if sigma_trans == 0:
            self._ret_t = np.zeros(3)
        else:
            self._t_rv = scipy.stats.multivariate_normal(np.zeros(3), sigma_trans**2 * np.eye(3))

        if sigma_rot == 0:
            self._ret_R = np.eye(3)
        else:
            self._xi_rv = scipy.stats.multivariate_normal(np.zeros(3), sigma_rot **2 * np.eye(3))

    def generate(self, mean=False):
        """Sample a Gaussian Pose RV.

        Returns
        -------
        sample : list of RigidTransform
            The random variable's sampled value.
        """
        samples = []
        for i in range(self.size):
            if mean or self.sigma_trans == 0:
                t_sample = self._ret_t
            else:
                t_sample = self._t_rv.rvs(size=1)

            if mean or self.sigma_rot == 0:
                R_sample = self._ret_R
            else:
                xi = self._xi_rv.rvs(size=1)
                S_xi = np.array([[0, -xi[2], xi[1]],
                                [xi[2], 0, -xi[0]],
                                [-xi[1], xi[0], 0]])
                R_sample = scipy.linalg.expm(S_xi)
            samples.append(RigidTransform(rotation=R_sample,
                                          translation=t_sample,
                                          from_frame=self.from_frame,
                                          to_frame=self.to_frame))
        if self.size == 1:
            return samples[0]
        return samples

class UniformPoseRV(PoseRV):
    """A uniformly-distributed pose random variable.

    This samples rotations uniformly at random and translations
    uniformly in an axis-aligned box.

    Attributes
    ----------
    bounds : (2,3) float
        Lower and upper axis-aligned bounds for the translation.
    size : int
        The number of instances of the RV to generate on each call.
    from_frame : str
        The frame of reference the RV's transforms are from.
    to_frame : str
        The frame of reference the RV's transforms are to.
    """

    def __init__(self, bounds=np.zeros((2,3)), size=1, from_frame='world', to_frame='world'):
        self.bounds = bounds
        self.size = size
        self.from_frame = from_frame
        self.to_frame = to_frame

    def generate(self, mean=False):
        """Sample a Uniform Pose RV.

        Returns
        -------
        sample : list of RigidTransform
            The random variable's sampled value.
        """
        samples = []
        for i in range(self.size):
            if mean:
                t_sample = np.mean(self.bounds, axis=0)
                rot = 0.0
            else:
                t_sample = np.random.uniform(self.bounds[0], self.bounds[1])
                rot = np.random.uniform(0, 2*np.pi)
            direc = np.random.normal(size=3)
            direc = direc / np.linalg.norm(direc)
            R_sample = transformations.rotation_matrix(rot, direc)[:3,:3]
            samples.append(RigidTransform(rotation=R_sample,
                                          translation=t_sample,
                                          from_frame=self.from_frame,
                                          to_frame=self.to_frame))
        if self.size == 1:
            return samples[0]
        return samples

class ViewsphereUniformPoseRV(PoseRV):
    """A uniformly-distributed pose random variable in a slice of a viewing
    sphere.

    The poses here are camera poses, where the z axis faces towards the scene,
    the x axis faces right, and the y axis faces down.

    Attributes
    ----------
    rad_bounds : (2,) float
        Lower and upper bounds for radius of viewsphere, in meters.
    elev_bounds : (2,) float
        Lower and upper bounds for elevation angle, in degrees.
    az_bounds : (2,) float
        Lower and upper bounds for azimuth angle, in degrees.
    roll_bounds : (2,) float
        Lower and upper bounds for roll angle, in degrees.
    center_bounds: (2,3) float
        Lower and upper bounds on the center of the viewsphere, in meters.
    size : int
        The number of instances of the RV to generate on each call.
    from_frame : str
        The frame of reference the RV's transforms are from.
    to_frame : str
        The frame of reference the RV's transforms are to.
    """

    def __init__(self, rad_bounds, elev_bounds, az_bounds,
                 roll_bounds, center_bounds, size=1,
                 from_frame='world', to_frame='world'):
        self.rad_bounds = rad_bounds
        self.elev_bounds = elev_bounds
        self.az_bounds = az_bounds
        self.roll_bounds = roll_bounds
        self.center_bounds = center_bounds
        self.size = size
        self.from_frame = from_frame
        self.to_frame = to_frame

    def generate(self, mean=False):
        """Sample a Viewsphere Uniform Pose RV.

        Returns
        -------
        sample : list of RigidTransform
            The random variable's sampled value.
        """
        samples = []
        for i in range(self.size):

            # Sample parameters
            if mean:
                rad = np.mean(self.rad_bounds)
                elev = np.deg2rad(np.mean(self.elev_bounds))
                az = np.deg2rad(np.mean(self.elev_bounds))
                roll = np.deg2rad(np.mean(self.roll_bounds))
                center = np.mean(self.center_bounds, axis=0)
            else:
                rad = np.random.uniform(self.rad_bounds[0], self.rad_bounds[1])
                elev = np.deg2rad(np.random.uniform(self.elev_bounds[0], self.elev_bounds[1]))
                az = np.deg2rad(np.random.uniform(self.az_bounds[0], self.az_bounds[1]))
                roll = np.deg2rad(np.random.uniform(self.roll_bounds[0], self.roll_bounds[1]))
                center = np.random.uniform(self.center_bounds[0], self.center_bounds[1])

            # Generate pose
            view_z = np.array([sph2cart(rad, az, elev)]).squeeze()
            view_c = view_z + center
            view_z = -view_z / np.linalg.norm(view_z)

            # Try to line up camera X with axis point
            if view_z[2] > 0:
                view_x = np.array([view_z[2], 0, -view_z[0]])
            elif view_z[2] < 0:
                view_x = np.array([-view_z[2], 0, view_z[0]])
            else:
                view_x = np.array([1.0, 0.0, 0.0])

            view_x_norm = np.linalg.norm(view_x)
            view_x /= view_x_norm
            view_y = np.cross(view_z, view_x)
            view_y /= np.linalg.norm(view_y)

            # Reverse the x direction if needed so that y points down
            #if view_y[2] > 0:
            #    view_x = -view_x
            #    view_y = np.cross(view_z, view_x)
            #    view_y /= np.linalg.norm(view_y)

            # Rotate by the roll
            R = np.vstack((view_x, view_y, view_z)).T
            roll_rot_mat = transformations.rotation_matrix(roll, view_z, np.zeros(3))[:3,:3]
            R = roll_rot_mat.dot(R)

            # Create TF
            samples.append(RigidTransform(R, view_c, from_frame=self.from_frame, to_frame=self.to_frame))

        if self.size == 1:
            return samples[0]
        return samples
