#!/usr/bin/env python3.7

"""
The copyrights of this software are owned by Duke University.
Please refer to the LICENSE.txt and README.txt files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
"""

import random
import warnings
from typing import List, Tuple

from shapely.affinity import scale
from shapely.geometry import LineString, Point
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats

# ascent
from src.utils import *
from .sample import Sample


class FiberSet(Exceptionable, Configurable, Saveable):
    """
    Required (Config.) JSON's:
        MODEL
        SIM
    """

    def __init__(self, sample: Sample, exceptions_config: list):
        """
        :param exceptions_config: preloaded exceptions.json data
        """

        # set up superclasses
        Configurable.__init__(self)
        Exceptionable.__init__(self, SetupMode.OLD, exceptions_config)

        # initialize empty lists of fiber points
        self.sample = sample
        self.fibers = None
        self.out_to_fib = None
        self.out_to_in = None
        self.add(SetupMode.NEW, Config.FIBER_Z, os.path.join('config', 'system', 'fiber_z.json'))

    def init_post_config(self):
        if any([config.value not in self.configs.keys() for config in (Config.MODEL, Config.SIM)]):
            self.throw(78)
        return self

    def generate(self, sim_directory: str, super_sample: bool = False):
        """
        :return:
        """

        xy_mode_name: str = self.search(Config.SIM, 'fibers', 'xy_parameters', 'mode')
        xy_mode: FiberXYMode = [mode for mode in FiberXYMode if str(mode).split('.')[-1] == xy_mode_name][0]

        if not xy_mode == FiberXYMode.SL_PSEUDO_INTERP:
            fibers_xy = self._generate_xy(sim_directory)
            self.out_to_fib, self.out_to_in = self._generate_maps(fibers_xy)
            self.fibers = self._generate_z(fibers_xy, super_sample=super_sample)

        else:
            # SL generation algorithm
            sample_position = self.sample.configs[Config.SAMPLE.value].get('position', None)
            if sample_position is not None:
                pass
                # print('\t\tUsing {} µm positioning for SL curve'.format(sample_position))
            else:
                sample_position = 5000  # default
                # print('\t\tNo positioning for SL curve found. Using {} µm.'.format(sample_position))

            z_nerve = self.search(Config.MODEL, 'medium', 'proximal', 'length')
            z_medium = self.search(Config.MODEL, 'medium', 'distal', 'length')
            # NOTE: for now, the sample position will be interpreted as the z-position of the SL branch
            z_offset = sample_position  # + z_nerve / 2  # sample_position is distance from center of cuff to SL branch
            r_medium = self.search(Config.MODEL, 'medium', 'distal', 'radius')
            buffer = 50  # minimum distance from top of distal model

            if z_offset >= z_medium - 1000:
                print('\t\tWARNING: SL z_offset ({}) within 1000 µm of distal model length ({})'.format(z_offset,
                                                                                                        z_medium))

            def fit_z(t):
                return (10 ** 5 / t) + z_offset

            def fit_3d(t, theta, function):
                return t * np.cos(theta), t * np.sin(theta), function(t)

            def magnitude(vec):
                return np.sqrt(sum(item ** 2 for item in vec))

            # generate parameter range
            t_min = 0.001
            t_max = r_medium - buffer
            t_step = 1
            t_range = np.arange(t_min, t_max, t_step)

            while fit_z(t_range[0]) > z_medium - 50:
                # print('\t\tclip')
                t_range = t_range[1:]

            # init theta
            theta = self.search(Config.SIM, 'theta') if 'theta' in self.configs[Config.SIM.value].keys() else 0

            # set angle theta to orientation point if defined
            slide = self.sample.slides[0]
            # if slide.orientation_point_index is not None:
            #     outer = slide.fascicles[0] if slide.monofasc() else slide.nerve
            #     orientation_x, orientation_y = tuple(outer.points[slide.orientation_point_index][:2])
            #     theta = np.arctan2(orientation_y, orientation_x)

            x, y, z = fit_3d(t_range, theta, fit_z)
            points = list(zip(x, y, z))
            fiber_length = sum(
                magnitude(np.asarray(points[i]) - np.asarray(points[i + 1])) for i in range(len(points) - 1))

            fibers_xy = np.asarray([(0, 0)])
            fibers = self._generate_z(fibers_xy, override_length=fiber_length, super_sample=super_sample)

            fiber_z_points = np.asarray(fibers[0])[:, 2]
            fiber_z_points = fiber_z_points - np.min(fiber_z_points)
            ratios = fiber_z_points / np.max(fiber_z_points)

            self.fibers = [interparc(ratios, x, y, z)]

        return self

    def write(self, mode: WriteMode, path: str):
        """
        :param mode:
        :param path:
        :return:
        """

        diams = []
        for i, fiber_pre in enumerate(self.fibers if self.fibers is not None else []):

            if not isinstance(fiber_pre, dict):
                fiber = fiber_pre
            else:
                fiber = fiber_pre['fiber']
                diam = fiber_pre['diam']
                diams.append(diam)

            with open(os.path.join(path, str(i) + WriteMode.file_endings.value[mode.value]), 'w') as f:
                for row in [len(fiber)] + list(fiber):
                    if not isinstance(row, int):
                        for el in row:
                            f.write(str(el) + ' ')
                    else:
                        f.write(str(row) + ' ')
                    f.write("\n")

        if len(diams) > 0:
            diams_key_path = os.path.join(path, 'diams.txt')
            with open(diams_key_path, "w") as f2:
                np.savetxt(f2, diams, fmt='%0.1f')

        return self

    def _generate_maps(self, fibers_xy) -> Tuple[List, List]:
        out_to_fib = []
        out_to_in = []

        inner_ind = 0
        for i, fascicle in enumerate(self.sample.slides[0].fascicles):
            out_to_in.append([])
            out_to_fib.append([])
            for j, inner in enumerate(fascicle.inners):
                out_to_in[i].append(inner_ind)
                out_to_fib[i].append([])
                inner_ind += 1
                for q, fiber in enumerate(fibers_xy):
                    if Point(fiber).within(inner.polygon()):
                        out_to_fib[i][j].append(q)

        return out_to_fib, out_to_in

    def _generate_xy(self, sim_directory: str) -> np.ndarray:
        # get required parameters from configuration JSON (using inherited Configurable methods)
        xy_mode_name: str = self.search(Config.SIM, 'fibers', 'xy_parameters', 'mode')
        xy_mode: FiberXYMode = [mode for mode in FiberXYMode if str(mode).split('.')[-1] == xy_mode_name][0]
        xy_parameters: dict = self.search(Config.SIM, 'fibers', 'xy_parameters')
        my_xy_seed: int = xy_parameters.get('seed', 0)

        # initialize result lists
        points: List[Tuple[float]] = []

        # small behavioral parameters
        buffer: float = self.search(Config.SIM, 'fibers', 'xy_trace_buffer')
        plot: bool = self.search(Config.SIM, 'fibers', 'plot')

        # perform implemented mode
        if self.search_mode(FiberZMode, Config.MODEL) == FiberZMode.EXTRUSION:

            if xy_mode == FiberXYMode.CENTROID:
                for fascicle in self.sample.slides[0].fascicles:
                    for inner in fascicle.inners:
                        for _ in (0,):
                            points.append(inner.centroid())

            elif xy_mode == FiberXYMode.UNIFORM_DENSITY:
                # DENSITY UNIT: axons / um^2

                # this determines whether the density should be determined top-down or bottom-up
                # case top_down == true: fetch target density and cap minimum axons if too low
                # case top_down == false: (i.e. bottom-up) find density from target number and smallest inner by area
                #   also cap the number at a maximum!
                top_down: bool = self.search(Config.SIM, 'fibers', 'xy_parameters', 'top_down')

                if top_down:  # do top-down approach
                    # get required parameters
                    target_density = self.search(Config.SIM, 'fibers', 'xy_parameters', 'target_density')
                    minimum_number = self.search(Config.SIM, 'fibers', 'xy_parameters', 'minimum_number')

                    for fascicle in self.sample.slides[0].fascicles:
                        for inner in fascicle.inners:
                            fiber_count = target_density * inner.area()
                            if fiber_count < minimum_number:
                                fiber_count = minimum_number
                            for point in inner.random_points(fiber_count, buffer=buffer, my_xy_seed=my_xy_seed):
                                points.append(point)
                            my_xy_seed += 1

                else:  # do bottom-up approach
                    # get required parameters
                    target_number = self.search(Config.SIM, 'fibers', 'xy_parameters', 'target_number')
                    maximum_number = self.search(Config.SIM, 'fibers', 'xy_parameters', 'maximum_number')

                    # calculate target density
                    min_area = np.amin([[fascicle.smallest_trace().area()
                                         for fascicle in self.sample.slides[0].fascicles]])
                    target_density = float(target_number) / min_area

                    for fascicle in self.sample.slides[0].fascicles:
                        for inner in fascicle.inners:
                            fiber_count = target_density * inner.area()
                            if fiber_count > maximum_number:
                                fiber_count = maximum_number
                            for point in inner.random_points(fiber_count, buffer=buffer, my_xy_seed=my_xy_seed):
                                points.append(point)
                            my_xy_seed += 1

            elif xy_mode == FiberXYMode.UNIFORM_COUNT:
                count: int = self.search(Config.SIM, 'fibers', 'xy_parameters', 'count')

                for fascicle in self.sample.slides[0].fascicles:
                    for inner in fascicle.inners:
                        for point in inner.random_points(count, buffer=buffer, my_xy_seed=my_xy_seed):
                            points.append(point)
                        my_xy_seed += 1

            elif xy_mode == FiberXYMode.WHEEL:
                # get required parameters
                spoke_count: int = self.search(Config.SIM, 'fibers', 'xy_parameters', 'spoke_count')
                point_count: int = self.search(Config.SIM, 'fibers', 'xy_parameters',
                                               'point_count_per_spoke')  # this number is PER SPOKE
                find_centroid: bool = self.search(Config.SIM, 'fibers', 'xy_parameters', 'find_centroid')
                angle_offset_is_in_degrees: bool = self.search(Config.SIM, 'fibers',
                                                               'xy_parameters',
                                                               'angle_offset_is_in_degrees')
                angle_offset: float = self.search(Config.SIM, 'fibers', 'xy_parameters', 'angle_offset')

                # convert angle offset to radians if necessary
                if angle_offset_is_in_degrees:
                    angle_offset *= 2 * np.pi / 360

                # master loop!
                for fascicle in self.sample.slides[0].fascicles:
                    for inner in fascicle.inners:
                        if find_centroid:
                            points.append(inner.centroid())

                        # loop through spoke angles
                        for spoke_angle in (np.linspace(0, 2 * np.pi, spoke_count + 1)[:-1] + angle_offset):
                            # find the mean radius for a reference distance when "casting the spoke ray"
                            new_inner = inner.deepcopy()
                            new_inner.offset(None, -buffer)

                            mean_radius = new_inner.mean_radius()

                            # get a point that is assumed to be outside the trace
                            raw_outer_point = np.array(new_inner.centroid()) + [5 * mean_radius * np.cos(spoke_angle),
                                                                                5 * mean_radius * np.sin(spoke_angle)]

                            # build a vector starting from the centroid of the trace
                            raw_spoke_vector = LineString([new_inner.centroid(),
                                                           tuple(raw_outer_point)])

                            # get that vector's intersection with the trace to find "trimmed" endpoint
                            intersection_with_boundary = raw_spoke_vector.intersection(new_inner.polygon().boundary)

                            # fix type of intersection with boundary
                            if not isinstance(intersection_with_boundary, Point):
                                intersection_with_boundary = list(intersection_with_boundary)[0]

                            # build trimmed vector
                            trimmed_spoke_vector = LineString([new_inner.centroid(),
                                                               tuple(intersection_with_boundary.coords)[0]])

                            # get scale vectors whose endpoints will be the desired points ([1:] to not include 0)
                            scaled_vectors: List[LineString] = [scale(trimmed_spoke_vector, *([factor] * 3),
                                                                      origin=trimmed_spoke_vector.coords[0])
                                                                for factor in np.linspace(0, 1, point_count + 2)[1:-1]]

                            # loop through the end points of the vectors
                            for point in [vector.coords[1] for vector in scaled_vectors]:
                                points.append(point)

            elif xy_mode == FiberXYMode.EXPLICIT:

                if not os.path.exists(os.path.join(sim_directory, 'explicit.txt')):
                    self.throw(83)

                with open(os.path.join(sim_directory, 'explicit.txt')) as f:
                    # advance header
                    next(f)
                    reader = csv.reader(f, delimiter=" ")
                    for row in reader:
                        points.append(tuple([float(row[0]), float(row[1])]))

                # check that all fibers are within exactly one inner
                for fiber in points:
                    if not any([Point(fiber).within(inner.polygon())
                                for fascicle in self.sample.slides[0].fascicles for inner in fascicle.inners]):
                        print("Explicit fiber coordinate: {} does not fall in an inner".format(fiber))
                        self.throw(71)

            if plot:
                plt.figure()
                self.sample.slides[0].plot(final=False, fix_aspect_ratio=True)
                for point in points:
                    plt.plot(point[0], point[1], 'r*')
                plt.show()
        else:
            self.throw(30)

        return points

    def plot(self, ax: plt.Axes = None,
             fiber_colors: List[Tuple[float, float, float, float]] = None,
             size=10):

        for fiber_ind, fiber in enumerate(self.fibers):
            ax.plot(fiber[0][0], fiber[0][1], color=fiber_colors[fiber_ind], marker='o', markersize=size)

    def _generate_z(self, fibers_xy: np.ndarray, override_length=None, super_sample: bool = False) -> np.ndarray:

        fibers = []

        def clip(values: list, start, end, myel: bool, is_points: bool = False) -> list:

            step = 1
            if myel:
                step = 11

            while (start - (values[0] if not is_points else values[0][-1]) > -0.1) or \
                    ((values[-1] if not is_points else values[-1][-1]) - end > -0.1):
                values = values[step:-step]

            return values

        def generate_myel_fiber_zs(my_z_shift_to_center, diameter):
            delta_z = \
                paranodal_length_2 = \
                inter_length = None

            sampling_mode = self.search(Config.FIBER_Z,
                                        MyelinationMode.parameters.value,
                                        fiber_geometry_mode_name,
                                        "sampling")

            node_length, paranodal_length_1, inter_length_str = (
                self.search(Config.FIBER_Z, MyelinationMode.parameters.value, fiber_geometry_mode_name, key)
                for key in ('node_length', 'paranodal_length_1', 'inter_length')
            )

            # load in all the required specifications for finding myelinated z coordinates
            if sampling_mode == MyelinatedSamplingType.DISCRETE.value:

                diameters, my_delta_zs, paranodal_length_2s = (
                    self.search(Config.FIBER_Z, MyelinationMode.parameters.value, fiber_geometry_mode_name, key)
                    for key in ('diameters', 'delta_zs', 'paranodal_length_2s')
                )

                diameter_index = diameters.index(diameter)
                delta_z = my_delta_zs[diameter_index]
                paranodal_length_2 = paranodal_length_2s[diameter_index]
                inter_length = eval(inter_length_str)

            elif sampling_mode == MyelinatedSamplingType.INTERPOLATION.value:

                paranodal_length_2_str, delta_z_str, inter_length_str = (
                    self.search(Config.FIBER_Z, MyelinationMode.parameters.value, fiber_geometry_mode_name, key)
                    for key in ('paranodal_length_2', 'delta_z', 'inter_length')
                )
                paranodal_length_2 = eval(paranodal_length_2_str)

                if fiber_geometry_mode_name == FiberGeometry.B_FIBER.value:
                    inter_length = eval(inter_length_str)
                    delta_z = eval(delta_z_str)
                elif fiber_geometry_mode_name == FiberGeometry.MRG_INTERPOLATION.value:
                    if diameter > 16.0 or diameter < 2.0:
                        self.throw(77)
                    if diameter >= 5.643:
                        delta_z = eval(delta_z_str["diameter_greater_or_equal_5.643um"])
                    else:
                        delta_z = eval(delta_z_str["diameter_less_5.643um"])
                    inter_length = eval(inter_length_str)

            z_steps: List = []
            while (sum(z_steps) - half_fiber_length) < 0.001:
                z_steps += [(node_length / 2) + (paranodal_length_1 / 2),
                            (paranodal_length_1 / 2) + (paranodal_length_2 / 2),
                            (paranodal_length_2 / 2) + (inter_length / 2),
                            *([inter_length] * 5),
                            (inter_length / 2) + (paranodal_length_2 / 2),
                            (paranodal_length_2 / 2) + (paranodal_length_1 / 2),
                            (paranodal_length_1 / 2) + (node_length / 2)]

            # account for difference between last node z and half fiber length -> must shift extra distance
            my_z_shift_to_center += sum(z_steps) - half_fiber_length

            reverse_z_steps = z_steps.copy()
            reverse_z_steps.reverse()

            # concat, cumsum, and other stuff to get final list of z points
            my_zs = np.array(
                list(
                    np.cumsum(
                        np.concatenate(
                            ([0], reverse_z_steps, z_steps)
                        )
                    )
                ),
            )

            return my_zs, delta_z, my_z_shift_to_center

        def build_fiber_with_offset(z_values: list, myel: bool, length: float, dz: float, my_x: float, my_y: float,
                                    additional_offset: float = 0):

            z_values = clip(z_values,
                            self.search(Config.SIM, 'fibers', FiberZMode.parameters.value, 'min'),
                            self.search(Config.SIM, 'fibers', FiberZMode.parameters.value, 'max'),
                            myel)

            # get offset param - NOTE: raw value is a FRACTION of dz (explanation for multiplication by dz)
            if 'offset' in self.search(Config.SIM, 'fibers', FiberZMode.parameters.value).keys():
                offset = self.search(Config.SIM, 'fibers', FiberZMode.parameters.value, 'offset') * dz

                if 0 <= offset <= 1:
                    pass
                else:
                    self.throw(99)

            else:
                offset = None

            random_offset = False
            if offset is None:
                offset = 0
                random_offset = True

            random_offset_value = dz * (random.random() - 0.5) if random_offset else 0

            # compute offset z coordinate
            z_offset = [z + offset - additional_offset + random_offset_value for z in z_values]

            xy_mode_name: str = self.search(Config.SIM, 'fibers', 'xy_parameters', 'mode')
            xy_mode: FiberXYMode = [mode for mode in FiberXYMode if str(mode).split('.')[-1] == xy_mode_name][0]

            # only clip if NOT an SL fiber
            if xy_mode != FiberXYMode.SL_PSEUDO_INTERP:
                z_offset = clip(z_offset,
                                self.search(Config.SIM, 'fibers', FiberZMode.parameters.value, 'min'),
                                self.search(Config.SIM, 'fibers', FiberZMode.parameters.value, 'max'),
                                myel)

            fiber = [(x, y, z) for z in z_offset]

            return fiber

        # %% START ALGORITHM

        # get top-level fiber z generation
        fiber_z_mode: FiberZMode = self.search_mode(FiberZMode, Config.MODEL)

        # all functionality is only defined for EXTRUSION as of now
        if fiber_z_mode == FiberZMode.EXTRUSION:

            model_length = self.search(Config.MODEL, 'medium', 'proximal', 'length') if (
                    override_length is None) else override_length

            if not ('min' in self.configs['sims']['fibers']['z_parameters'].keys() and 'max' in
                    self.configs['sims']['fibers']['z_parameters'].keys()):
                fiber_length = model_length if override_length is None else override_length
                self.configs['sims']['fibers']['z_parameters']['min'] = 0
                self.configs['sims']['fibers']['z_parameters']['max'] = fiber_length

                if override_length is None:
                    warnings.warn('Program assumed fiber length same as proximal length since "min" and "max" fiber '
                                  'length not defined in Config.Sim "fibers" -> "z_parameters"')
            else:
                fiber_length = (self.search(Config.SIM, 'fibers', FiberZMode.parameters.value, 'max')
                                - self.search(Config.SIM, 'fibers', FiberZMode.parameters.value, 'min')) \
                    if override_length is None else override_length

            half_fiber_length = fiber_length / 2
            z_shift_to_center = (model_length - fiber_length) / 2.0

            xy_mode_name: str = self.search(Config.SIM, 'fibers', 'xy_parameters', 'mode')
            xy_mode: FiberXYMode = [mode for mode in FiberXYMode if str(mode).split('.')[-1] == xy_mode_name][0]

            # check that proximal model length is greater than or equal to fiber length (fibers only in nerve trunk)
            # override this functionality if using SL (not in nerve trunk)
            if not xy_mode == FiberXYMode.SL_PSEUDO_INTERP:
                assert model_length >= fiber_length, 'proximal length: ({}) < fiber length: ({})'.format(model_length,
                                                                                                         fiber_length)

            fiber_geometry_mode_name: str = self.search(Config.SIM, 'fibers', 'mode')

            # use key from above to get myelination mode from fiber_z
            if super_sample:
                myelinated = False
            else:
                myelinated: bool = self.search(
                    Config.FIBER_Z,
                    MyelinationMode.parameters.value,
                    fiber_geometry_mode_name,
                    "myelinated"
                )

                diams = []
                my_z_seed = self.search(Config.SIM, 'fibers', FiberZMode.parameters.value, 'seed')
                diameter = self.search(Config.SIM, 'fibers', FiberZMode.parameters.value, 'diameter')
                diam_distribution: bool = True if type(diameter) is dict else False
                if diam_distribution:
                    sampling_mode = self.search(Config.FIBER_Z,
                                                MyelinationMode.parameters.value,
                                                fiber_geometry_mode_name,
                                                "sampling")
                    if sampling_mode != MyelinatedSamplingType.INTERPOLATION.value:
                        self.throw(104)

                    distribution_mode_name = self.search(Config.SIM,
                                                         'fibers',
                                                         FiberZMode.parameters.value, 'diameter',
                                                         'mode')
                    distribution_mode: DiamDistMode = [mode for mode in DiamDistMode if
                                                       str(mode).split('.')[-1] == distribution_mode_name][0]
                    # seed rng
                    my_diam_seed: int = self.search(Config.SIM,
                                                    "fibers",
                                                    FiberZMode.parameters.value,
                                                    'diameter',
                                                    'seed')
                    np.random.seed(my_diam_seed)

                    fiber_diam_dist = None
                    if distribution_mode == DiamDistMode.UNIFORM:

                        # load parameters
                        lower_fiber_diam: float = self.search(Config.SIM,
                                                              "fibers",
                                                              FiberZMode.parameters.value,
                                                              'diameter',
                                                              'lower')
                        upper_fiber_diam: float = self.search(Config.SIM,
                                                              "fibers",
                                                              FiberZMode.parameters.value,
                                                              'diameter',
                                                              'upper')

                        # parameter checking
                        # positive values, order makes sense, etc
                        if lower_fiber_diam < 0:
                            self.throw(100)
                        if lower_fiber_diam > upper_fiber_diam:
                            self.throw(101)

                        fiber_diam_dist = stats.uniform(lower_fiber_diam, upper_fiber_diam - lower_fiber_diam)

                    elif distribution_mode == DiamDistMode.TRUNCNORM:

                        # load parameters
                        n_std_fiber_diam_limit: float = self.search(Config.SIM,
                                                                    "fibers",
                                                                    FiberZMode.parameters.value,
                                                                    'diameter',
                                                                    'n_std_limit')
                        mu_fiber_diam: float = self.search(Config.SIM,
                                                           "fibers",
                                                           FiberZMode.parameters.value,
                                                           'diameter',
                                                           'mu')
                        std_fiber_diam: float = self.search(Config.SIM,
                                                            "fibers",
                                                            FiberZMode.parameters.value,
                                                            'diameter',
                                                            'std')
                        lower_fiber_diam = mu_fiber_diam - n_std_fiber_diam_limit * std_fiber_diam
                        upper_fiber_diam = mu_fiber_diam + n_std_fiber_diam_limit * std_fiber_diam

                        # parameter checking
                        # positive values, order makes sense, etc
                        if n_std_fiber_diam_limit == 0 and std_fiber_diam != 0:
                            self.throw(102)
                        if lower_fiber_diam < 0:
                            self.throw(103)

                        fiber_diam_dist = stats.truncnorm((lower_fiber_diam - mu_fiber_diam) / std_fiber_diam,
                                                          (upper_fiber_diam - mu_fiber_diam) / std_fiber_diam,
                                                          loc=mu_fiber_diam,
                                                          scale=std_fiber_diam)

                    diams = fiber_diam_dist.rvs(len(fibers_xy))

            if myelinated and not super_sample:  # MYELINATED
                random.seed(my_z_seed)
                if len(diams) == 0:
                    diams = [diameter] * len(fibers_xy)
                for (x, y), diam in zip(fibers_xy, diams):
                    zs, delta_z, z_shift = generate_myel_fiber_zs(z_shift_to_center, diam)

                    fiber_pre = build_fiber_with_offset(zs,
                                                        myelinated,
                                                        fiber_length,
                                                        delta_z,
                                                        x, y,
                                                        z_shift_to_center + z_shift)
                    if diam_distribution:
                        fiber = {'diam': diam, 'fiber': fiber_pre}
                    else:
                        fiber = fiber_pre
                    fibers.append(fiber)

            else:  # UNMYELINATED
                if super_sample:
                    if 'dz' in self.configs[Config.SIM.value]['supersampled_bases'].keys():
                        delta_z = self.search(Config.SIM, 'supersampled_bases', 'dz')
                        my_z_seed = 123
                    else:
                        self.throw(79)

                else:
                    delta_z = self.search(Config.FIBER_Z,
                                          MyelinationMode.parameters.value,
                                          fiber_geometry_mode_name,
                                          'delta_zs')

                z_top_half = np.arange(fiber_length / 2, fiber_length + delta_z, delta_z)
                z_bottom_half = -np.flip(z_top_half) + fiber_length
                while z_top_half[-1] > fiber_length:
                    # trim top of top half
                    z_top_half = z_top_half[:-1]
                    z_bottom_half = z_bottom_half[1:]

                for x, y in fibers_xy:
                    fiber = build_fiber_with_offset(list(np.concatenate((z_bottom_half[:-1], z_top_half))),
                                                    myelinated,
                                                    fiber_length,
                                                    delta_z,
                                                    x, y,
                                                    z_shift_to_center)
                    fibers.append(fiber)

        else:
            self.throw(31)

        return fibers
