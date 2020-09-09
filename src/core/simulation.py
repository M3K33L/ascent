import copy
import json
import os
import subprocess
from typing import Tuple, List

import itertools
import shutil
import distutils.dir_util as du
import pickle
import warnings

import numpy as np
import scipy.interpolate as sci
import matplotlib.pyplot as plt

from .hocwriter import HocWriter
from .fiberset import FiberSet
from .waveform import Waveform
from src.core import Sample
from src.utils import Exceptionable, Configurable, Saveable, SetupMode, Config, WriteMode, FiberXYMode, Env


class Simulation(Exceptionable, Configurable, Saveable):

    def __init__(self, sample: Sample, exception_config: list):

        # Initializes superclasses
        Exceptionable.__init__(self, SetupMode.OLD, exception_config)
        Configurable.__init__(self)

        self.sample = sample
        self.factors = dict()
        self.wave_product = []
        self.wave_key = []
        self.fiberset_product = []
        self.fiberset_key = []
        self.fiberset_map_pairs: List[Tuple[List, List]] = []
        self.ss_fiberset_map_pairs: List[Tuple[List, List]] = []
        self.src_product = []
        self.src_key = []
        self.potentials_product = []
        self.master_product_indices = []  # order: potentials (active_src, fiberset), waveform
        self.ss_product = []  # order: (contact index, fiberset)

    def resolve_factors(self) -> 'Simulation':

        if len(self.factors.items()) > 0:
            self.factors = dict()

        def search(dictionary, remaining_n_dims, path):
            if remaining_n_dims < 1:
                return
            for key, value in dictionary.items():
                if type(value) == list and len(value) > 1:
                    # print('adding key {} to sub {}'.format(key, sub))
                    self.factors[path + '->' + key] = value
                    remaining_n_dims -= 1
                elif type(value) == dict:
                    # print('recurse: {}'.format(value))
                    search(value, remaining_n_dims, path + '->' + key)

        for flag in ['fibers', 'waveform', 'supersampled_bases']:
            if flag in self.configs[Config.SIM.value].keys():
                search(
                    self.configs[Config.SIM.value][flag],
                    self.search(Config.SIM, "n_dimensions"),
                    flag
                )

        return self

    def write_fibers(self, sim_directory: str) -> 'Simulation':
        # loop PARAMS in here, but loop HISTOLOGY in FiberSet object

        fibersets_directory = os.path.join(sim_directory, 'fibersets')
        if not os.path.exists(fibersets_directory):
            os.makedirs(fibersets_directory)

        self.fibersets = []
        fiberset_factors = {key: value for key, value in self.factors.items() if key.split('->')[0] == 'fibers'}

        self.ss_fibersets = []

        self.fiberset_key = list(fiberset_factors.keys())

        self.fiberset_product = list(itertools.product(*fiberset_factors.values()))

        for i, fiberset_set in enumerate(self.fiberset_product):

            fiberset_directory = os.path.join(fibersets_directory, str(i))
            if not os.path.exists(fiberset_directory):
                os.makedirs(fiberset_directory)

            sim_copy = self._copy_and_edit_config(self.configs[Config.SIM.value], self.fiberset_key, list(fiberset_set))

            fiberset = FiberSet(self.sample, self.configs[Config.EXCEPTIONS.value])
            fiberset \
                .add(SetupMode.OLD, Config.SIM, sim_copy) \
                .add(SetupMode.OLD, Config.MODEL, self.configs[Config.MODEL.value]) \
                .generate(sim_directory) \
                .write(WriteMode.DATA, fiberset_directory)

            self.fiberset_map_pairs.append((fiberset.out_to_fib, fiberset.out_to_in))
            self.fibersets.append(fiberset)

        if 'supersampled_bases' not in self.configs[Config.SIM.value].keys():
            generate_ss_bases = False
        else:
            generate_ss_bases: bool = self.search(Config.SIM, 'supersampled_bases', 'generate')

        if not generate_ss_bases:
            pass

        else:

            ss_fibercoords_directory = os.path.join(sim_directory, 'ss_coords')

            if not os.path.exists(ss_fibercoords_directory):
                os.makedirs(ss_fibercoords_directory)

            fiberset = FiberSet(self.sample, self.configs[Config.EXCEPTIONS.value])
            fiberset \
                .add(SetupMode.OLD, Config.SIM, self.configs[Config.SIM.value]) \
                .add(SetupMode.OLD, Config.MODEL, self.configs[Config.MODEL.value]) \
                .generate(sim_directory, super_sample=generate_ss_bases) \
                .write(WriteMode.DATA, ss_fibercoords_directory)

            self.ss_fiberset_map_pairs.append((fiberset.out_to_fib, fiberset.out_to_in))
            self.ss_fibersets.append(fiberset)

        return self

    def write_waveforms(self, sim_directory: str) -> 'Simulation':
        directory = os.path.join(sim_directory, 'waveforms')
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.waveforms = []
        wave_factors = {key: value for key, value in self.factors.items() if key.split('->')[0] == 'waveform'}

        self.wave_key = list(wave_factors.keys())
        self.wave_product = list(itertools.product(*wave_factors.values()))

        for i, wave_set in enumerate(self.wave_product):
            sim_copy = self._copy_and_edit_config(self.configs[Config.SIM.value], self.wave_key, list(wave_set))

            waveform = Waveform(self.configs[Config.EXCEPTIONS.value])
            waveform \
                .add(SetupMode.OLD, Config.SIM, sim_copy) \
                .add(SetupMode.OLD, Config.MODEL, self.configs[Config.MODEL.value]) \
                .init_post_config() \
                .generate() \
                .write(WriteMode.DATA, os.path.join(directory, str(i))) \

            if 'plot' not in self.configs[Config.SIM.value]['waveform'].keys():
                plot = False
            else:
                plot: bool = self.search(Config.SIM, 'waveform', 'plot')

            if plot:
                waveform.plot(final=True)

            self.waveforms.append(waveform)

        return self

    def validate_srcs(self, sim_directory) -> 'Simulation':
        # potentials key (s) = (r x p)
        # index of line in output is s, write row containing of (r and p) to file

        # if active_srcs in sim config has key for the cuff in your model, use the list of contact weights
        cuff = self.search(Config.MODEL, "cuff", "preset")
        if cuff in self.configs[Config.SIM.value]["active_srcs"].keys():
            active_srcs_list = self.search(Config.SIM, "active_srcs", cuff)
        else:
            # otherwise, use the default weights (generally you don't want to rely on this as cuffs have different
            # numbers of contacts
            active_srcs_list = self.search(Config.SIM, "active_srcs", "default")
            print("\t\tWARNING: Attempting to use default value for active_srcs: {}".format(active_srcs_list))

        #  loop over the contact weights, make sure the the values obey two rules:
        #      (1) current conservation
        #      (2) unitary amounts solved for bases in COMSOL: if n_srcs > 1 then sum(abs) = 2, sum = 0
        #                                                      if n_srcs = 1 then = 1 (special case)
        for active_srcs in active_srcs_list:
            active_src_abs = [abs(src_weight) for src_weight in active_srcs]
            if not all(abs(i) <= 1 for i in active_src_abs):
                self.throw(93)

            if len(active_srcs) == 1:
                if sum(active_srcs) not in [1, -1]:
                    self.throw(50)
            else:
                # if sum(active_srcs) is not 0:
                #     self.throw(49)
                if sum(active_src_abs) is not 2:
                    self.throw(50)
                pass

        self.potentials_product = list(itertools.product(
            list(range(len(active_srcs_list))),
            list(range(len(self.fiberset_product)))
        ))

        self.ss_product = list(itertools.product(
            list(range(len(active_srcs_list[0]))),
            [0]))

        self.src_key = ['->'.join(['active_srcs', cuff])]
        self.src_product = active_srcs_list

        # loop over product
        output = [len(self.potentials_product)]
        for active_src_select, fiberset_select in self.potentials_product:
            output.append((active_src_select, fiberset_select))

        # write to file
        key_file_dir = os.path.join(sim_directory, "potentials")
        key_filepath = os.path.join(sim_directory, "potentials", "key.dat")
        if not os.path.exists(key_file_dir):
            os.mkdir(key_file_dir)

        with open(key_filepath, 'w') as f:
            for row in output:
                if not isinstance(row, int):
                    for el in row:
                        f.write(str(el) + ' ')
                else:
                    f.write(str(row) + ' ')
                f.write("\n")

        s_s = range(int(output[0]))
        q_s = range(len(self.wave_product))
        prods = list(itertools.product(s_s, q_s))
        self.master_product_indices = prods

        return self

    ############################

    def build_n_sims(self, sim_dir, sim_num) -> 'Simulation':

        def load(path: str):
            return pickle.load(open(path, 'rb'))

        # loop cartesian product
        # key_filepath = os.path.join(sim_dir, "potentials", "key.dat")  # s is line number
        # f = open(key_filepath, "r")
        # contents = f.read()
        # s_s = range(int(contents[0]))
        # q_s = range(len(self.wave_product))
        #
        # prods = list(itertools.product(s_s, q_s))
        # self.master_product_indices = prods

        n_inners = 0
        for fascicle in self.sample.morphology['Fascicles']:
            n_inners += len(fascicle["inners"])

        n_fiber_coords = []
        for fiberset in self.fibersets:
            n_fiber_coords.append(len(fiberset.fibers[0]))

        for t, (potentials_ind, waveform_ind) in enumerate(self.master_product_indices):
            # build file structure sim/#/n_sims/t/data/(inputs and outputs)
            self._build_file_structure(os.path.join(sim_dir, str(sim_num)), t)
            nsim_inputs_directory = os.path.join(sim_dir, str(sim_num), 'n_sims', str(t), 'data', 'inputs')

            # copy corresponding waveform to sim/#/n_sims/t/data/inputs
            source_waveform_path = os.path.join(sim_dir, str(sim_num), "waveforms", "{}.dat".format(waveform_ind))
            destination_waveform_path = os.path.join(sim_dir, str(sim_num), "n_sims", str(t), "data", "inputs", "waveform.dat")
            if not os.path.isfile(destination_waveform_path):
                shutil.copyfile(source_waveform_path, destination_waveform_path)

            # get source, waveform, and fiberset values for the corresponding neuron simulation t
            active_src_ind, fiberset_ind = self.potentials_product[potentials_ind]
            active_src_vals = [self.src_product[active_src_ind]]
            wave_vals = self.wave_product[waveform_ind]
            fiberset_vals = self.fiberset_product[fiberset_ind]

            # pair down simulation config to no lists of parameters (corresponding to the neuron simulation index t)
            # print('active_src_ind: {}'.format(str(active_src_ind)))
            # print('src_key: {}'.format(str(self.src_key)))
            # print('active_src_vals: {}'.format(str(active_src_vals)))

            sim_copy = self._copy_and_edit_config(self.configs[Config.SIM.value],
                                                  self.src_key, active_src_vals, copy_again=False)

            sim_copy = self._copy_and_edit_config(sim_copy,
                                                  self.wave_key, wave_vals, copy_again=False)

            sim_copy = self._copy_and_edit_config(sim_copy,
                                                  self.fiberset_key, fiberset_vals, copy_again=False)

            # save the paired down simulation config to its corresponding neuron simulation t folder
            with open(os.path.join(sim_dir, str(sim_num), "n_sims", str(t), "{}.json".format(t)), "w") as handle:
                handle.write(json.dumps(sim_copy, indent=2))

            n_tsteps = len(self.waveforms[waveform_ind].wave)

            # add config and write launch.hoc
            n_sim_dir = os.path.join(sim_dir, str(sim_num), "n_sims", str(t))
            hocwriter = HocWriter(os.path.join(sim_dir, str(sim_num)), n_sim_dir, self.configs[Config.EXCEPTIONS.value])
            hocwriter \
                .add(SetupMode.OLD, Config.MODEL, self.configs[Config.MODEL.value]) \
                .add(SetupMode.OLD, Config.SIM, sim_copy) \
                .build_hoc(n_inners, n_fiber_coords[fiberset_ind], n_tsteps)

            # copy in potentials data into neuron simulation data/inputs folder
            # the potentials files are matched to their inner and fiber index, and saved in destination folder with
            # this naming convention... this allows for control of upper/lower bounds for thresholds by fascicle
            # (useful since within a fascicle thresholds should be similar)
            p = fiberset_ind
            inner_list = []
            fiber_list = []

            # fetch xy mode to check for override necessity
            xy_mode_name: str = self.search(Config.SIM, 'fibers', 'xy_parameters', 'mode')
            xy_mode: FiberXYMode = [mode for mode in FiberXYMode if str(mode).split('.')[-1] == xy_mode_name][0]

            supersampled_bases: dict = self.search(Config.SIM, 'supersampled_bases')

            # NOT SUPER SAMPLING - PROBED COMSOL AT /FIBERSETS --> /POTENTIALS
            if supersampled_bases is None or supersampled_bases.get('use') is False:
                for root, dirs, files in os.walk(os.path.join(sim_dir, str(sim_num), 'potentials', str(p))):
                    for file in files:
                        q = int(file.split('.')[0])

                        # NOTE: if SL interp, writes files as inner0_fiber<q>.dat
                        l: int
                        k: int
                        if not xy_mode == FiberXYMode.SL_PSEUDO_INTERP:
                            l, k = self.indices_fib_to_n(p, q)
                        else:
                            l, k = 0, q

                        is_member = np.in1d(l, inner_list)
                        if not is_member:
                            inner_list.append(l)
                            if len(fiber_list) < len(inner_list):
                                fiber_list.append(0)
                            fiber_list[inner_list.index(l)] += 1

                        filename_direct = 'inner{}_fiber{}.dat'.format(l, k)
                        if not os.path.exists(nsim_inputs_directory):
                            os.makedirs(nsim_inputs_directory)

                        shutil.copyfile(
                            os.path.join(sim_dir, str(sim_num), 'potentials', str(potentials_ind), str(q) + '.dat'),
                            os.path.join(nsim_inputs_directory, filename_direct)
                        )

            # SUPER SAMPLING - PROBED COMSOL AT SS_COORDS --> /SS_BASES
            elif supersampled_bases is not None and supersampled_bases.get('use') is True:
                for root, dirs, files in os.walk(os.path.join(sim_dir, str(sim_num), 'fibersets', str(p))):
                    for file in files:

                        ss_bases = [[] for _ in active_src_vals[0]]
                        source_sim = supersampled_bases.get('source_sim')

                        # check that dz in source_sim matches the dz (if provided) in current sim

                        source_sim_obj_dir = os.path.join(sim_dir, str(source_sim))

                        source_sim_obj_file = os.path.join(source_sim_obj_dir, 'sim.obj')

                        source_simulation: Simulation = load(source_sim_obj_file)

                        source_dz = source_simulation.configs['sims']['supersampled_bases']['dz']

                        if 'dz' in supersampled_bases.keys():
                            if supersampled_bases.get('dz') != source_dz:
                                self.throw(79)
                        elif 'dz' not in supersampled_bases.keys():
                            warnings.warn('dz not provided in Sim, so will accept dz={} specified in source Sim'.format(source_dz))

                        for basis_ind in range(len(active_src_vals[0])):

                            ss_bases[basis_ind].append([])
                            ss_bases_src_path = os.path.join(sim_dir,
                                                          str(source_sim),
                                                          'ss_bases',
                                                          str(basis_ind))

                            ss_fiberset_path = os.path.join(sim_dir,
                                                            str(source_sim),
                                                            'ss_coords')

                            if not os.path.exists(ss_bases_src_path):
                                self.throw(81)

                            for f_root, f_dirs, f_files in os.walk(ss_bases_src_path):
                                for f_file in f_files:
                                    q = int(f_file.split('.')[0])

                                    if not os.path.exists(os.path.join(f_root, f_file)):
                                        self.throw(81)
                                    else:
                                        ss_bases[basis_ind][q] = np.loadtxt(os.path.join(f_root, f_file))[1:]

                                    if basis_ind == len(active_src_vals[0]) - 1:

                                        ss_weighted_bases_vec = np.zeros(len(ss_bases[basis_ind][q]))
                                        for src_ind, src_weight in enumerate(active_src_vals[0]):
                                            ss_weighted_bases_vec += ss_bases[src_ind][q]*src_weight

                                        # down-sample super_save_vec
                                        neuron_fiberset_file = open(os.path.join(root, file), 'r')
                                        neuron_fiberset_file_lines = neuron_fiberset_file.readlines()[1:]
                                        neuron_fiber_coords = []
                                        for neuron_fiberset_file_line in neuron_fiberset_file_lines:
                                            neuron_fiber_coords = \
                                                np.append(neuron_fiber_coords,
                                                          float(neuron_fiberset_file_line.split(' ')[-2])
                                                          )

                                        ss_fiberset_file = open(os.path.join(ss_fiberset_path, file), 'r')
                                        ss_fiberset_file_lines = ss_fiberset_file.readlines()[1:]
                                        ss_fiber_coords = []
                                        for ss_fiberset_file_line in ss_fiberset_file_lines:
                                            ss_fiber_coords = np.append(ss_fiber_coords,
                                                                        float(ss_fiberset_file_line.split(' ')[-2]))

                                        # create interpolation from super_coords and super_bases
                                        f = sci.interp1d(ss_fiber_coords, ss_weighted_bases_vec)
                                        neuron_potentials_input = f(neuron_fiber_coords)

                                        # as is convention, append length to start
                                        neuron_potentials_input = np.insert(neuron_potentials_input,
                                                                            0,
                                                                            len(neuron_potentials_input)
                                                                            )

                                        # NOTE: if SL interp, writes files as inner0_fiber<q>.dat
                                        l: int
                                        k: int
                                        if not xy_mode == FiberXYMode.SL_PSEUDO_INTERP:
                                            l, k = self.indices_fib_to_n(p, q)
                                        else:
                                            l, k = 0, q

                                        ss_filename = 'inner{}_fiber{}.dat'.format(l, k)

                                        np.savetxt(os.path.join(nsim_inputs_directory, ss_filename),
                                                   neuron_potentials_input,
                                                   fmt='%0.18f')

        return self

    def indices_fib_to_n(self, p, q) -> Tuple[int, int]:
        """
        :param p: fiberset index
        :param q: fiber index within fiberset
        :return: (l, k) as in "inner<l>_fiber<k>.dat" for NEURON sim
        """

        def search(arr, target) -> Tuple[int, int, int]:
            for a, outer in enumerate(arr):
                for b, inner in enumerate(outer):
                    for c, fib in enumerate(inner):
                        if fib == target:
                            return a, b, c

        out_fib, out_in = self.fiberset_map_pairs[p]
        #out_fib[0][0] = [n for n in range(100)]
        i, j, k = search(out_fib, q)
        return out_in[i][j], k

    def indices_n_to_fib(self, p, l, k) -> Tuple[int, int]:

        def search(arr, target) -> Tuple[int, int]:
            for a, outer in enumerate(arr):
                for b, inner in enumerate(outer):
                    if inner == target:
                        return a, b

        out_fib, out_in = self.fiberset_map_pairs[p]
        i, j = search(out_in, l)
        return out_fib[i][j][k]


    @staticmethod
    def _build_file_structure(sim_obj_dir, t):
        sim_dir = os.path.join(sim_obj_dir, "n_sims", str(t))

        if not os.path.exists(sim_dir):
            subfolder_names = ["inputs", "outputs"]
            for subfolder_name in subfolder_names:
                os.makedirs(os.path.join(sim_dir, "data", subfolder_name))

    def _copy_and_edit_config(self, config, key, set, copy_again=True):

        cp = config
        if copy_again:
            cp = copy.deepcopy(config)

        for path, value in zip(key, list(set)):
            path_parts = path.split('->')
            pointer = cp
            for path_part in path_parts[:-1]:
                pointer = pointer[path_part]
            pointer[path_parts[-1]] = value
        return cp

    @staticmethod
    def export_run(num: int, project_root: str, target: str, overwrite: bool = True):
        target_dir = os.path.join(target, 'runs')
        target_full = os.path.join(target_dir, str(num) + '.json')
        if overwrite and os.path.exists(target_full):
            os.remove(target_full)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        source = os.path.join(project_root, 'config', 'user', 'runs', str(num) + '.json')

        shutil.copy2(source, target_full)

    @staticmethod
    def export_n_sims(sample: int, model: int, sim: int, sim_obj_dir: str, target: str, overwrite: bool = True):

        sim_dir = os.path.join(sim_obj_dir, str(sim), 'n_sims')
        sim_export_base = os.path.join(target, 'n_sims', '{}_{}_{}_'.format(sample, model, sim))

        for product_index in [f for f in os.listdir(sim_dir) if os.path.isdir(os.path.join(sim_dir, f))]:
            target = sim_export_base + product_index

            if overwrite and os.path.exists(target):
                shutil.rmtree(target)

            shutil.copytree(
                os.path.join(sim_dir, product_index),
                sim_export_base + product_index
            )

    @staticmethod
    def export_neuron_files(target: str):
        # neuron files
        du.copy_tree(os.path.join(os.environ[Env.PROJECT_PATH.value], 'src', 'neuron'), target)

        submit_target = os.path.join(target, 'submit.py')
        if os.path.isfile(submit_target):
            os.remove(submit_target)

        submit_source = os.path.join('src', 'neuron', 'submit.py')
        shutil.copy2(submit_source, submit_target)

    @staticmethod
    def import_n_sims(sample: int, model: int, sim: int, sim_dir: str, source: str):

        sim_dir = os.path.join(sim_dir, 'n_sims')

        for dirname in [f for f in os.listdir(source) if os.path.isdir(os.path.join(source, f))]:
            this_sample, this_model, this_sim, product_index = tuple(dirname.split('_'))
            if sample == int(this_sample) and model == int(this_model) and sim == int(this_sim):
                shutil.rmtree(os.path.join(sim_dir, product_index))
                shutil.copytree(os.path.join(source, dirname), os.path.join(sim_dir, product_index))


    def potentials_exist(self, sim_dir: str) -> bool:
        """
        Return bool deciding if potentials have already been written
        :param sim_dir: directory of this simulation
        :return: boolean!
        """
        return all(os.path.exists(os.path.join(sim_dir, 'potentials', str(p))) for p, _ in self.master_product_indices)


    def ss_bases_exist(self, sim_dir: str) -> bool:
        """
        Return bool deciding if potentials have already been written
        :param sim_dir: directory of this simulation
        :return: boolean!
        """
        return all(
            os.path.exists(os.path.join(sim_dir, 'ss_bases', str(basis))) for basis, _ in self.ss_product
        )
