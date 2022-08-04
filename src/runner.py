#!/usr/bin/env python3.7

"""
The copyrights of this software are owned by Duke University.
Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
"""


import base64
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
import warnings
from copy import deepcopy
from typing import List

import numpy as np
from quantiphy import Quantity
from shapely.geometry import Point

from src.core import Sample, Simulation, Waveform
from src.utils import (
    Config,
    Configurable,
    CuffShiftMode,
    Env,
    Exceptionable,
    ExportMode,
    NerveMode,
    PerineuriumResistivityMode,
    ReshapeNerveMode,
    SetupMode,
    TemplateOutput,
    WriteMode,
)


class Runner(Exceptionable, Configurable):
    def __init__(self, number: int):
        """
        :param number: the number of the run
        """

        # initialize Configurable super class
        Configurable.__init__(self)

        # initialize Exceptionable super class
        Exceptionable.__init__(self, SetupMode.NEW)

        # this corresponds to the run index (as file name in config/user/runs/<run_index>.json
        self.number = number

    def load_configs(self) -> dict:
        """Load all configuration files into class
        :return: dictionary of all configs (Sample, Model(s), Sims(s))
        """

        def validate_and_add(config_source: dict, key: str, path: str):
            """Validate and add config to class
            :param config_source: all configs, to which we add new ones
            :param key: the key of the dict in Configs
            :param path: path to the JSON file of the config
            :return: updated dict of all configs
            """
            self.validate_path(path)
            if os.path.exists(path):
                if key not in config_source:
                    config_source[key] = []
                try:
                    config_source[key] += [self.load(path)]
                except Exception:
                    warnings.warn(f'Issue loading {key} config: {path}')
                    self.throw(144)

            else:
                print(f'Missing {key} config: {path}')
                self.throw(37)

        configs = {}

        sample = self.search(Config.RUN, 'sample')

        if not isinstance(sample, int):
            self.throw(95)

        models = self.search(Config.RUN, 'models', optional=True)
        sims = self.search(Config.RUN, 'sims', optional=True)

        sample_path = os.path.join(os.getcwd(), 'samples', str(sample), 'sample.json')
        validate_and_add(configs, 'sample', sample_path)

        model_paths = [
            os.path.join(os.getcwd(), 'samples', str(sample), 'models', str(model), 'model.json') for model in models
        ]

        for model_path in model_paths:
            validate_and_add(configs, 'models', model_path)

        sim_paths = [os.path.join(os.getcwd(), 'config', 'user', 'sims', f'{sim}.json') for sim in sims]
        for sim_path in sim_paths:
            validate_and_add(configs, 'sims', sim_path)

        return configs

    def load_obj(self, path: str):
        """Load object from file
        :param path: path to python obj file
        :return: obj file
        """
        with open(path, 'rb') as o:
            object = pickle.load(o)
        object.add(SetupMode.OLD, Config.CLI_ARGS, self.configs[Config.CLI_ARGS.value])
        return object

    def setup_run(self):
        """perform all setup steps for a run
        :return: Dictionary of all configs
        """
        # load all json configs into memory
        all_configs = self.load_configs()

        run_pseudonym = self.configs[Config.RUN.value].get('pseudonym')
        if run_pseudonym is not None:
            print('Run pseudonym:', run_pseudonym)

        # ensure NEURON files exist in export location
        Simulation.export_neuron_files(os.environ[Env.NSIM_EXPORT_PATH.value])
        Simulation.export_system_config_files(os.path.join(os.environ[Env.NSIM_EXPORT_PATH.value], 'config', 'system'))

        for deprecated_key in ['break_points', 'local_avail_cpus', 'submission_context', 'partial_fem']:
            if deprecated_key in self.configs[Config.RUN.value]:
                warnings.warn(f"Specifying {deprecated_key} in run.json is deprecated, and has no effect.")

        return all_configs

    def generate_sample(self, all_configs, smart=True):
        """Generate the sample object for this run.
        :param all_configs: all configs for this run
        :param smart: if True, reuse objects from previous runs
        :return: (sample object, sample number)
        """

        sample_num = self.configs[Config.RUN.value]['sample']

        sample_file = os.path.join(os.getcwd(), 'samples', str(sample_num), 'sample.obj')

        sample_pseudonym = all_configs[Config.SAMPLE.value][0].get('pseudonym')

        print(
            f"SAMPLE {self.configs[Config.RUN.value]['sample']}",
            f'- {sample_pseudonym}' if sample_pseudonym is not None else '',
        )

        # instantiate sample
        if smart and os.path.exists(sample_file):
            print(f"Found existing sample {self.configs[Config.RUN.value]['sample']} ({sample_file})")
            sample = self.load_obj(sample_file)
        else:
            # init slide manager
            sample = Sample(self.configs[Config.EXCEPTIONS.value])
            # run processes with slide manager (see class for details)

            sample.add(SetupMode.OLD, Config.SAMPLE, all_configs[Config.SAMPLE.value][0]).add(
                SetupMode.OLD, Config.RUN, self.configs[Config.RUN.value]
            ).add(SetupMode.OLD, Config.CLI_ARGS, self.configs[Config.CLI_ARGS.value]).init_map(
                SetupMode.OLD
            ).build_file_structure().populate().write(
                WriteMode.SECTIONWISE2D
            ).output_morphology_data().save(
                os.path.join(sample_file)
            )

        return sample, sample_num

    def prep_model(self, all_configs, model_index, model_config, sample, sample_num):
        """Prepare model prior to handoff to Java
        :param all_configs: all configs for this run
        :param model_index: index of model
        :param model_config: config for this model
        :param sample: sample object
        :param sample_num: sample number
        :return: model number
        """
        model_num = self.configs[Config.RUN.value]['models'][model_index]
        model_pseudonym = model_config.get('pseudonym')
        print(f'\tMODEL {model_num}', f'- {model_pseudonym}' if model_pseudonym is not None else '')

        # use current model index to computer maximum cuff shift (radius) .. SAVES to file in method
        model_config = self.compute_cuff_shift(model_config, sample, all_configs[Config.SAMPLE.value][0])

        model_config_file_name = os.path.join(
            os.getcwd(), 'samples', str(sample_num), 'models', str(model_num), 'model.json'
        )

        # write edited model config in place
        TemplateOutput.write(model_config, model_config_file_name)

        # use current model index to compute electrical parameters ... SAVES to file in method
        self.compute_electrical_parameters(all_configs, model_index)

        return model_num

    def sim_setup(self, sim_index, sim_config, sample_num, model_num, smart, sample, model_config):
        """Create simulation object and prepare for generation of NEURON sims
        :param sim_index: index of sim
        :param sim_config: config for this sim
        :param sample_num: sample number
        :param model_num: model number
        :param smart: if True, use existing objects
        :param sample: sample object
        :param model_config: config for this model
        :return: simulation object, directory of sim
        """
        sim_num = self.configs[Config.RUN.value]['sims'][sim_index]
        sim_pseudonym = sim_config.get('pseudonym')
        print(
            f"\t\tSIM {self.configs[Config.RUN.value]['sims'][sim_index]}",
            f'- {sim_pseudonym}' if sim_pseudonym is not None else '',
        )

        sim_obj_dir = os.path.join(
            os.getcwd(), 'samples', str(sample_num), 'models', str(model_num), 'sims', str(sim_num)
        )

        sim_obj_file = os.path.join(sim_obj_dir, 'sim.obj')

        # init fiber manager
        if smart and os.path.exists(sim_obj_file):
            print(f'\t    Found existing sim object for sim {sim_index} ({sim_obj_file})')

            simulation: Simulation = self.load_obj(sim_obj_file)

        else:
            if not os.path.exists(sim_obj_dir):
                os.makedirs(sim_obj_dir)

            if not os.path.exists(sim_obj_dir + '/plots'):
                os.makedirs(sim_obj_dir + '/plots')

            simulation: Simulation = Simulation(sample, self.configs[Config.EXCEPTIONS.value])
            simulation.add(SetupMode.OLD, Config.MODEL, model_config).add(SetupMode.OLD, Config.SIM, sim_config).add(
                SetupMode.OLD, Config.RUN, self.configs[Config.RUN.value]
            ).add(
                SetupMode.OLD, Config.CLI_ARGS, self.configs[Config.CLI_ARGS.value]
            ).resolve_factors().write_waveforms(
                sim_obj_dir
            ).write_fibers(
                sim_obj_dir
            ).validate_srcs(
                sim_obj_dir
            ).save(
                sim_obj_file
            )
        return simulation, sim_obj_dir

    def validate_supersample(self, simulation, sample_num, model_num):
        """Validate supersampling parameters
        :param simulation: simulation object
        :param sample_num: sample number
        :param model_num: model number
        :return: directory of source simulation
        """
        source_sim_index = simulation.configs['sims']['supersampled_bases']['source_sim']

        source_sim_obj_dir = os.path.join(
            os.getcwd(), 'samples', str(sample_num), 'models', str(model_num), 'sims', str(source_sim_index)
        )

        # do Sim.fibers.xy_parameters match between Sim and source_sim?
        try:
            source_sim: simulation = self.load_obj(os.path.join(source_sim_obj_dir, 'sim.obj'))
            print(f'\t    Found existing source sim {source_sim_index} for supersampled bases ({source_sim_obj_dir})')
        except FileNotFoundError:
            traceback.print_exc()
            self.throw(129)

        source_xy_dict: dict = source_sim.configs['sims']['fibers']['xy_parameters']
        xy_dict: dict = simulation.configs['sims']['fibers']['xy_parameters']

        if source_xy_dict != xy_dict:
            self.throw(82)
        return source_sim_obj_dir

    def generate_nsims(self, sim_index, model_num, sample_num):
        """Generate NEURON simulations
        :param sim_index: index of sim
        :param model_num: model number
        :param sample_num: sample number
        :return: None
        """
        sim_num = self.configs[Config.RUN.value]['sims'][sim_index]
        sim_obj_path = os.path.join(
            os.getcwd(),
            'samples',
            str(self.configs[Config.RUN.value]['sample']),
            'models',
            str(model_num),
            'sims',
            str(sim_num),
            'sim.obj',
        )

        sim_dir = os.path.join(
            os.getcwd(), 'samples', str(self.configs[Config.RUN.value]['sample']), 'models', str(model_num), 'sims'
        )

        # load up correct simulation and build required sims
        simulation: Simulation = self.load_obj(sim_obj_path)
        simulation.build_n_sims(sim_dir, sim_num)

        # get export behavior
        export_behavior = None
        if self.configs[Config.CLI_ARGS.value].get('export_behavior') is not None:
            export_behavior = self.configs[Config.CLI_ARGS.value]['export_behavior']
        elif self.configs[Config.RUN.value].get('export_behavior') is not None:
            export_behavior = self.configs[Config.RUN.value]['export_behavior']
        else:
            export_behavior = 'selective'
        # check to make sure we have a valid behavior
        if not np.any([export_behavior == x.value for x in ExportMode]):
            self.throw(139)

        # export simulations
        Simulation.export_n_sims(
            sample_num,
            model_num,
            sim_num,
            sim_dir,
            os.environ[Env.NSIM_EXPORT_PATH.value],
            export_behavior=export_behavior,
        )

        # ensure run configuration is present
        Simulation.export_run(self.number, os.environ[Env.PROJECT_PATH.value], os.environ[Env.NSIM_EXPORT_PATH.value])

    def run(self, smart: bool = True):
        """Main function to run the pipeline.
        :param smart: bool telling the program whether to reprocess the sample or not if it already exists as sample.obj
        :return: nothing to memory, spits out all pipeline related data to file
        """
        # NOTE: single sample per Runner, so no looping of samples
        #       possible addition of functionality for looping samples in start.py

        all_configs = self.setup_run()

        self.potentials_exist: List[bool] = []  # if all of these are true, skip Java
        self.ss_bases_exist: List[bool] = []  # if all of these are true, skip Java

        sample, sample_num = self.generate_sample(all_configs, smart=smart)

        # iterate through models
        if 'models' not in all_configs:
            print('NO MODELS TO MAKE IN Config.RUN - killing process')
        else:
            for model_index, model_config in enumerate(all_configs[Config.MODEL.value]):
                # loop through each model
                model_num = self.prep_model(all_configs, model_index, model_config, sample, sample_num)
                if 'sims' in all_configs:
                    # iterate through simulations
                    for sim_index, sim_config in enumerate(all_configs['sims']):
                        # generate simulation object
                        simulation, sim_obj_dir = self.sim_setup(
                            sim_index, sim_config, sample_num, model_num, smart, sample, model_config
                        )
                        if (
                            'supersampled_bases' in simulation.configs['sims']
                            and simulation.configs['sims']['supersampled_bases']['use']
                        ):
                            source_sim_obj_dir = self.validate_supersample(simulation, sample_num, model_num)
                            self.ss_bases_exist.append(simulation.ss_bases_exist(source_sim_obj_dir))
                        else:
                            self.potentials_exist.append(simulation.potentials_exist(sim_obj_dir))

            if self.configs[Config.CLI_ARGS.value].get('break_point') == 'pre_java' or (
                ('break_points' in self.configs[Config.RUN.value])
                and self.search(Config.RUN, 'break_points').get('pre_java') is True
            ):
                print('KILLING PRE JAVA')
                return

            # handoff (to Java) -  Build/Mesh/Solve/Save bases; Extract/Save potentials if necessary
            if 'models' in all_configs and 'sims' in all_configs:
                self.model_parameter_checking(all_configs)
                # only transition to java if necessary (there are potentials that do not exist)
                if not all(self.potentials_exist) or not all(self.ss_bases_exist):
                    print('\nTO JAVA\n')
                    self.handoff(self.number)
                    print('\nTO PYTHON\n')
                else:
                    print('\nSKIPPING JAVA - all required extracted potentials already exist\n')

                self.remove(Config.RUN)
                run_path = os.path.join('config', 'user', 'runs', f'{self.number}.json')
                self.add(SetupMode.NEW, Config.RUN, run_path)

                #  continue by using simulation objects
                models_exit_status = self.search(Config.RUN, "models_exit_status")

                for model_index, _model_config in enumerate(all_configs[Config.MODEL.value]):
                    model_num = self.configs[Config.RUN.value]['models'][model_index]
                    conditions = [
                        models_exit_status is not None,
                        len(models_exit_status) > model_index,
                    ]
                    model_ran = models_exit_status[model_index] if all(conditions) else True
                    ss_use_notgen = []
                    # check if all supersampled bases are "use" and not generating
                    for sim_config in all_configs['sims']:
                        if (
                            'supersampled_bases' in sim_config
                            and sim_config['supersampled_bases']['use']
                            and not sim_config['supersampled_bases']['generate']
                        ):
                            ss_use_notgen.append(True)
                        else:
                            ss_use_notgen.append(False)
                    if model_ran or np.all(ss_use_notgen):
                        for sim_index, _sim_config in enumerate(all_configs['sims']):
                            # generate output neuron sims
                            self.generate_nsims(sim_index, model_num, sample_num)
                        print(
                            f'Model {model_num} data exported to appropriate '
                            f'folders in {os.environ[Env.NSIM_EXPORT_PATH.value]}'
                        )

                    elif not models_exit_status[model_index]:
                        print(
                            f'\nDid not create NEURON simulations for Sims associated with: \n'
                            f'\t Model Index: {model_num} \n'
                            f'since COMSOL failed to create required potentials. \n'
                        )

            elif 'models' in all_configs and 'sims' not in all_configs:
                # Model Configs Provided, but not Sim Configs
                print('\nTO JAVA\n')
                self.handoff(self.number)
                print('\nNEURON Simulations NOT created since no Sim indices indicated in Config.SIM\n')

    def handoff(self, run_number: int, class_name='ModelWrapper'):
        """Handoff to Java.
        :param run_number: int, run number
        :param class_name: str, class name of Java class to run
        :return: None
        """
        comsol_path = os.environ[Env.COMSOL_PATH.value]
        jdk_path = os.environ[Env.JDK_PATH.value]
        project_path = os.environ[Env.PROJECT_PATH.value]
        run_path = os.path.join(project_path, 'config', 'user', 'runs', f'{run_number}.json')

        # Encode command line args as jason string, then encode to base64 for passing to java
        argstring = json.dumps(self.configs[Config.CLI_ARGS.value])
        argbytes = argstring.encode('ascii')
        argbase = base64.b64encode(argbytes)
        argfinal = argbase.decode('ascii')

        if sys.platform.startswith('win'):  # windows
            server_command = [f'{comsol_path}\\bin\\win64\\comsolmphserver.exe', '-login', 'auto']
            compile_command = (
                f'""{jdk_path}\\javac" '
                f'-cp "..\\bin\\json-20190722.jar";"{comsol_path}\\plugins\\*" '
                f'model\\*.java -d ..\\bin"'
            )
            java_command = (
                f'""{comsol_path}\\java\\win64\\jre\\bin\\java" '
                f'-cp "{comsol_path}\\plugins\\*";"..\\bin\\json-20190722.jar";"..\\bin" '
                f'model.{class_name} "{project_path}" "{run_path}" "{argfinal}""'
            )
        else:
            server_command = [f'{comsol_path}/bin/comsol', 'mphserver', '-login', 'auto']

            compile_command = (
                f'{jdk_path}/javac -classpath ../bin/json-20190722.jar:'
                f'{comsol_path}/plugins/* model/*.java -d ../bin'
            )
            # https://stackoverflow.com/questions/219585/including-all-the-jars-in-a-directory-within-the-java-classpath
            if sys.platform.startswith('linux'):  # linux
                java_comsol_path = comsol_path + '/java/glnxa64/jre/bin/java'
            else:  # mac
                java_comsol_path = comsol_path + '/java/maci64/jre/Contents/Home/bin/java'

            java_command = (
                f'{java_comsol_path} '
                f'-cp .:$(echo {comsol_path}/plugins/*.jar | '
                f'tr \' \' \':\'):../bin/json-20190722.jar:'
                f'../bin model.{class_name} "{project_path}" "{run_path}" "{argfinal}"'
            )

        # start comsol server
        subprocess.Popen(server_command, close_fds=True)
        # wait for server to start
        time.sleep(10)
        os.chdir('src')
        # compile java code
        exit_code = os.system(compile_command)
        if exit_code != 0:
            self.throw(140)
        # run java code
        exit_code = os.system(java_command)
        if exit_code != 0:
            self.throw(141)
        os.chdir('..')

    def compute_cuff_shift(self, model_config: dict, sample: Sample, sample_config: dict):
        """Compute the Cuff Shift for a given model.
        :param model_config: dict, model config
        :param sample: Sample, sample object
        :param sample_config: dict, sample config
        :return: model_config: dict, model config
        """
        # NOTE: ASSUMES SINGLE SLIDE

        # add temporary model configuration
        self.add(SetupMode.OLD, Config.MODEL, model_config)
        self.add(SetupMode.OLD, Config.SAMPLE, sample_config)

        # fetch slide
        slide = sample.slides[0]

        # fetch nerve mode
        nerve_mode: NerveMode = self.search_mode(NerveMode, Config.SAMPLE)

        if nerve_mode == NerveMode.PRESENT:
            if 'deform_ratio' not in self.configs[Config.SAMPLE.value]:
                deform_ratio = 1
            else:
                deform_ratio = self.search(Config.SAMPLE, 'deform_ratio')
            if deform_ratio > 1:
                self.throw(109)
        else:
            deform_ratio = None

        # get center and radius of nerve's min_bound circle
        nerve_copy = deepcopy(slide.nerve if nerve_mode == NerveMode.PRESENT else slide.fascicles[0].outer)

        # fetch cuff config
        cuff_config: dict = self.load(
            os.path.join(os.getcwd(), "config", "system", "cuffs", model_config['cuff']['preset'])
        )

        (
            cuff_code,
            cuff_r_buffer,
            expandable,
            offset,
            r_bound,
            r_f,
            theta_c,
            theta_i,
            x,
            y,
        ) = self.get_cuff_shift_parameters(cuff_config, deform_ratio, nerve_copy, sample_config, slide)

        r_i, theta_f = self.check_cuff_expansion_radius(cuff_code, cuff_config, expandable, r_f, theta_i)

        # remove sample config
        self.remove(Config.SAMPLE)

        cuff_shift_mode: CuffShiftMode = self.search_mode(CuffShiftMode, Config.MODEL)

        if cuff_shift_mode not in CuffShiftMode:
            self.throw(154)

        # remove (pop) temporary model configuration
        model_config = self.remove(Config.MODEL)
        model_config['min_radius_enclosing_circle'] = r_bound
        if slide.orientation_angle is not None:
            theta_c = (
                (slide.orientation_angle) * (360 / (2 * np.pi)) % 360
            )  # overwrite theta_c, use our own orientation

        # check if a naive mode was chosen
        naive = cuff_shift_mode in [
            CuffShiftMode.NAIVE_ROTATION_MIN_CIRCLE_BOUNDARY,
            CuffShiftMode.NAIVE_ROTATION_TRACE_BOUNDARY,
        ]

        # initialize as 0, only replace values as needed, must be initialized here in case cuff shift mode is NONE
        x_shift = y_shift = 0
        # set pos_ang
        if naive or cuff_shift_mode == CuffShiftMode.NONE:
            model_config['cuff']['rotate']['pos_ang'] = 0
            if slide.orientation_point is not None:
                print(
                    'Warning: orientation tif image will be ignored because a NAIVE or NONE cuff shift mode was chosen.'
                )
        else:
            model_config['cuff']['rotate']['pos_ang'] = theta_c - theta_f

        # min circle x and y shift
        if cuff_shift_mode in [
            CuffShiftMode.NAIVE_ROTATION_MIN_CIRCLE_BOUNDARY,
            CuffShiftMode.AUTO_ROTATION_MIN_CIRCLE_BOUNDARY,
        ]:

            if r_i > r_f:
                x_shift = x - (r_i - offset - cuff_r_buffer - r_bound) * np.cos(theta_c * ((2 * np.pi) / 360))
                y_shift = y - (r_i - offset - cuff_r_buffer - r_bound) * np.sin(theta_c * ((2 * np.pi) / 360))

            elif slide.nerve is None or deform_ratio != 1:
                x_shift, y_shift = x, y

        # min trace modes
        elif cuff_shift_mode in [
            CuffShiftMode.NAIVE_ROTATION_TRACE_BOUNDARY,
            CuffShiftMode.AUTO_ROTATION_TRACE_BOUNDARY,
        ]:
            if r_i < r_f:
                x_shift, y_shift = x, y

            else:
                id_boundary = Point(0, 0).buffer(r_i - offset)
                n_boundary = Point(x, y).buffer(r_f)

                if id_boundary.boundary.distance(n_boundary.boundary) < cuff_r_buffer:
                    nerve_copy.shift([x, y, 0])
                    print(
                        "WARNING: NERVE CENTERED ABOUT MIN CIRCLE CENTER (BEFORE PLACEMENT) BECAUSE "
                        "CENTROID PLACEMENT VIOLATED REQUIRED CUFF BUFFER DISTANCE\n"
                    )

                center_x = 0
                center_y = 0
                step = 1  # [um] STEP SIZE
                x_step = step * np.cos(-theta_c + np.pi)  # STEP VECTOR X-COMPONENT
                y_step = step * np.sin(-theta_c + np.pi)  # STEP VECTOR X-COMPONENT

                # shift nerve within cuff until one step within the minimum separation from cuff
                while nerve_copy.polygon().boundary.distance(id_boundary.boundary) >= cuff_r_buffer:
                    nerve_copy.shift([x_step, y_step, 0])
                    center_x -= x_step
                    center_y -= y_step

                # to maintain minimum separation from cuff, reverse last step
                center_x += x_step
                center_y += y_step

                x_shift, y_shift = center_x, center_y

        model_config['cuff']['shift']['x'] = x_shift
        model_config['cuff']['shift']['y'] = y_shift

        return model_config

    def get_cuff_shift_parameters(self, cuff_config, deform_ratio, nerve_copy, sample_config, slide):
        # fetch 1-2 letter code for cuff (ex: 'CT')
        cuff_code: str = cuff_config['code']
        # fetch radius buffer string (ex: '0.003 [in]')
        cuff_r_buffer_str: str = [
            item["expression"]
            for item in cuff_config["params"]
            if item["name"] == '_'.join(['thk_medium_gap_internal', cuff_code])
        ][0]
        # calculate value of radius buffer in micrometers (ex: 76.2)
        cuff_r_buffer: float = Quantity(
            Quantity(
                cuff_r_buffer_str.translate(cuff_r_buffer_str.maketrans('', '', ' []')),
                scale='m',
            ),
            scale='um',
        ).real  # [um] (scaled from any arbitrary length unit)
        # Get the boundary and center information for computing cuff shift
        if self.search_mode(ReshapeNerveMode, Config.SAMPLE) and not slide.monofasc() and deform_ratio == 1:
            x, y = 0, 0
            r_bound = np.sqrt(sample_config['Morphology']['Nerve']['area'] / np.pi)
        else:
            x, y, r_bound = nerve_copy.make_circle()
        # next calculate the angle of the "centroid" to the center of min bound circle
        # if mono fasc, just use 0, 0 as centroid (i.e., centroid of nerve same as centroid of all fasc)
        # if poly fasc, use centroid of all fascicle as reference, not 0, 0
        # angle of centroid of nerve to center of minimum bounding circle
        reference_x = reference_y = 0.0
        if not slide.monofasc() and not (round(slide.nerve.centroid()[0]) == round(slide.nerve.centroid()[1]) == 0):
            self.throw(123)  # if the slide has nerve and is not centered at the nerve throw error
        if not slide.monofasc():
            reference_x, reference_y = slide.fascicle_centroid()
        theta_c = (np.arctan2(reference_y - y, reference_x - x) * (360 / (2 * np.pi))) % 360
        # calculate final necessary radius by adding buffer
        r_f = r_bound + cuff_r_buffer
        # fetch initial cuff rotation (convert to rads)
        theta_i = cuff_config.get('angle_to_contacts_deg') % 360
        # fetch boolean for cuff expandability
        expandable: bool = cuff_config['expandable']
        offset = 0
        for key, coef in cuff_config["offset"].items():
            value_str = [item["expression"] for item in cuff_config["params"] if item['name'] == key][0]
            value: float = Quantity(
                Quantity(value_str.translate(value_str.maketrans('', '', ' []')), scale='m'),
                scale='um',
            ).real  # [um] (scaled from any arbitrary length unit)
            offset += coef * value
        return cuff_code, cuff_r_buffer, expandable, offset, r_bound, r_f, theta_c, theta_i, x, y

    def check_cuff_expansion_radius(self, cuff_code, cuff_config, expandable, r_f, theta_i):
        # check radius iff not expandable
        if not expandable:
            r_i_str: str = [
                item["expression"] for item in cuff_config["params"] if item["name"] == '_'.join(['R_in', cuff_code])
            ][0]
            r_i: float = Quantity(
                Quantity(r_i_str.translate(r_i_str.maketrans('', '', ' []')), scale='m'),
                scale='um',
            ).real  # [um] (scaled from any arbitrary length unit)

            if not r_f <= r_i:
                self.throw(51)

            theta_f = theta_i
        else:
            # get initial cuff radius
            r_i_str: str = [
                item["expression"]
                for item in cuff_config["params"]
                if item["name"] == '_'.join(['r_cuff_in_pre', cuff_code])
            ][0]
            r_i: float = Quantity(
                Quantity(r_i_str.translate(r_i_str.maketrans('', '', ' []')), scale='m'),
                scale='um',
            ).real  # [um] (scaled from any arbitrary length unit)

            if r_i < r_f:
                fixed_point = cuff_config.get('fixed_point')
                if fixed_point is None:
                    self.throw(126)
                if fixed_point == 'clockwise_end':
                    theta_f = theta_i * (r_i / r_f)
                elif fixed_point == 'center':
                    theta_f = theta_i
            else:
                theta_f = theta_i
        return r_i, theta_f

    def compute_electrical_parameters(self, all_configs, model_index):
        """Compute electrical parameters for a given model.
        :param all_configs: all configs for this run
        :param model_index: index of the model to compute parameters for
        :return: None, writes output to file
        """

        # fetch current model config using the index
        model_config = all_configs[Config.MODEL.value][model_index]
        model_num = self.configs[Config.RUN.value]['models'][model_index]

        # initialize Waveform object
        waveform = Waveform(self.configs[Config.EXCEPTIONS.value])

        # add model config to Waveform object, enabling it to generate waveforms
        waveform.add(SetupMode.OLD, Config.MODEL, model_config)

        # compute rho and sigma from waveform instance
        if (
            model_config.get('modes').get(PerineuriumResistivityMode.config.value)
            == PerineuriumResistivityMode.RHO_WEERASURIYA.value
        ):
            freq_double = model_config.get('frequency')
            rho_double = waveform.rho_weerasuriya(freq_double)
            sigma_double = 1 / rho_double
            tmp = {
                'value': str(sigma_double),
                'label': f'RHO_WEERASURIYA @ {freq_double} Hz',
                'unit': '[S/m]',
            }
            model_config['conductivities']['perineurium'] = tmp

        elif (
            model_config.get('modes').get(PerineuriumResistivityMode.config.value)
            == PerineuriumResistivityMode.MANUAL.value
        ):
            pass
        else:
            self.throw(48)

        dest_path: str = os.path.join(
            'samples', str(self.configs[Config.RUN.value]['sample']), 'models', str(model_num), 'model.json'
        )

        TemplateOutput.write(model_config, dest_path)

    def populate_env_vars(self):
        """Get environment variables from config file.
        :return: None
        """
        if Config.ENV.value not in self.configs:
            self.throw(75)

        for key in Env.vals.value:
            value = self.search(Config.ENV, key)
            assert type(value) is str
            os.environ[key] = value

    def model_parameter_checking(self, all_configs):
        """Check model parameters for validity.
        :param all_configs: all configs for this run
        :return: None
        """
        for _, model_config in enumerate(all_configs[Config.MODEL.value]):
            distal_exists = model_config['medium']['distal']['exist']
            if distal_exists and model_config['medium']['proximal']['distant_ground'] is True:
                self.throw(107)
