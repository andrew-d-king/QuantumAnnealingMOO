# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from dwave.system.samplers import DWaveSampler
from helpers import *

num_reads = 1000  # Number of QPU shots taken per call
num_objectives = 4  # Number of objective functions, either 3 or 4
num_iterations = 500  # Maximum number of QPU calls to combine
num_repetitions = 5  # Number of repetitions for error bars
solver_index = 0
solvername = ['Advantage2_system1.6', 'Advantage_system4.1'][solver_index]

hv_max = get_hv_max(num_objectives)
reference_point = get_reference_point(num_objectives)

embeddings = np.loadtxt(f'../data/embeddings_{solvername}.txt', dtype=int)  # load disjoint 42q embeddings
num_embeddings, N = embeddings.shape

weights = fetch_weights(num_objectives)
sampler = DWaveSampler(profile=solvername)

for max_iterations in range(num_iterations):

    for experiment_repetition in range(num_repetitions):

        # Try to load data from a partially completed experiment
        try:
            data = np.load(f'../data/results/{solvername}/Nobj{num_objectives}_rep{experiment_repetition}.npz')

            results = {
                'qpu_access_time': data['qpu_access_time'].tolist(),
                'qpu_annealing_time': data['qpu_annealing_time'].tolist(),
                'hypervolume': data['hypervolume'].tolist(),
                'nondominated_samples': data['nondominated_samples'],
            }

        except FileNotFoundError:
            results = {
                'qpu_access_time': [],
                'qpu_annealing_time': [],
                'hypervolume': [],
                'nondominated_samples': None,
            }

        for iteration in range(len(results['hypervolume']), num_iterations):
            print(f'Starting iteration {iteration:4d}... ', end='')
            c_vectors = make_c_vectors(experiment_repetition, iteration, embeddings, num_objectives)
            J = make_couplings(c_vectors, embeddings)

            print(f'calling QPU... ', end='')
            sampleset = sampler.sample_ising({}, J, num_reads=num_reads, auto_scale=True, annealing_time=1)

            print(f'collating QPU output... ', end='')
            samples = extract_distinct_samples(sampleset, embeddings)

            print(f'combining nondominated samples with previous... ', end='')
            results['nondominated_samples'] = build_nondominated_samples(
                results['nondominated_samples'], samples, weights)

            print(f'computing hypervolume... ', end='')
            results['hypervolume'].append(get_hypervolume(results['nondominated_samples'], weights, reference_point))
            results['qpu_access_time'].append(sampleset.info['timing']['qpu_access_time'] / 1e6)
            results['qpu_annealing_time'].append(
                sampleset.info['timing']['qpu_anneal_time_per_sample'] / 1e6 * num_reads)

            print(f'saving to file... ', end='')
            np.savez_compressed(f'../data/results/{solvername}/Nobj{num_objectives}_rep{experiment_repetition}.npz',
                                **results)
            print(f'DONE. ')

            print(
                f'Rep{experiment_repetition:2d} after {iteration + 1} QPU calls: '
                f'{len(results["nondominated_samples"]):5d} nondominated points, hypervolume is '
                f'{results["hypervolume"][-1]:.9f}, HVmax-HV = {hv_max - results["hypervolume"][-1]:.9f}'
            )
