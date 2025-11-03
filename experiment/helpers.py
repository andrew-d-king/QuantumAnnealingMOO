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
import numpy as np
import urllib.request
import json
import moocore

def get_reference_point(Nobj):
    return {
        3: [-12.137398079531431, -19.64152167587139, -18.33061914071653],
        4: [-17.34831473307451, -25.11279714770653, -18.471718787635094, -17.89300836655866]
    }[Nobj]

def get_hv_max(Nobj):
    return {3: 43471.70365440166, 4: 1266143.349404145}[Nobj]
def fetch_weights(Nobj):
    filename = {3: {}, 4: {}}
    filename[3, 0] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_3o_0/problem_graph_0.json')
    filename[3, 1] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_3o_0/problem_graph_1.json')
    filename[3, 2] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_3o_0/problem_graph_2.json')
    filename[4, 0] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_4o_0/problem_graph_0.json')
    filename[4, 1] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_4o_0/problem_graph_1.json')
    filename[4, 2] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_4o_0/problem_graph_2.json')
    filename[4, 3] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_4o_0/problem_graph_3.json')

    weights = []
    for obj in range(Nobj):
        weights.append([])
        with urllib.request.urlopen(filename[Nobj, obj]) as response:
            data = json.loads(response.read().decode())
        weights[obj] = {
            tuple(sorted([edge['source'], edge['target']])): edge['weight'] for edge in data['links']
        }

    meta_weights = {
        key: np.array([w[key] for w in weights]) for key in weights[0]
    }
    return meta_weights

def make_c_vectors(experiment_repetition, iteration, embeddings, num_objectives):
    np.random.seed(10000 * experiment_repetition + iteration)
    rand_points = np.random.rand(len(embeddings), num_objectives + 1)
    rand_points[:, [0, 1]] = [0, 1]
    rand_points[:, 1] = 1
    return np.diff(np.sort(rand_points, axis=1), axis=1)

def make_couplings(c: np.array, embeddings):
    weights = fetch_weights(len(c[0]))
    J = {}

    for iemb, embedding in enumerate(embeddings):
        for u, v in weights:
            J[embedding[u], embedding[v]] = np.dot(weights[u, v], c[iemb])

    return J

def build_nondominated_samples(previous_samples, new_samples, _weights):
    if previous_samples is None:
        all_samples = np.unique(new_samples, axis=0)
    else:
        all_samples = np.unique(np.vstack((previous_samples, new_samples)), axis=0)
    cut_edges = all_samples[:, np.array(list(_weights.keys()))[:, 0]] != all_samples[:,
                                                                         np.array(list(_weights.keys()))[:, 1]]
    obj_fun = np.matmul(cut_edges, np.array(list(_weights.values())))
    return all_samples[moocore.is_nondominated(obj_fun, maximise=True, keep_weakly=False)]

def extract_distinct_samples(ss, embs):
    spin_array = ss.record.sample
    variables = ss.variables
    all_spins_array = np.ones((spin_array.shape[0], np.max(embs) + 1))
    all_spins_array[:, variables] = spin_array

    samples = np.unique(np.reshape(np.array([all_spins_array[:, emb].copy() for emb in embs]) == 1, (-1, len(embs[0]))),
                        axis=0)
    return np.unique(np.logical_xor(samples, np.tile(samples[:, 0], (samples.shape[1], 1)).T), axis=0)

def get_hypervolume(_samples, _weights, _reference_point):
    cut_edges = _samples[:, np.array(list(_weights.keys()))[:, 0]] != _samples[:, np.array(list(_weights.keys()))[:, 1]]
    obj_fun = np.matmul(cut_edges, np.array(list(_weights.values())))
    return moocore.hypervolume(-obj_fun, ref=-np.array(_reference_point))
