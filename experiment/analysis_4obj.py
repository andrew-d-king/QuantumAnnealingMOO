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
import matplotlib.pyplot as plt
from helpers import *

num_objectives = 4
solvernames = ['Advantage2_system1.7', 'Advantage_system4.1']

hv_max = get_hv_max(num_objectives)

plt.figure(figsize=(9, 6), layout='constrained')
for isolver,solvername in enumerate(solvernames):
    embeddings = np.loadtxt(f'../data/embeddings_{solvername}.txt', dtype=int)
    num_embeddings, N = embeddings.shape
    qtime = []
    atime = []
    hv = []
    for experiment_repetition in range(5):

        data = np.load(f'../data/results/{solvername}/Nobj{num_objectives}_rep{experiment_repetition}.npz')
        results = {
            'qpu_access_time': data['qpu_access_time'].tolist(),
            'qpu_annealing_time': data['qpu_annealing_time'].tolist(),
            'hypervolume': data['hypervolume'].tolist(),
            'nondominated_samples': data['nondominated_samples'],
        }
        qtime.append(results['qpu_access_time'])
        atime.append(results['qpu_annealing_time'])
        hv.append(results['hypervolume'])

    completed_iterations = min([len(_) for _ in qtime])
    qtime = np.array([_[:completed_iterations] for _ in qtime])
    atime = np.array([_[:completed_iterations] for _ in atime])
    hv = np.array([_[:completed_iterations] for _ in hv])

    plt.fill_between(np.cumsum(np.mean(np.array(qtime), axis=0)), 1 + hv_max - np.min(np.array(hv), axis=0),
                     1 + hv_max - np.max(np.array(hv), axis=0), alpha=0.2)
    plt.plot(np.cumsum(np.mean(np.array(qtime), axis=0)), 1 + hv_max - np.mean(np.array(hv), axis=0),
             label=fr'QA, {solvername}, measured access time')




data = dict(np.load('../data/s43588-025-00873-y_plotdata/plotdata_4obj_mps.npz'))
plt.fill_between(data['x'], 1 + hv_max - np.min(data['y'], axis=0),
                 1 + hv_max - np.max(data['y'], axis=0), alpha=0.2)
phandle = plt.plot(data['x'], 1 + hv_max - np.mean(data['y'], axis=0),
                   label=fr'$p=6$ QAOA, MPS simulation ($\chi=20$) assuming 10,000 shots/s')

data = dict(np.load('../data/s43588-025-00873-y_plotdata/plotdata_4obj_hw_3percent.npz'))
plt.plot(data['x'], data['y'], linestyle='--', color=str(data['color']),
         label=fr'$p=6$ QAOA, ibm_fez estimate for fidelity 3.71% assuming 10,000 shots/s')

data = dict(np.load('../data/s43588-025-00873-y_plotdata/plotdata_4obj_hw_53percent.npz'))
plt.plot(data['x'], data['y'], linestyle='--', color=str(data['color']),
         label=fr'$p=6$ QAOA, ibm_fez estimate for fidelity 53.0% assuming 10,000 shots/s')

data = dict(np.load('../data/s43588-025-00873-y_plotdata/plotdata_4obj_dcm.npz'))
plt.plot(data['x'], data['y'], linestyle=(0, (3, 2, 1, 2, 1, 2)), color=str(data['color']), label=data['label'])

data = dict(np.load('../data/s43588-025-00873-y_plotdata/plotdata_4obj_dpa-a.npz'))
plt.plot(data['x'], data['y'], linestyle=(0, (3, 3, 1, 3, 1, 3)), color=str(data['color']), label=data['label'])

plt.plot([.01, 1e5], [1, 1], linestyle='-', color=[.5, .5, .5])
plt.xlim([.1, 40000])

plt.loglog()
plt.grid(which='both', axis='both', color=[.9, .9, .9], alpha=.5)
plt.ylabel(r'$(\text{HV}_{\text{max}} - \text{HV}_t + 1)$', fontsize=12)
plt.xlabel(r'time (s)', fontsize=12)
plt.gca().invert_yaxis()
plt.legend(loc=(0., -0.6), mode="expand", frameon=False)

plt.title('Four objective functions')
plt.savefig(f'4obj_results.pdf', bbox_inches='tight')
plt.show()
