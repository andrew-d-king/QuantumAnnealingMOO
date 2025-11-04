## QuantumAnnealingMOO

### Running multi-objective max-cut problems on D-Wave quantum annealing processors

Code and data accompanying arXiv preprint [**Multi-objective optimization by quantum 
annealing**](http://arxiv.org/abs/2511.01762).

### Installation

Experiments can be reproduced or reanalyzed by installing the packages in `requirements.txt` under Python3.11.

### Structure

#### Data

Contains saved quantum annealing data and qubit embeddings for the solvers used. Saved data allows reanalysis of
experimental data without access to D-Wave QPUs.

Also includes, for comparison, plot data extracted from
from [**Quantum approximate multi-objective optimization**](https://www.nature.com/articles/s43588-025-00873-y) by Ayse
Kotil, Elijah Pelofske, Stephanie Riedmüller, Daniel J. Egger, Stephan Eidenbenz, Thorsten Koch, and Stefan Woerner, via
[zenodo.org/records/16878921](https://zenodo.org/records/16878921).

#### Experiment

Contains code for running quantum annealing experiments and analysis. Saved data will not be overwritten unless
parameters are changed to allow more iterations or more independent repetitions.

#### Analysis

The `analysis` object provides a framework for modular combination of data analysis and plotting. This part of the
project is relatively underdeveloped, but is used in the examples.

### Authors

- **Andrew D. King**

### License

This project is licensed under the [Apache-2.0](LICENSE) License - see the [LICENSE](LICENSE) file for
details

### Acknowledgments

We acknowledge the use of the [`moocore` package](https://github.com/multi-objective/moocore) implementing the
algorithms described in

* Manuel López-Ibáñez, Luís Paquete, and Thomas Stützle. [**Exploratory Analysis of Stochastic Local Search Algorithms
  in Biobjective Optimization.**](https://doi.org/10.1007/978-3-642-02538-9_9) In T. Bartz-Beielstein, M. Chiarandini,
  L. Paquete, and M. Preuss, editors,
  *Experimental Methods for the Analysis of Optimization Algorithms*, pages 209–222. Springer, Berlin, Germany, 2010.
  doi: 10.1007/978-3-642-02538-9_9
