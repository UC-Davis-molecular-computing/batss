# batss Python package

The `batss` package is used for simulating population protocols. The package and further example notebooks can be found on [Github](https://github.com/UC-Davis-molecular-computing/batss).

If you find batss useful in a scientific project, please cite its associated paper:

> <ins>Exactly simulating stochastic chemical reaction networks in sub-constant time per reaction</ins>.  
  Joshua Petrack and David Doty.
  preprint  
  [ [paper](http://arxiv.org/abs/2508.04079) | [BibTeX](TODO) ]

NOTE: below here is old documentation from a related project [ppsim](https://github.com/UC-Davis-molecular-computing/ppsim-rust). Please check back later for updated documentation. In the meantime some examples of usage can be seen in the folder examples.

The core of the simulator uses a modification of a [batching algorithm](https://arxiv.org/abs/2005.03584) for population protocols (chemical reaction networks with exactly two reactants and two products per reaction) that gives significant asymptotic gains for protocols with relatively small reachable state sets. The package is designed to be run in a Python notebook, to concisely describe complex protocols, efficiently simulate their dynamics, and provide helpful visualization of the simulation.

## Table of contents

* [Installation](#installation)
* [First example protocol](#first-example-protocol)
* [Larger state protocol](#larger-state-protocol)
* [Protocol with Multiple Fields](#protocol-with-multiple-fields)
* [Simulating Chemical Reaction Networks (CRNs)](#simulating-chemical-reaction-networks-crns)

## Installation

The package can be installed with `pip` via


```python
pip install batss
```

## First example protocol

The most important part of the package is the `Simulation` class, which is responsible for parsing a protocol, performing the simulation, and giving data about the simulation.


```python
from batss import Simulation
```


## More examples
See [examples](examples/).