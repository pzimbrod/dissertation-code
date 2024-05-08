# Efficient Simulation of Multiphysics Problems with Application to Metal-Additive Manufacturing

This is the code repository for the corresponding dissertation thesis. It contains code that is addressed in chapters 6 and 7.

## Allen Cahn & Advection Equation

The codes for the two numerical examples are discussed in the publication: [doi:10.3390/asi7030035](https://doi.org/10.3390/asi7030035).
To run the examples, head to the `multiphysics-pde-methods` directory that links to the corresponding code repository here on GitHub. There, additional instructions on installation and running can be found.

## Laser Powder Bed Fusion

To run the advection benchmark, a working installation of [FEniCSx](https://fenicsproject.org/download/) and its dependencies are needed. Head to the linked website for further installation instructions.

To run the example on one process, simply execute

```bash
python3 powder-bed-fusion/main.py
```

To run in parallel using `n` processes, run

```bash
mpirun -np n python3 powder-bed-fusion/main.py
```

Alternatively, for a more documented version of this script, one can also execute the Jupyter notebook `powder-bed-fusion/main.ipynb`.

The parameters of the simulation can all be changed in the `main` file and are given in dictionary form.

Due to the complexity of the simulation, its classes are split up into different files and bundled in the `PBFModel` package.
The main data structures are:

- `PBFModel` represents the top-level data structure that encapsulates an entire instance of one simulation.
- `Mesh` is heavily based on the `dolfinx` implementation of a computational mesh, plus some additional attributes to abbreviate the remaining code.
- `FEData` all necessary data structures that are specific for the (generic formulation of the) Finite Element method.
- `FEOperators` contains the weak formulations of differential operators suitable for evaluation using Finite Elements. They are expressed in the [Unified Form Language](https://fenics.readthedocs.io/projects/ufl/en/latest/index.html).

Each class is more thoroughly documented in the respective source file.