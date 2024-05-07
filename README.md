# Efficient Simulation of Multiphysics Problems with Application to Metal-Additive Manufacturing

This is the code repository for the corresponding dissertation thesis. It contains code that is addressed in chapters 6 and 7.

## Allen Cahn Equation

The Allen Cahn example is implemented using the ```Julia``` programming language.
If you would like to reproduce the results reported, navigate to ```allen-cahn``` in your terminal. There, you can set up the project using the ```Project.toml``` file with the commands:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Advection Equation

To run the advection benchmark, you need a working installation of [Firedrake](https://www.firedrakeproject.org/download.html) and its dependencies. Head to the linked website for further installation instructions.

To run the example, simply execute

```bash
python3 advection/advection_DG.py
```

for serial execution. To run in parallel using `n` processes, run

```bash
mpirun -np n python3 advection/advection_DG.py
```

By default, the script executes the Finite Volume variant of the example.
To switch to the Discontinuous Galerkin version, simply change the value of `degree` in line 20 to the desired order:

https://github.com/pzimbrod/dissertation-code/blob/main/advection/advection_DG.py#L20

## Laser Powder Bed Fusion

To run the advection benchmark, you need a working installation of [FEniCSx](https://fenicsproject.org/download/) and its dependencies. Head to the linked website for further installation instructions.

To run the example, simply execute

```bash
python3 powder-bed-fusion/main.py
```

for serial execution. To run in parallel using `n` processes, run

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