# styrene
A Python framework for simulating industrial styrene adiabatic reactors using the kinetic model proposed by Lee & Fromend (2008) and program developed by Leite et al (2021).

## Contents
[Install](#install) / [Usage](#usage) / [Citation](#citation) / [References](#references) / [Contact](#contact)

## Install
First, make sure you have a Python 3 environment installed.

To install from github:
```
pip install -e git+https://github.com/bruscalia/collocation#egg=collocation
```

## Usage

```python
import numpy as np
from styrene.reactor import MultiBed
```

```python
test_reac = MultiBed()
test_reac.add_radial_bed(72950)
test_reac.set_inlet(T=886, P=1.25)
test_reac.add_radial_bed(82020)
test_reac.add_radial_bed(78330)
test_reac.add_resets(2, T=898.2)
test_reac.add_resets(3, T=897.6)
```

```python
test_reac.solve()
profiles = test_reac.get_dataframe()
```

```python
fig, ax = plt.subplots(figsize=[7, 4], dpi=100, sharex=True)

ax.plot(profiles.index * 1e-3, profiles["Fst"], color="darkgreen", label="Styrene")
ax.plot(profiles.index * 1e-3, profiles["Feb"], color="black", label="Ethylbenzene")

ax.set_ylabel("$F$ [kmol/h]")
ax.set_xlabel("$W$ [kg x 10³]")

ax.legend()

fig.tight_layout()
plt.show()
```

![profiles](data\composition_profiles_example.png)


## References
Lee, W. J. & Froment, G. F., 2008. Ethylbenzene Dehydrogenation into Styrene: Kinetic Modeling and Reactor Simulation. Industrial & Engineering Chemistry Research, February, 47(23), pp. 9183-9194.

Leite, B., Costa, A. O. S. & Costa Junior, E. F., 2021. Simulation and optimization of axial-flow and radial-flow reactors for dehydrogenation of ethylbenzene into styrene based on a heterogeneous kinetic model. Chem. Eng. Sci., Volume 244, p. 116805.

## Contact
e-mail: bruscalia12@gmail.com

