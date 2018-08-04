#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Alex Scharp, Yaqi Zhang
##################################
"""
This module contains test code
"""

# standard library
import sys

# third party library
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# local library
from phase1 import simulate_unrolling, plot_trajectory

sns.set()  # use seaborn style


def test_unrolling():
    # 1. set parameters
    init_pegs = np.array([[0.25, 0.38], [0.4, 0.1],
                          [0.6, 0.4], [0.0, 0.6], [0.4, 0.6], [0.2, 0.8]])
    init_nodes = np.array([[0.0, 0.0], [0.3, 0.5],
                           [1.0, 0.5], [1.5, 0.2]])
    direction = 1
    diameter = 0.2
    init_setting = (init_nodes, init_pegs, direction, diameter)

    # 2. simulate unrolling process
    trajectory = simulate_unrolling(init_nodes, init_pegs, diameter, direction)
    beziers, arcs, diameters, collide_pegs = trajectory

    # 3. plot the trajectory
    fig, ax = plt.subplots()
    plot_trajectory(init_setting, trajectory, ax=ax)
    plt.show()


if __name__ == "__main__":
    test_unrolling()
