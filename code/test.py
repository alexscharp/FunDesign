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
from phase1 import simulate_unrolling, plot_trajectory, create_animation

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


def test_animation():
    # 1. set parameters
    init_pegs = np.array([[0.25, 0.38], [0.4, 0.1],
                          [0.6, 0.4], [0.0, 0.6], [0.4, 0.6], [0.2, 0.8]])
    init_pegs = np.array([[0.25, 0.35], [0.4, 0.1],
                          [0.6, 0.4], [0.05, 0.6], [0.4, 0.6], [0.22, 0.7],
                          [-0.25, 1.5]])
    init_nodes = np.array([[0.0, 0.0], [0.3, 0.5],
                           [1.0, 0.5], [1.5, 0.2]])
    direction = 1
    diameter = 0.2
    init_setting = (init_nodes, init_pegs, direction, diameter)

    # 2. simulate unrolling process
    trajectory = simulate_unrolling(init_nodes, init_pegs, diameter, direction)
    origins, beziers, arcs, diameters, collide_pegs = trajectory

    # 3. plot the trajectory
    # fig, ax = plt.subplots()
    # plot_trajectory(init_setting, trajectory, ax=ax)
    # plt.show()

    # animation example
    num_pts = 400
    step = 2
    outfile = 'unrolling-with-original.mp4'
    film_writer_title = 'unrolling'
    circle_pnts = 40
    create_animation(init_setting, trajectory, num_pts, step, circle_pnts,
                     outfile=outfile, film_writer_title=film_writer_title)


if __name__ == "__main__":
    # test_unrolling()
    test_animation()
