#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
"""
This module contains code for rotating bezier curve
"""

# standard library
import sys

# third party library
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import bezier

sns.set() # use seaborn style


if __name__ == "__main__":
    # cases 1
    # init_nodes = np.array([[0.0, 0.0], [0.3, 0.5],
    #    [1.0, 0.5], [1.5, 0.2]])
    # cases 2
    # init_nodes = np.array([[0.0, 0.0], [0.3, 0.5],
    #    [1.0, 0.1], [1.58, 0.2]])
    # cases 3 length = 0.8 * original
    # init_nodes = np.array([[0, 0], [0.24, 0.4], [0.736, 0.48], [1.1808, 0.3424]])
    # cases 3 length = 0.6 * original
    # init_nodes = np.array([[0, 0], [0.18, 0.3], [0.504, 0.42], [0.8424, 0.4032]])
    # init_nodes = np.array([[0, 0], [0.12, 0.2], [0.304, 0.32], [0.5136,
    #    0.3728]])
    # cases 4
    # init_nodes = np.array([[0.0, 0.0], [0.3, 0.2],
    #    [1.0, 0.1], [1.65, 0.2]])
    # cases 5
    # init_nodes = np.array([[0.0, 0.0], [0.3, 0.2],
    #    [1.0, 0.1], [1.65, 0.05]])
    # cases 6
    init_nodes1 = np.array([[0.0, 0.0], [0.3, 0.5],
        [1.0, 0.5], [1.5, 0.2]])
    init_nodes2 = np.array([[0.0, 0.0], [0.3, 1.0],
        [1.0, -0.5], [1.5, 0.2]])
    init_nodes3 = np.array([[0.0, 0.0], [0.3, -0.5],
        [1.0, 1.0], [1.5, 0.2]])
    curve1 = bezier.Curve.from_nodes(init_nodes1.transpose())
    curve2 = bezier.Curve.from_nodes(init_nodes2.transpose())
    curve3 = bezier.Curve.from_nodes(init_nodes3.transpose())
    print(curve1.length)
    print(curve2.length)
    print(curve3.length)
    # left_curve = curve.specialize(0, 0.4)
    # print(left_curve.nodes)
    fig, ax = plt.subplots()
    curve1.plot(ax=ax, num_pts=128)
    curve2.plot(ax=ax, num_pts=128)
    curve3.plot(ax=ax, num_pts=128)
    """
    # rotation example
    angle = np.pi / 6
    T = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    curve = bezier.Curve.from_nodes(np.dot(T, init_nodes))
    curve.plot(ax=ax, num_pts=128)
    """
    plt.show()
