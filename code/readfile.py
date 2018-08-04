#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
"""
This module contains functions to read
file that contains Bezier and Arc infos
"""

# standard library
import sys

# third party library
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import bezier

sns.set()  # use seaborn style


if __name__ == "__main__":
    path = '../output.txt'
    with open(path, 'r') as f:
        lines = f.readlines()
    beziers, arcs = [], []
    nlines = len(lines)
    n = 0
    while n < nlines:
        line = lines[n]
        if line.strip() == 'Bezier':
            blines = lines[n + 1:n + 5]
            beziers.append([[float(item) for item in bline.split()]
                            for bline in blines])
        elif line.strip() == 'Arc':
            cx, cy = lines[n + 1].split()
            cx, cy = float(cx), float(cy)
            radius = float(lines[n + 2]) / 2
            theta1 = float(lines[n + 3])
            theta2 = float(lines[n + 4])
            arcs.append((cx, cy, radius, theta1, theta2))
        n += 1
    # print(beziers)
    print(arcs)
    # plot bezier
    fig, ax = plt.subplots(figsize=(8, 8))
    num_pts = 128
    curves = [
        bezier.Curve.from_nodes(
            np.array(nodes).transpose()) for nodes in beziers]
    for curve in curves:
        curve.plot(ax=ax, num_pts=num_pts)
    num_pts = 100
    for cx, cy, radius, theta1, theta2 in arcs:
        theta1, theta2 = np.deg2rad(theta1), np.deg2rad(theta2)
        thetas = np.linspace(theta1, theta2, num_pts)
        xs = cx + radius * np.cos(thetas)
        ys = cy + radius * np.sin(thetas)
        ax.plot(xs, ys)
    plt.show()
