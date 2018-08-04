#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Yaqi Zhang
##################################
"""
This module contains code reading path.txt and pegs.txt
and plot
"""

# standard library
import sys

# third party library
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set()  # use seaborn style


if __name__ == "__main__":
    path = np.loadtxt('path.txt')
    pegs = np.loadtxt('pegs.txt')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(path[:, 0], path[:, 1], 'b-')
    ax.plot(pegs[:, 0], pegs[:, 1], 'go')
    plt.show()
