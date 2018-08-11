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
from subprocess import call

# third party library
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# local library
from unroll_pull import (simulate_unrolling, plot_trajectory,
                         unrolling_animation, create_movie_writer, _compute_path_length,
                         _remesh_paths, merge_paths, save_figure)

sns.set()  # use seaborn style


def create_freepulling(init_setting, trajectory, outfile=None,
                       film_writer_title='writer', num_ptr=400):
    points, _ = merge_paths(init_setting, trajectory, num_pts=400)
    path = []
    # remove the redundant points in the trajectory path
    for i in range(points.shape[0]):
        if i == 0 or not np.all(np.isclose(points[i, :], points[i - 1, :])):
            path.append(points[i])
    path = np.array(path)
    path = path[:, 0:2]
    num_pts = 400
    steps = 80
    path = _remesh_paths(path, num_pts)
    ts = np.linspace(0, 1, steps)
    path_obj = None
    fig, ax = plt.subplots()
    ax.set_xlim([-0.8, 1.2])
    ax.set_ylim([-0.0, 2.0])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.plot(init_pegs[:, 0], init_pegs[:, 1], 'go', markersize=6)
    if outfile:
        writer = create_movie_writer(title=film_writer_title, fps=10)
        writer.setup(fig, outfile=outfile, dpi=100)
        name, _ = outfile.rsplit('.', 1)
    for t in ts:
        if path_obj is not None:
            path_obj.remove()
        idx = int((1 - t) * num_pts)
        path_obj, = ax.plot(path[0:idx, 0], path[0:idx, 1], 'b-', linewidth=1)
        if outfile:
            writer.grab_frame()
            if t == ts[-1]:
                save_figure(fig, name + '.png', dpi=100)
        plt.pause(0.2)
        plt.draw()
    if outfile:
        writer.finish()
        print('Creating movie {:s}'.format(outfile))
    plt.show()


def create_pulling_animation(init_setting, trajectory, pivots,
                             outfile=None, film_writer_title='writer', pivots_file=None):
    points, _ = merge_paths(init_setting, trajectory, num_pts=400)
    path = []
    # remove the redundant points in the trajectory path
    for i in range(points.shape[0]):
        if i == 0 or not np.all(np.isclose(points[i, :], points[i - 1, :])):
            path.append(points[i])
    path = np.array(path)
    path = path[:, 0:2]
    # compute pivots
    vector = pivots[-1, :] - pivots[-2, :]
    vector /= np.linalg.norm(vector, 2)
    path_length = _compute_path_length(path)
    left_length = path_length - _compute_path_length(pivots)
    free_end = pivots[-1, :] + left_length * vector
    pivots = np.vstack((pivots, free_end))
    if pivots_file:
        np.savetxt(pivots_file, pivots, fmt='%0.8f', delimiter=' ')
        print('save pivots coordinates to {}'.format(pivots_file))
    num_pts = 400
    steps = 80
    init_path = path
    init_path = _remesh_paths(init_path, num_pts)
    end_path = _remesh_paths(pivots, num_pts)
    ts = np.linspace(0.0, 1.0, steps)
    path_obj = None
    fig, ax = plt.subplots()
    ax.set_xlim([-1.0, 1.4])
    ax.set_ylim([-0.4, 2.0])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.plot(init_pegs[:, 0], init_pegs[:, 1], 'go', markersize=6)
    if outfile:
        writer = create_movie_writer(title=film_writer_title, fps=10)
        writer.setup(fig, outfile=outfile, dpi=100)
        name, _ = outfile.rsplit('.')
    # print(_compute_path_length(init_path))
    # print(_compute_path_length(end_path))
    for t in ts:
        if path_obj is not None:
            path_obj.remove()
        path = (1 - t) * init_path + t * end_path
        path_obj, = ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=1)
        if outfile:
            writer.grab_frame()
            if t == ts[-1]:
                save_figure(fig, name + '.png', dpi=100)
        # plt.pause(0.2)
        # plt.draw()
    if outfile:
        writer.finish()
        print('Creating movie {:s}'.format(outfile))
    # plt.show()


def test_rotation(angle):
    # 1. set parameters
    init_pegs = np.array([[0.25, 0.35], [0.4, 0.1], [0.47, 0.29], [0.38, 0.38],
                          [0.6, 0.4], [0.4, 0.6], [0.20, 0.7],
                          [0.20, 0.15], [0.05, 1.27], [
                              0.25, 0.90], [0.03, 0.40],
                          [0.58, 0.63], [0.15, 0.05], [0.31, 0.20],
                          [0.64, 0.23], [0.31, 0.93], [0.03, 0.38],
                          [0.30, 0.03], [0.56, 0.06], [0.13, 1.0],
                          [0.38, 0.74], [0.11, 1.28], [0.17, 1.09],
                          [-0.01, 1.13], [0.42, 0.49],
                          [0.00, 0.88], [-0.11, 1.28], [0.45, 0.47],
                          [0.12, 0.33], [-0.02, 1.31],
                          [0.15, 0.69], [0.14, 0.46],  # YZ
                          [0.19, 0.49], [-0.23, 0.648],  # YZ ADD
                          [-0.15, 0.62],  # [0.05, 0.60]
                          ])
    init_nodes = np.array([[0.0, 0.0], [0.3, 0.5],
                           [1.0, 0.5], [1.5, 0.2]])

    T = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    init_nodes = np.dot(T, init_nodes.transpose()).transpose()

    direction = 1
    diameter = 0.1
    init_setting = (init_nodes, init_pegs, direction, diameter)

    # 2. simulate unrolling process
    trajectory = simulate_unrolling(init_nodes, init_pegs, diameter, direction)
    origins, beziers, arcs, diameters, collide_pegs = trajectory

    # 3. plot the trajectory
    fig, ax = plt.subplots()
    plot_trajectory(init_setting, trajectory, ax=ax)
    plt.show()


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
    origins, beziers, arcs, diameters, collide_pegs = trajectory

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
    unrolling_animation(init_setting, trajectory, num_pts, step, circle_pnts,
                        outfile=outfile, film_writer_title=film_writer_title)


def pipeline():
    # test_unrolling()
    # test_animation()
    # angle = float(sys.argv[1])
    # angle = np.deg2rad(angle)
    # test_rotation(angle)
    # 1. set parameters
    init_pegs = np.array([[0.25, 0.35], [0.4, 0.1],
                          [0.47, 0.29], [0.38, 0.38],
                          [0.6, 0.4], [0.4, 0.6], [0.20, 0.7],
                          [0.20, 0.15], [0.05, 1.27], [
                              0.25, 0.90], [0.03, 0.40],
                          [0.58, 0.63], [0.15, 0.05], [0.31, 0.20],
                          [0.64, 0.23], [0.31, 0.93], [0.03, 0.38],
                          [0.30, 0.03], [0.56, 0.06], [0.13, 1.0],
                          [0.38, 0.74], [0.11, 1.28], [0.17, 1.09],
                          [-0.01, 1.13], [0.42, 0.49],
                          [0.00, 0.88], [-0.11, 1.28], [0.45, 0.47],
                          [0.12, 0.33], [-0.02, 1.31],
                          [0.15, 0.69], [0.14, 0.46],  # YZ
                          [0.19, 0.49], [-0.23, 0.648],  # YZ ADD
                          [-0.15, 0.62],  # [0.05, 0.60]
                          ])
    np.savetxt('pegs.txt', init_pegs, fmt='%0.8f', delimiter=' ')

    # cases 1
    # init_nodes = np.array([[0.0, 0.0], [0.3, 0.5],
    #    [1.0, 0.5], [1.5, 0.2]])
    # cases 2 different shape
    # init_nodes = np.array([[0.0, 0.0], [0.3, 0.5],
    #    [1.0, 0.1], [1.58, 0.2]])
    # cases 3 different length
    # length = 0.8 * original angle = 61
    # init_nodes = np.array([[0, 0], [0.24, 0.4], [0.736, 0.48],
    #    [1.1808, 0.3424]])
    # length = 0.6 * original angle = 62.5
    # init_nodes = np.array([[0, 0], [0.18, 0.3], [0.504, 0.42], [0.8424, 0.4032]])
    # length = 0.4 * original angle = 61
    # init_nodes = np.array([[0, 0], [0.12, 0.2], [0.304, 0.32], [0.5136, 0.3728]])
    # cases 4 different shape
    # init_nodes = np.array([[0.0, 0.0], [0.3, 0.2],
    #    [1.0, 0.1], [1.65, 0.2]])

    # cases 5 different shape
    # init_nodes = np.array([[0.0, 0.0], [0.3, 0.2],
    #    [1.0, 0.1], [1.65, 0.05]])

    # cases 6 different shape
    # init_nodes = np.array([[0.0, 0.0], [0.3, 1.0],
    #    [1.0, -0.5], [1.5, 0.2]])

    # cases 7 different shape
    init_nodes = np.array([[0.0, 0.0], [0.3, -0.5],
                           [1.0, 1.0], [1.5, 0.2]])

    angle = 0  # 20 30 for cases 6, 0 30 for cases 7
    angle = np.deg2rad(angle)
    T = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    init_nodes = np.dot(T, init_nodes.transpose()).transpose()

    direction = 1
    diameter = 0.1
    init_setting = (init_nodes, init_pegs, direction, diameter)

    # 2. simulate unrolling process
    trajectory = simulate_unrolling(init_nodes, init_pegs, diameter, direction)
    origins, beziers, arcs, diameters, collide_pegs = trajectory

    # 3. plot the trajectory
    fig, ax = plt.subplots()
    plot_trajectory(init_setting, trajectory, ax=ax)
    # ax.plot(pivots[:, 0], pivots[:, 1], 'ro', markersize=2)
    plt.show()

    # 4. unrolling animation
    """
    num_pts = 400
    step = 2
    outfile = 'cases7/unrolling-0.mp4'
    # outfile = None
    film_writer_title = 'unrolling'
    unrolling_animation(init_setting, trajectory, num_pts, step, 40, outfile=outfile,
            film_writer_title=film_writer_title)
    """

    # 5. create pulling animation
    # 1. cases 1
    # angle 60
    # pivots = np.array([[-0.25, -0.2], [-0.15, 0.62], [-0.23, 0.648], [-0.5, 0.4]])
    # angle 26
    # pivots = np.array([[0, -0.2], [0.03, 0.38], [0.03, 0.40], [-0.11, 1.28],
    #    [-0.25, 1.4]])
    # angle 40
    # pivots = np.array([[0, -0.4], [-0.15, 0.62], [-0.23, 0.648]])

    # 2. cases 2 same length different shape
    # angle = 0
    # pivots = np.array([[-0.25, -0.1], [0.38, 0.38], [0.19, 0.49],
    #    [-0.15, 0.62], [-0.23, 0.648]])
    # angle = 30
    # pivots = np.array([[-0.25, 0.0], [0.03, 0.38], [0.03, 0.40],
    #    [-0.15, 0.62], [-0.23, 0.648]])
    # angle = 60
    # pivots = np.array([[-0.25, -0.1], [0.13, 1.0], [0, 0.88], [-0.23, 0.648]])

    # cases 3 different length
    # angle = 61 length = 0.8
    # pivots = np.array([[-0.25, 0.0], [-0.15, 0.62], [-0.23, 0.648], [-0.3, 0.6]])
    # angle = 62.5 length = 0.6
    # pivots = np.array([[0, -0.2], [-0.15, 0.62], [-0.23, 0.648]])
    # angle = 61 length = 0.4
    # outfile=None

    # cases 4 different shape
    # angle = 30
    # pivots = np.array([[-0.2, -0.2], [0.2, 0.15], [0.31, 0.20], [0.6, 0.4]])

    # cases 5 different shape
    # angle = 30
    # pivots = np.array([[-0.2, -0.2], [0.25, 0.35], [0.15, 0.69], [-0.02, 1.31]])

    # case 6 different shape
    # angle = 20
    # pivots = np.array([[-0.5, 0], [-0.15, 0.62], [-0.01, 1.13], [0.05, 1.27],
    #    [-0.06, 1.47]])
    # angle = 30
    # pivots = None

    # case 7 different shape
    # angle = 30
    # pivots = np.array([[-0.25, 0], [0.30, 0.03], [0.4, 0.1], [0.6, 0.4]])
    # angle = 0
    """
    pivots = np.array([[-0.25, 0], [0.30, 0.03], [0.4, 0.1],
        [0.47, 0.29], [0.15, 0.69], [0, 0.88]])
    outfile = 'cases7/pulling-0.mp4'
    # outfile = None
    film_writer_title = 'writer'
    pivots_file='cases7/pivots-0.txt'
    # pivots_file=None
    create_pulling_animation(init_setting, trajectory, pivots, outfile=outfile,
           film_writer_title=film_writer_title, pivots_file=pivots_file)
    # create_freepulling(init_setting, trajectory, outfile=outfile,
    #    film_writer_title='writer', num_ptr=400)
    """


if __name__ == "__main__":
    # case 2-1
    case = 2
    init_nodes = np.array([[0.0, 0.0], [0.3, 0.5],
        [1.0, 0.1], [1.58, 0.2]])
    angle = 0
    name = 'case-{:d}-{:d}'.format(case, angle)
    pivots = np.array([[-0.25, -0.1], [0.38, 0.38], [0.19, 0.49],
        [-0.15, 0.62], [-0.23, 0.648]])
    # 1. set initial condition
    angle = np.deg2rad(angle)
    T = np.array([[np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])
    init_nodes = np.dot(T, init_nodes.transpose()).transpose()

    init_pegs = np.array([[0.25, 0.35], [0.4, 0.1],
                          [0.47, 0.29], [0.38, 0.38],
                          [0.6, 0.4], [0.4, 0.6], [0.20, 0.7],
                          [0.20, 0.15], [0.05, 1.27],
                          [0.25, 0.90], [0.03, 0.40],
                          [0.58, 0.63], [0.15, 0.05], [0.31, 0.20],
                          [0.64, 0.23], [0.31, 0.93], [0.03, 0.38],
                          [0.30, 0.03], [0.56, 0.06], [0.13, 1.0],
                          [0.38, 0.74], [0.11, 1.28], [0.17, 1.09],
                          [-0.01, 1.13], [0.42, 0.49],
                          [0.00, 0.88], [-0.11, 1.28], [0.45, 0.47],
                          [0.12, 0.33], [-0.02, 1.31],
                          [0.15, 0.69], [0.14, 0.46],
                          [0.19, 0.49], [-0.23, 0.648],
                          [-0.15, 0.62],
                          ])

    direction = 1
    diameter = 0.1
    init_setting = (init_nodes, init_pegs, direction, diameter)

    # 2. simulate unrolling process
    trajectory = simulate_unrolling(init_nodes, init_pegs, diameter, direction)
    origins, beziers, arcs, diameters, collide_pegs = trajectory

    # 3. plot the trajectory
    # fig, ax = plt.subplots()
    # plot_trajectory(init_setting, trajectory, ax=ax)
    # plt.show()

    # 4. create unrolling animation
    num_pts = 400
    step = 2
    unrolling_outfile = name + '-unrolling.mp4'
    unrolling_film_writer_title = 'unrolling'
    unrolling_animation(init_setting, trajectory, num_pts, step, 40,
            outfile=unrolling_outfile,
            film_writer_title=unrolling_film_writer_title)

    # 5. create pulling animation
    pulling_film_writer_title = 'pulling'
    pulling_outfile = name + '-pulling.mp4'
    if pivots is not None:
        pivots_file = name + '-pivots.txt'
        create_pulling_animation(init_setting, trajectory, pivots,
                outfile=pulling_outfile, film_writer_title=
                pulling_film_writer_title, pivots_file=pivots_file)
    else:
        create_freepulling(init_setting, trajectory, outfile=pulling_outfile,
                film_writer_title='writer', num_ptr=400)

    # only use below section of code in OS X
    call(['open', unrolling_outfile])
    call(['open', pulling_outfile])
