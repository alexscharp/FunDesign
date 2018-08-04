#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################################
# University of Wisconsin-Madison
# Author: Alex Scharp, Yaqi Zhang
##################################
"""
This module contains code for phase-I
"""

# standard library
import sys
from itertools import zip_longest

# third party library
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.animation as manimation
import matplotlib.patches as patches
import seaborn as sns
import bezier
sns.set()


def _compute_arc(center, radius, theta1=0.0, theta2=np.pi * 2, num_pts=100):
    """return xs, ys of an arc"""
    cx, cy = center
    thetas = np.linspace(theta1, theta2, num_pts)
    return cx + radius * np.cos(thetas), cy + radius * np.sin(thetas)


def _is_peg_inside(center, diameter, peg):
    return np.hypot(*(peg - center)) <= (diameter / 2)


def _compute_diameter(max_diameter, ts):
    return max_diameter * (1 - ts)


def _compute_coil_center_diameter(curve, max_diameter, direction, t):
    """compute coil center and its diameter on parameter t"""
    nodes = curve.nodes.transpose()
    diameter = _compute_diameter(max_diameter, t)
    tangent = compute_tangents(nodes, t)
    tx, ty = tangent[0, 0], tangent[0, 1]
    nx, ny = -direction * ty, direction * tx
    point = curve.evaluate(t)
    x, y = point[0, 0], point[1, 0]
    cx, cy = x + 0.5 * diameter * nx, y + 0.5 * diameter * ny
    return np.array([cx, cy]), diameter


def detect_collision(pegs, curve, max_diameter, direction, num_pts,
                     collide_tolerance=0.001):
    tvals = np.linspace(0, 1, num_pts)
    points = curve.evaluate_multi(tvals).transpose()
    nodes = curve.nodes.transpose()
    diameters = _compute_diameter(max_diameter, tvals)
    tangents = compute_tangents(nodes, tvals)
    norms = np.zeros(tangents.shape)
    norms[:, 0], norms[:, 1] = -direction * \
        tangents[:, 1], direction * tangents[:, 0]
    centers = points + 0.5 * diameters.reshape((num_pts, 1)) * norms
    off_pnts = points + diameters.reshape((num_pts, 1)) * norms
    path_vertices = np.vstack((points, np.flipud(off_pnts)))
    path = Path(path_vertices, closed=True)
    cand_pegs = pegs[np.where(path.contains_points(pegs))]
    find = False
    # search in
    for idx, (center, diameter) in enumerate(zip(centers, diameters)):
        for peg in cand_pegs:
            if _is_peg_inside(center, diameter, peg):
                r_center, r_diameter = center, diameter
                r_idx, r_tval = idx, tvals[idx]
                collide_peg = peg
                find = True
                break
        if find:
            break
    # if find right circle, search left circle
    if find:
        # search the left circle
        assert r_idx > 0, 'The peg is inside initial coil!'
        l_idx = r_idx
        while l_idx >= 0:
            l_center, l_diameter = centers[l_idx], diameters[l_idx]
            if not _is_peg_inside(l_center, l_diameter, collide_peg):
                break
            l_idx -= 1
        l_tval = tvals[l_idx]
        # print(l_tval, r_tval)

        # use binary search to make better estimation of collide t
        steps = 0
        while l_tval < r_tval and steps < 100:
            steps += 1
            mid_tval = (l_tval + r_tval) / 2.0
            center, diameter = _compute_coil_center_diameter(curve,
                                                             max_diameter, direction, mid_tval)
            radius = diameter / 2.0
            dist = np.hypot(*(center - collide_peg))
            if abs(dist - radius) < collide_tolerance:
                collide_tval = mid_tval
                collide_center = center
                collide_diameter = diameter
                break
            elif dist > radius:  # outside
                l_tval = mid_tval
            else:
                r_tval = mid_tval
        else:
            collide_tval = (l_tval + r_tval) / 2.0
        return True, collide_tval, collide_center, collide_diameter, collide_peg

    else:
        # no collision if found
        return False, None, None, None, None
    # print(mid_tval)
    '''
    # plot curve and circles
    fig, ax = plt.subplots()
    patch = patches.PathPatch(path, alpha=0.5)
    # xs, ys = _compute_arc(r_center, r_diameter / 2.0)
    # ax.plot(xs, ys, 'g--')
    # xs, ys = _compute_arc(l_center, l_diameter / 2.0)
    # ax.plot(xs, ys, 'm--')
    xs, ys = _compute_arc(collide_center, collide_diameter / 2.0)
    ax.plot(xs, ys, 'b-')
    ax.plot(pegs[:, 0], pegs[:, 1], 'go', markersize=5)
    ax.plot(cand_pegs[:, 0], cand_pegs[:, 1], 'ro', markersize=5)
    ax.add_patch(patch)

    # curve and circles
    curve.plot(ax=ax, num_pts=128)
    ax.plot(centers[:, 0], centers[:, 1], 'bo')
    for center, diameter in zip(centers, diameters):
        xs, ys = _compute_arc(center, diameter/2.0, 0.0, np.pi * 2)
        ax.plot(xs, ys, 'g--')
    ax.axis('equal')
    plt.show()
    '''

def compute_bezier_points(nodes, num_pts):
    assert(nodes.shape[0] == 4)
    basis = (lambda t: (1 - t) ** 3, lambda t: 3 * t * (1 - t) ** 2,
            lambda t: 3 * t ** 2 * (1 - t), lambda t: t ** 3)
    ts = np.linspace(0, 1, num_pts)
    bs = np.array([func(ts) for func in basis]).transpose()
    points = np.dot(bs, nodes)
    return points


def plot_bezier(nodes, num_pts=128):
    points = compute_bezier_points(nodes, num_pts)
    fig, ax = plt.subplots()
    ax.plot(points[:, 0], points[:, 1], 'r--')
    # compare
    curve = bezier.Curve.from_nodes(nodes.transpose())
    curve.plot(ax=ax, num_pts=128)
    ax.axis('equal')
    plt.show()


def _normalize(v):
    """normalize a vector"""
    norm = np.linalg.norm(v, 2)
    assert(norm != 0)
    return v / norm


def compute_tangents(nodes, ts):
    assert(nodes.shape[0] == 4)
    basis = (lambda t: -3 * (1 - t) ** 2,
             lambda t: 3 * (1 - t) ** 2 - 6 * t * (1 - t),
             lambda t: -3 * t ** 2 + 6 * t * (1 - t),
             lambda t: 3 * t ** 2)
    bs = np.array([func(ts) for func in basis]).transpose()
    tangents = np.dot(bs, nodes)
    if len(tangents.shape) == 1:
        tangents = tangents[np.newaxis, :]
    tangents = np.apply_along_axis(_normalize, axis=1, arr=tangents)
    return tangents


def _compute_tangent_at_peg(center, diameter, peg, direction):
    normal = peg - center
    normal /= np.linalg.norm(normal, 2)
    tangent = -direction * normal[1], direction * normal[0]
    return np.array(tangent)


def _compute_angle(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    return np.arctan2(det, dot)


def handle_collision(curve, collide_tval, collide_center, collide_diameter,
                     collide_peg, pegs, direction):
    # 0. compute left pegs
    left_pegs = [row for row in pegs if not np.all(
            np.isclose(row, collide_peg))]
    left_pegs = np.array(left_pegs)
    # 1. compute arc
    mid_point = curve.evaluate(collide_tval).transpose().squeeze()
    cx, cy = collide_center
    theta1 = np.arctan2(mid_point[1] - cy, mid_point[0] - cx)
    theta2 = np.arctan2(collide_peg[1] - cy, collide_peg[0] - cx)
    if abs(theta2 - theta1) > np.pi / 2.0:
        # arc = (collide_center, collide_diameter, theta1, theta2)
        arc = None
    else:
        arc = (collide_center, collide_diameter, theta1, theta2)
    # 2. compute left curve and right curve
    # 2.1 compute left curve
    left_curve = curve.specialize(0.0, collide_tval)
    # 2.2 compute original right curve
    right_curve = curve.specialize(collide_tval, 1.0)
    left_nodes = left_curve.nodes.transpose()
    right_nodes = right_curve.nodes

    if arc is None: # pass through peg area
        # 2.3 compute transition matrix
        transition = np.array([[0.0], [0.0]])
        # 2.4 compute rotation origin and rotation angle
        rotation_origin = mid_point[:, np.newaxis]
        init_tangent = left_nodes[-1, :] - left_nodes[-2, :]
        init_tangent /= np.linalg.norm(init_tangent, 2)
        end_tangent = _compute_tangent_at_peg(
                collide_center, collide_diameter, collide_peg, -direction)
        rotation_angle = - _compute_angle(end_tangent, init_tangent)
    else:
        # 2.3 compute transition matrix
        transition = collide_peg - mid_point
        transition = transition[:, np.newaxis]
        # 2.4 compute rotation origin and rotation angle
        rotation_origin = collide_peg[:, np.newaxis]
        init_tangent = left_nodes[-1, :] - left_nodes[-2, :]
        init_tangent /= np.linalg.norm(init_tangent, 2)
        end_tangent = _compute_tangent_at_peg(
            collide_center, collide_diameter, collide_peg, direction)
        rotation_angle = - _compute_angle(end_tangent, init_tangent)
    # 2.5 transform
    # 2.5.1 do the transition first
    right_nodes += transition
    # 2.5.2 do rotation
    cosine, sine = np.cos(rotation_angle), np.sin(rotation_angle)
    T = np.array([[cosine, -sine], [sine, cosine]])
    right_nodes -= rotation_origin
    right_nodes = np.dot(T, right_nodes)
    right_nodes += rotation_origin

    new_right_curve = bezier.Curve.from_nodes(right_nodes)
    return left_curve, new_right_curve, arc, collide_diameter, left_pegs


def simulate_unrolling(init_nodes, init_pegs, diameter, direction):
    curve = bezier.Curve.from_nodes(init_nodes.transpose())
    pegs = init_pegs
    num_pts = 100
    tvals = np.linspace(0, 1, num=num_pts)
    beziers, arcs, diameters, collide_pegs = [], [], [], []
    while True:
        diameters.append(diameter)
        is_collide, collide_tval, collide_center, collide_diameter, collide_peg = detect_collision(
            pegs, curve, diameter, direction, num_pts)
        if is_collide:
            collide_pegs.append(collide_peg)
            left_curve, new_right_curve, arc, diameter, pegs = handle_collision(curve, collide_tval,
                                                                                collide_center, collide_diameter, collide_peg, pegs, direction)
            # arc_center, arc_diameter, theta1, theta2 = arc
            beziers.append(left_curve)
            arcs.append(arc)
            curve = new_right_curve
        else:
            beziers.append(curve)
            break
    assert(len(beziers) == len(diameters))
    assert(len(collide_pegs) == len(arcs))
    trajectory = (beziers, arcs, diameters, collide_pegs)
    return trajectory


def plot_trajectory(init_setting, trajectory, ax=None):
    # 1. unpack
    init_nodes, init_pegs, direction, diameter = init_setting
    beziers, arcs, diametes, collide_pegs = trajectory
    # 2. plot
    if not ax:
        fig = plt.subplot(111)
        ax = fig.axes
    for curve in beziers:
        curve.plot(ax=ax, num_pts=128, color='b')
    for arc in arcs:
        if arc:
            arc_center, arc_diameter, theta1, theta2 = arc
            xs, ys = _compute_arc(arc_center, arc_diameter / 2.0,
                                  theta1=theta1, theta2=theta2)
            ax.plot(xs, ys, 'b-')
    ax.plot(init_pegs[:, 0], init_pegs[:, 1], 'go', markersize=5)
    for x, y in collide_pegs:
        ax.plot(x, y, 'ro', markersize=5)
    ax.axis('equal')
    return ax


def animate_trajectory(init_setting, trajectory):
    # 1. unpack
    init_nodes, init_pegs, direction, diameter = init_setting
    beziers, arcs, diametes, collide_pegs = trajectory
    # add cycles
    length = sum(curve.length for curve in beziers)
    print(length)


def _compute_arc_perimeter(arc):
    _, diameter, theta1, theta2 = arc
    return diameter / 2.0 * abs(theta2 - theta1)


def create_movie_writer(title='Movie Writer', fps=15):
    """
    create ffmpeg writer

    Args:
        title: title of the movie writer
        fps: frames per second

    Returns:
        movie writer
    """
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title, artist='Matplotlib',
                    comment='Movie Support')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    return writer


def merge_paths(init_setting, trajectory, num_pts):
    """merge subpaths in trajectory into one path represented by
       num_pts points (num_pts X 5) each row is represented as
       [x, y, cx, cy, radius]
    """
    _, _, direction, _ = init_setting
    beziers, arcs, diameters, _ = trajectory
    # add cycles
    bezier_lens = [curve.length for curve in beziers]
    arc_lens = [_compute_arc_perimeter(arc) for arc in arcs if arc is not None]
    bezier_len = sum(bezier_lens)
    arc_len = sum(arc_lens)
    total_len = bezier_len + arc_len
    total_pts = num_pts
    bezier_pts = [int(np.around(length / total_len * total_pts))
                  for length in bezier_lens]
    arc_pts = [int(np.around(length / total_len * total_pts))
               for length in arc_lens]
    total_pts = sum(bezier_pts) + sum(arc_pts)
    # print(total_pts)
    # [x, y, cx, cy, radius]
    # compute arc points
    arc_points = []
    for idx, (arc, pts) in enumerate(zip(arcs, arc_pts)):
        if arc is not None:
            (cx, cy), diameter, theta1, theta2 = arc
            radius = diameter / 2.0
            xs, ys = _compute_arc((cx, cy), radius, theta1, theta2, 
                    num_pts=pts)
            temp = np.zeros((pts, 5))
            for i in range(pts):
                temp[i, :] = np.array([xs[i], ys[i], cx, cy, radius])
            arc_points.append(temp)
        else:
            arc_points.append(None)
    bezier_points = []
    for idx, (curve, pts) in enumerate(zip(beziers, bezier_pts)):
        tvals = np.linspace(0.0, 1.0, pts)
        temp = np.zeros((pts, 5))
        max_diameter = diameters[idx]
        min_diameter = diameters[idx + 1] if idx < len(diameters) - 1 else 0.0
        points = curve.evaluate_multi(tvals).transpose()
        radiuses = 0.5 * ((1 - tvals) * max_diameter + tvals * min_diameter)
        nodes = curve.nodes.transpose()
        tangents = compute_tangents(nodes, tvals)
        norms = np.zeros(tangents.shape)
        norms[:, 0], norms[:, 1] = -direction * \
            tangents[:, 1], direction * tangents[:, 0]
        centers = points + radiuses.reshape((pts, 1)) * norms
        temp[:, 0:2] = points
        temp[:, 2:4] = centers
        temp[:, 4] = radiuses
        bezier_points.append(temp)

    # combine them together
    points = []
    for bps, aps in zip_longest(bezier_points, arc_points, fillvalue=None):
        if bps is not None:
            points.append(bps)
        if aps is not None:
            points.append(aps)
    # assert(len(points) == len(bezier_points) + len(arc_points))
    points = np.vstack(points)
    assert(points.shape[1] == 5)
    # print(points.shape)
    return points


def create_animation(init_setting, trajectory, num_pts, step, outfile=None,
                     film_writer_title='writer'):
    _, init_pegs, _, _ = init_setting
    points = merge_paths(init_setting, trajectory, num_pts=num_pts)
    total_pts = points.shape[0]
    # plot
    path, circle = None, None
    fig, ax = plt.subplots()
    # add here for phase I example
    ax.set_xlim([-1.2, 0.8])
    ax.set_ylim([-0.4, 1.0])
    # plt.xticks([])
    # plt.yticks([])
    ax.plot(init_pegs[:, 0], init_pegs[:, 1], 'go')
    if outfile:
        writer = create_movie_writer(title=film_writer_title, fps=10)
        writer.setup(fig, outfile=outfile, dpi=100)
    for idx in range(1, total_pts, step):
        if path is not None:
            path.remove()
        if circle is not None:
            circle.remove()
        path, = ax.plot(points[:idx, 0], points[:idx, 1], 'b-')
        cx, cy, radius = points[idx - 1, 2:]
        if radius > 0:
            xs, ys = _compute_arc((cx, cy), radius, 0.0, np.pi * 2, 20)
            circle, = ax.plot(xs, ys, 'b-')
            if outfile:
                writer.grab_frame()
        plt.pause(0.1)
        plt.draw()
    if outfile:
        writer.finish()
        print('Creating movie {:s}'.format(outfile))
    plt.show()


if __name__ == "__main__":
    # 1. set parameters
    '''
    init_pegs = np.array([[0.25, 0.38], [0.4, 0.1],
                          [0.6, 0.4], [0.0, 0.6], [0.4, 0.6], [0.2, 0.8]])
    '''
    # [0.25, 0.38] --> [0.25, 0.35]; [0.0, 0.6] --> [0.05, 0.6]
    init_pegs = np.array([[0.25, 0.35], [0.4, 0.1],
                          [0.6, 0.4], [0.05, 0.6], [0.4, 0.6], [0.22, 0.7]])
    init_nodes = np.array([[0.0, 0.0], [0.3, 0.5],
                           [1.0, 0.5], [1.5, 0.2]])
    direction = 1
    diameter = 0.2
    init_setting = (init_nodes, init_pegs, direction, diameter)

    # 2. simulate unrolling process
    trajectory = simulate_unrolling(init_nodes, init_pegs, diameter, direction)
    beziers, arcs, diameters, collide_pegs = trajectory
    for arc in arcs:
        if arc:
            center, diameter, theta1, theta2 = arc
            print(np.rad2deg(theta2 - theta1))
    print(collide_pegs)

    # 3. plot the trajectory
    fig, ax = plt.subplots()
    plot_trajectory(init_setting, trajectory, ax=ax)
    plt.show()

    # animation example
    # num_pts = 400
    # step = 2
    # outfile = None
    # # outfile = 'unrolling.mp4'
    # film_writer_title = 'unrolling'
    # create_animation(init_setting, trajectory, num_pts, step, outfile=outfile,
    #        film_writer_title=film_writer_title)
