import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import random, seed
from math import pi, sin, cos, sqrt
import matplotlib.pyplot as plt

"""Generuojamos figūros plokštumoje. Iš taško x0, y0 reikia rasti kelią į x1, y1. Sukioti figūras taip, kad kelias būtų trumpiausias."""

# rotation_matrixes = np.array((2, 2, 2))



pi_2 = pi / 2

MINX = MINY = 0
MAXX = MAXY = 1
DEFAULT_SIDE = 0.1
DEFAULT_SAFETY_MARGIN = DEFAULT_SIDE * sqrt(2)
MAX_SQUARES = 20

__global_generation_counter = 0


def get_func_deg1(p0, p1):
    (x0, y0), (x1, y1) = p0, p1
    if x0 == x1:
        return None
    a = (y0 - y1)/(x0 - x1)
    b = y0 - x0 * a
    return lambda x: a * x + b


def is_point_in_square(p, sq):
    x, y = p
    p0, p1, p2, p3 = sq
    side_func0 = get_func_deg1(p0, p1)
    side_func1 = get_func_deg1(p1, p2)
    side_func2 = get_func_deg1(p2, p3)
    side_func3 = get_func_deg1(p3, p0)
    if not side_func0 or not side_func1 or not side_func2 or not side_func3:
        xmin = min(p0[0], p2[0])
        xmax = max(p0[0], p2[0])
        ymin = min(p0[1], p2[1])
        ymax = max(p0[1], p2[1])
        return xmin <= x <= xmax and ymin <= y <= ymax
    return ((y - side_func0(x)) * (y - side_func2(x))) <= 0 and \
           ((y - side_func1(x)) * (y - side_func3(x))) <= 0


def squares_overlap(square0, square1):
    for p0 in square0:
        if is_point_in_square(p0, square1):
            return True
    for p1 in square1:
        if is_point_in_square(p1, square0):
            return True
    xc0 = (square0[0][0] + square0[2][0]) / 2
    yc0 = (square0[0][1] + square0[2][1]) / 2
    if is_point_in_square((xc0, yc0), square1):
        return True
    # The "reverse center check" not needed, since squares are congruent
    """
    xc1 = (square1[0][0] + square1[2][0]) / 2
    yc1 = (square1[0][1] + square1[2][1]) / 2
    if is_point_in_square((xc1, yc1), square0):
        return True
    """
    return False


def __generation_monitor():
    global __global_generation_counter
    __global_generation_counter += 1


def generate_random_point(minx=MINX, miny=MINY, maxx=MAXX, maxy=MAXY, safety_margin=DEFAULT_SAFETY_MARGIN):
    if maxx - minx < 2 * safety_margin or maxy - miny < 2 * safety_margin:
        print("MUEEE")
        safety_margin = 0
    x = safety_margin + random() * (maxx - minx - 2 * safety_margin)
    y = safety_margin + random() * (maxy - miny - 2 * safety_margin)
    __generation_monitor()
    return x, y


def generate_random_angle(max_val=pi_2):
    return random() * max_val


def generate_random_square(side=DEFAULT_SIDE, squares_to_avoid=()):
    while 1:
        restart = False
        x0, y0 = generate_random_point()

        angle = generate_random_angle()
        x1 = x0 + side * cos(angle)
        y1 = y0 + side * sin(angle)

        angle += pi_2
        x2 = x1 + side * cos(angle)
        y2 = y1 + side * sin(angle)

        angle += pi_2
        x3 = x2 + side * cos(angle)
        y3 = y2 + side * sin(angle)

        ret = (x0, y0), (x1, y1), (x2, y2), (x3, y3)
        for square in squares_to_avoid:
            if squares_overlap(ret, square):
                restart = True
        if restart:
            continue
        return ret


def square_to_plot(square):
    xs, ys = zip(square[0], square[1], square[2], square[3])
    return xs + (xs[0],), ys + (ys[0],)


def get_square_center(square):
    sq = np.array(square)
    x = np.average(sq[:, 0])
    y = np.average(sq[:, 1])
    return x, y


def is_point_in_squares(p, squares):
    for sq in squares:
        if is_point_in_square(p, sq):
            return sq
    return None


def slope(p1, p2):
    return (p2[1] - p1[1]) * 1. / (p2[0] - p1[0])


def y_intercept(slope, p1):
    return p1[1] - 1. * slope * p1[0]


def intersect(line1, line2):
    min_allowed = 1e-5  # guard against overflow
    big_value = 1e10  # use instead (if overflow would have occurred)
    m1 = slope(line1[0], line1[1])
    # print('m1: %d' % m1)
    b1 = y_intercept(m1, line1[0])
    # print('b1: %d' % b1)
    m2 = slope(line2[0], line2[1])
    # print('m2: %d' % m2)
    b2 = y_intercept(m2, line2[0])
    # print('b2: %d' % b2)
    if abs(m1 - m2) < min_allowed:
        x = big_value
    else:
        x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    y2 = m2 * x + b2
    # print('(x,y,y2) = %d,%d,%d' % (x, y, y2))
    return (x, y)


def segment_intersect(line1, line2):
    intersection_pt = intersect(line1, line2)

    # print(line1[0][0], line1[1][0], line2[0][0], line2[1][0], intersection_pt[0])
    # print(line1[0][1], line1[1][1], line2[0][1], line2[1][1], intersection_pt[1])

    if (line1[0][0] < line1[1][0]):
        if intersection_pt[0] < line1[0][0] or intersection_pt[0] > line1[1][0]:
            # print('exit 1')
            return None
    else:
        if intersection_pt[0] > line1[0][0] or intersection_pt[0] < line1[1][0]:
            # print('exit 2')
            return None

    if (line2[0][0] < line2[1][0]):
        if intersection_pt[0] < line2[0][0] or intersection_pt[0] > line2[1][0]:
            # print('exit 3')
            return None
    else:
        if intersection_pt[0] > line2[0][0] or intersection_pt[0] < line2[1][0]:
            # print('exit 4')
            return None

    return intersection_pt


def get_square_trace_vector(current_point, next_point, direction_v, square):
    walls = list()
    walls.append((square[0], square[1]))
    walls.append((square[1], square[2]))
    walls.append((square[2], square[3]))
    walls.append((square[3], square[0]))

    current_step = (current_point, next_point)
    for i in range(len(walls)):
        l = walls[i]
        intersect_point = segment_intersect(l, current_step)
        if intersect_point is not None:
            # print(i)
            direction1 = np.array(intersect_point) - np.array(square[(i + 1) % 4])
            direction2 = np.array(intersect_point) - np.array(square[i])
            if not np.sqrt(np.sum((direction1 - direction_v) ** 2)) < np.sqrt(np.sum((direction2 - direction_v) ** 2)):
                return direction1, intersect_point, np.array(square[(i + 1) % 4])
            else:
                return direction2, intersect_point, np.array(square[i])
    return direction_v, None, None


def find_path(squares, x0, y0, x1, y1, step_size=0.001):
    destination_reached = False
    path_length = 0
    current_point = np.array([x0, y0])
    destination_point = np.array([x1, y1])
    path = list()
    path.append(current_point)

    while not destination_reached:

        direction_v = destination_point - current_point
        # vector normalisation
        direction_v *= (1.0 / float(np.max(direction_v)))
        next_point = current_point + direction_v * step_size
        s_intersected = is_point_in_squares((next_point[0], next_point[1]), squares)
        if s_intersected is not None:
            # next_point = np.array([current_point[0] + (random() - 0.5) * step_size, current_point[1] + (random() - 0.5) * step_size])
            square_wall_trace_v, intersect_point, wall_end = get_square_trace_vector(current_point, next_point, direction_v, s_intersected)
            # next_point -= current_point + direction_v * step_size
            if intersect_point is not None:
                next_point = intersect_point #- direction_v * step_size / 100
                path_length += np.sqrt(np.sum((next_point - current_point) ** 2))
                path.append(next_point)
                current_point = next_point
                next_point = wall_end - square_wall_trace_v * 0.00001
            else:
                next_point = current_point + square_wall_trace_v * step_size
            # s_intersected = is_point_in_squares((next_point[0], next_point[1]), squares)


        path_length += np.sqrt(np.sum((next_point - current_point) ** 2))
        # print(path_length)
        path.append(next_point)
        current_point = next_point
        if abs(np.sum(current_point - destination_point)) < step_size * 3:
            destination_reached = True
    return path_length, path


def get_rotation_matrix(degrees):
    rotation = np.array([[np.cos(pi * degrees / 180), np.sin(pi * degrees / 180)], [- np.sin(pi * degrees / 180), np.cos(pi * degrees / 180)]])
    return rotation


def get_partial_derivative(squares, x0, y0, x1, y1, explore_step=0.001, learning_step = 0.1):
    partial_derivative = np.zeros(len(squares))
    initial_l, pathy = find_path(squares, x0, y0, x1, y1, explore_step)
    for i in range(len(squares)):
        new_squares = list(squares)
        new_squares[i] = rotate_square(new_squares[i], 0.05)
        new_l, _ = find_path(new_squares, x0, y0, x1, y1, explore_step)
        partial_derivative[i] = (initial_l ** 5 - new_l ** 5) * 100
    print(partial_derivative)
    return partial_derivative, initial_l, pathy


def apply_step(squares, partial_derivative, learning_stepsize = 0.1):
    for i in range(len(squares)):
        squares[i] = rotate_square(squares[i], partial_derivative[i] * learning_stepsize)


def rotate_square(square, degrees, rotation_matrix=None):
    sq = np.array(square)
    center = get_square_center(sq)
    if rotation_matrix is None:
        rotation_matrix = get_rotation_matrix(degrees)
    sq_rotated = np.dot(sq - center, rotation_matrix) + center
    sq_rotated = (sq_rotated[0][0], sq_rotated[0][1]), (sq_rotated[1][0], sq_rotated[1][1]),\
                 (sq_rotated[2][0], sq_rotated[2][1]), (sq_rotated[3][0], sq_rotated[3][1])
    return sq_rotated


if __name__ == "__main__":
    # print(get_rotation_matrix(0.001))
    # print(get_rotation_matrix(-0.001))
    seed()
    squares = list()
    allow_overlapping = False  # CHANGE to True to allow square to overlap
    for _ in range(MAX_SQUARES):
        if allow_overlapping:
            square = generate_random_square()
        else:
            square = generate_random_square(squares_to_avoid=squares)
        # square = rotate_square(square, 1)
        squares.append(square)

    max_steps = 500
    explore_step = 0.01
    learning_step = 0.3
    history_squares = list()
    history_paths = list()
    history_step = 1
    prev_path_l = 100

    plot_squares = tuple()
    for sq in squares:
        plot_squares += square_to_plot(sq)
    # print("STATS:\n    Squares: {}\n    Allow  overlapping: {}\n    Generated values: {}".format(MAX_SQUARES,
    #                                                                                              allow_overlapping,
    #                                                                                              __global_generation_counter))
    plt.plot(*plot_squares)
    path_l, path = find_path(squares, 0.0, 0.0, 1.0, 1.0)
    print(path_l)
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1])
    plt.show()

    for i in range(max_steps):
        derivatives, path_l, path = get_partial_derivative(squares, 0.0, 0.0, 1.0, 1.0, explore_step=explore_step, learning_step=learning_step)
        print(path_l)
        if i % history_step == 0:
            history_squares.append(squares.copy())
            history_paths.append(path.copy())
        # if path_l >= prev_path_l:
        #     history_squares.append(squares.copy())
        #     history_paths.append(path.copy())
        #     break
        # else:
        #     prev_path_l = path_l
        apply_step(squares, derivatives, learning_step)


    # plot_squares = tuple()
    # for sq in squares:
    #     plot_squares += square_to_plot(sq)
    # print("STATS:\n    Squares: {}\n    Allow  overlapping: {}\n    Generated values: {}".format(MAX_SQUARES,
    #                                                                                              allow_overlapping,
    #                                                                                              __global_generation_counter))
    # plt.plot(*plot_squares)
    # plt.axis([MINX, MAXX, MINY, MAXY])
    # plt.show()

    # # rotated_squares = []
    # # for sq in squares:
    # #     rotated_squares.append(rotate_square(sq, 25))
    # # squares = rotated_squares
    # plot_squares = tuple()
    # for sq in squares:
    #     plot_squares += square_to_plot(sq)
    # # print("STATS:\n    Squares: {}\n    Allow  overlapping: {}\n    Generated values: {}".format(MAX_SQUARES,
    # #                                                                                              allow_overlapping,
    # #                                                                                              __global_generation_counter))
    # plt.plot(*plot_squares)
    # path_l, path = find_path(squares, 0.0, 0.0, 1.0, 1.0)
    # print(path_l)
    # path = np.array(path)
    # plt.plot(path[:, 0], path[:, 1])
    # plt.show()
    # fig = plt.figure()
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    fig = plt.figure()
    # plt.axis([MINX, MAXX, MINY, MAXY])
    ax = plt.axes(xlim=(MINX, MAXX), ylim=(MINY, MAXY))
    graphog, = ax.plot([], [])
    graph_squares = ax.plot( *([[], []]*MAX_SQUARES))
    print(len(history_paths))

    def animate(i):
        path = np.array(history_paths[i])
        path = np.array(path)
        graphog.set_data(path[:, 0], path[:, 1])
        plot_squares = tuple()
        squares = history_squares[i]
        for i in range(len(squares)):
            sq = squares[i]
            sq_n = np.array(sq)
            # plot_squares += square_to_plot(sq)
            graph_squares[i].set_data(np.append(sq_n[:, 0], sq_n[0, 0]), np.append(sq_n[:, 1], sq_n[0, 1]))
        return [graph_squares, graphog]


    ani = FuncAnimation(fig, animate, frames=len(history_paths), interval=10, repeat_delay=1000)
    plt.show()

    # n = 14
    # m = 14
    # k = 4
    # step_size = 0.01
    # steps = 150
    #
    # old_cor = generate_coordinates(n)
    # new_cor = generate_coordinates(m)
    # all_cor = np.append(old_cor, new_cor, axis=0)
    # history = []
    # history_iters = []
    # history.append(np.array(all_cor))
    # # Gradient descent
    # print(all_cor)
    # for i in range(steps):
    #     # print(all_cor)
    #     derivatives = get_derivatives(all_cor, m, k)
    #     print("DD", derivatives)
    #     dtd = distance_to_ideal(all_cor, m, n)
    #     history_iters.append(dtd)
    #     print("Distance to ideal: ", dtd)
    #     step(all_cor, derivatives, step_size)
    #     history.append(np.array(all_cor))
    #
    # print(all_cor)
    # history = np.array(history)
    # plt.plot(history_iters)
    # plt.show()
    # fig = plt.figure()
    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)
    # graphog, = plt.plot([], [], 'xb')
    # graphnew, = plt.plot([], [], 'or')
    # print(history.shape)
    #
    # def animate(i):
    #     graphog.set_data(history[i, :-m, 0], history[i, :-m, 1])
    #     graphnew.set_data(history[i, -m:, 0], history[i, -m:, 1])
    #     return graphog
    #
    #
    # ani = FuncAnimation(fig, animate, frames=steps, interval=20)
    # plt.show()


