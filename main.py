import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import random, seed
from math import pi, sin, cos, sqrt

"""Generuojamos figūros plokštumoje. Iš taško x0, y0 reikia rasti kelią į x1, y1. Sukioti figūras taip, kad kelias būtų trumpiausias."""

pi_2 = pi / 2

MINX = MINY = 0
MAXX = MAXY = 1
DEFAULT_SIDE = 0.1
DEFAULT_SAFETY_MARGIN = DEFAULT_SIDE * sqrt(2)
MAX_SQUARES = 10

__global_generation_counter = 0


def get_func_deg1(p0, p1):
    (x0, y0), (x1, y1) = p0, p1
    if x0 == x1:
        return None
    a = (y0 - y1) / (x0 - x1)
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

    return np.array(intersection_pt)


def get_square_trace_vector(current_point, next_point, squares):
    intersected_walls = list()
    intersected_points = list()
    # holds closest intersected point id and point
    closest_intersected = list()
    for square_np in squares:
        walls = np.zeros((4, 2, 2), dtype=np.float64)
        for i in range(4):
            walls[i, :, :] = np.array([square_np[i, :], square_np[(i + 1) % 4, :]])
        current_step = (current_point, next_point)
        for i in range(len(walls)):
            wall = walls[i]
            intersect_point = segment_intersect(wall, current_step)
            if intersect_point is not None:
                intersected_walls.append(wall)
                intersected_points.append(intersect_point)
    for i, point in enumerate(intersected_points):
        path_len = np.sqrt(np.sum((point - current_point) ** 2))
        closest_intersected.append(np.array([i, path_len]))
    closest_intersected.sort(key=lambda x: x[1])
    if len(closest_intersected) > 0:
        index = int(closest_intersected[0][0])
        len1 = np.sqrt(np.sum((intersected_walls[index][0] - current_point) ** 2))
        len2 = np.sqrt(np.sum((intersected_walls[index][1] - current_point) ** 2))
        if abs(len1) < 1e-9 or abs(len2) < 1e-9:
            index = int(closest_intersected[1][0])
        new_wall = np.array(intersected_walls[index])
        direction0 = intersected_points[index] - new_wall[0]
        direction1 = intersected_points[index] - new_wall[1]
        new_wall[0] -= direction0 * 1e-7
        new_wall[1] -= direction1 * 1e-7
        return intersected_points[index], new_wall
    else:
        return next_point, None


def find_path(squares, x0, y0, x1, y1, path, step_size=0.001, path_length=0, rec_d=0):
    if x0 == x1 and y0 == y1:
        return path_length, np.array(path)
    current_point = np.array([x0, y0])
    destination_point = np.array([x1, y1])
    path.append(current_point)
    intersected_point, wall = get_square_trace_vector(current_point, destination_point, squares)

    if wall is not None:
        path_len_leftest = np.sqrt(np.sum((wall[0] - current_point) ** 2)) + path_length
        path_len_rightest = np.sqrt(np.sum((wall[1] - current_point) ** 2)) + path_length
        if rec_d > 2:
            left_rec_points = np.sqrt(np.sum((wall[1] - destination_point) ** 2))
            right_rec_points = np.sqrt(np.sum((wall[0] - destination_point) ** 2))
            if left_rec_points < right_rec_points:
                path_len_right, path_right = None, None
                path_len_left, path_left = find_path(squares, wall[1, 0], wall[1, 1], destination_point[0],
                                                     destination_point[1], path.copy(), path_length=path_len_rightest,
                                                     rec_d=rec_d + 1)
            else:
                path_len_right, path_right = find_path(squares, wall[0, 0], wall[0, 1], destination_point[0],
                                                       destination_point[1], path.copy(), path_length=path_len_leftest,
                                                       rec_d=rec_d + 1)
                path_len_left, path_left = None, None
        else:
            path_len_right, path_right = find_path(squares, wall[0, 0], wall[0, 1], destination_point[0],
                                                   destination_point[1], path.copy(), path_length=path_len_leftest,
                                                   rec_d=rec_d + 1)
            path_len_left, path_left = find_path(squares, wall[1, 0], wall[1, 1], destination_point[0],
                                                 destination_point[1], path.copy(), path_length=path_len_rightest,
                                                 rec_d=rec_d + 1)

        if path_len_left is not None and path_len_right is not None:
            if path_len_left < path_len_right:
                return path_len_left, path_left
            else:
                return path_len_right, path_right
        elif path_len_left is not None and path_len_right is None:
            return path_len_left, path_left
        elif path_len_left is None and path_len_right is not None:
            return path_len_right, path_right
        else:
            return None, None
    path.append(destination_point)
    path_length += np.sqrt(np.sum((destination_point - current_point) ** 2))
    return path_length, path


def get_rotation_matrix(degrees):
    rotation = np.array([[np.cos(pi * degrees / 180), np.sin(pi * degrees / 180)],
                         [- np.sin(pi * degrees / 180), np.cos(pi * degrees / 180)]])
    return rotation


def get_partial_derivative(squares, x0, y0, x1, y1, explore_step=0.001, learning_step=0.1):
    partial_derivative = np.zeros(len(squares))
    initial_l, pathy = find_path(squares, x0, y0, x1, y1, list())
    for i in range(len(squares)):
        new_squares = np.array(squares)
        new_squares[i] = rotate_square(new_squares[i], 0.05)
        new_l, _ = find_path(new_squares, x0, y0, x1, y1, list())
        partial_derivative[i] = (initial_l ** 5 - new_l ** 5) * 100
    print(partial_derivative)
    return partial_derivative, initial_l, pathy


def apply_step(squares, partial_derivative, learning_stepsize=0.1):
    for i in range(len(squares)):
        d = partial_derivative[i]
        if abs(d) > 0.0:
            squares[i] = rotate_square(squares[i], d * learning_stepsize)


def rotate_square(square, degrees, rotation_matrix=None):
    sq = np.array(square)
    center = get_square_center(sq)
    if rotation_matrix is None:
        rotation_matrix = get_rotation_matrix(degrees)
    sq_rotated = np.dot(sq - center, rotation_matrix) + center
    sq_rotated = (sq_rotated[0][0], sq_rotated[0][1]), (sq_rotated[1][0], sq_rotated[1][1]), \
                 (sq_rotated[2][0], sq_rotated[2][1]), (sq_rotated[3][0], sq_rotated[3][1])
    return sq_rotated


def main():
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
    squares = np.array(squares)
    max_steps = 300
    explore_step = 0.01
    learning_step = 300
    history_squares = list()
    history_paths = list()
    history_step = 1

    plot_squares = tuple()
    for sq in squares:
        plot_squares += square_to_plot(sq)
    # print("STATS:\n    Squares: {}\n    Allow  overlapping: {}\n    Generated values: {}".format(MAX_SQUARES,
    #                                                                                              allow_overlapping,
    #                                                                                              __global_generation_counter))
    plt.plot(*plot_squares)
    path_l, path = find_path(squares, 0.0, 0.0, 1.0, 1.0, list())
    print(path_l)
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1])
    plt.show()
    history_squares.append(squares.copy())
    history_paths.append(path.copy())
    for i in range(max_steps):
        derivatives, path_l, path = get_partial_derivative(squares, 0.0, 0.0, 1.0, 1.0, explore_step=explore_step,
                                                           learning_step=learning_step)
        if np.sum(np.abs(derivatives)) < 1e-3:
            break
        print(path_l)
        if i % history_step == 0:
            history_squares.append(squares.copy())
            history_paths.append(path.copy())
        apply_step(squares, derivatives, learning_step)

    fig = plt.figure()
    # plt.axis([MINX, MAXX, MINY, MAXY])
    ax = plt.axes(xlim=(MINX, MAXX), ylim=(MINY, MAXY))
    graphog, = ax.plot([], [])
    graph_squares = ax.plot(*([[], []] * MAX_SQUARES))
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

    ani = FuncAnimation(fig, animate, frames=len(history_paths), interval=40, repeat_delay=1000)
    plt.show()


if __name__ == "__main__":
    main()
