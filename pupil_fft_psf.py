# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 12:35:08 2022

@author: Douglas Harvey
"""



import numpy as np
import numpy.fft as npft



def round_even(n, up_down = -1):
    """
    Round the integer n up or down to the nearest even integer.

    Parameters
    ----------
    n : int
        An integer to be rounded up or down.
    up_down : int (-1, 1), optional (default is -1) 
        1 is round up, -1 is round down.

    Returns
    -------
    int
        The nearest even integer to n.

    """
    if n % 2:
        return n + up_down
    else:
        return n


def round_odd(n, up_down = -1):
    """
    Round the integer n up or down to the nearest odd integer.

    Parameters
    ----------
    n : int
        An integer to be rounded up or down.
    up_down : int (-1, 1), optional (default is -1) 
        1 is round up, -1 is round down.

    Returns
    -------
    int
        The nearest odd integer to n.

    """
    if n % 1:
        return n + up_down
    else:
        return n


def aperture_arr3d(diameter_paras, size = 64, constant_diameter = False):
    """
    Generate an array of circles with diameters varying according to
    diameter_paras.
    
    Parameters
    ----------
    diameter_paras : iterable of floats, shape (3), (lower, upper, steps)
        A 3 element iterable containing lower, the diameter at output[0];
        upper, the diameter at output[steps]; steps, the number of intermediate
        diameters between lower and upper, defines the size of output in
        axis 0.
    size : int, optional (default is 64)
         The size of output axes 1 and 2.
    constant_diameter : bool, optional (default is False)
        If True the circle diameter will not vary along output axis 0 but will
        always be equal to upper.

    Returns
    -------
    output : numpy ndarray of ints, shape (steps, size, size)
        A (steps, size, size) array where 2d slices along axis 0 are subarrays
        of 0s with a centred circle of 1s. Circle diameter varies along axis 0
        according to diameter_paras.

    """
    lower, upper, steps = diameter_paras
    y_size = x_size = size
    y_centre = (y_size - 1)/2
    x_centre = (x_size - 1)/2
    # Define yy and xx arrays with shape (y_size, x_size) with each row in yy
    # or column in xx having a constant value ~= np.arange(size) at its index.
    yy, xx = np.mgrid[:y_size, :x_size]
    # Define a (y_size, x_size) array with elements equal to the distance
    # between their 2d index and (y_centre, x_centre).
    circle_distance = np.sqrt((yy - y_centre) ** 2 + (xx - x_centre) ** 2)
    def circle_binary(diameter):
        """
        Generate a binary circle array from the circle_distance array.

        Parameters
        ----------
        diameter : float
            Diameter within which elements are set to 1.

        Returns
        -------
        circle_binary : numpy ndarray of ints, shape (y_size, x_size)
            A (y_size, x_size) array of 0s with a circle of 1s at its centre.

        """
        circle_binary = np.zeros((y_size, x_size))
        circle_binary[(circle_distance <= diameter/2)] = 1
        return circle_binary
    # If diameter is not constant stack steps number of binary circle arrays
    # with diameters varying according to np.linspace((lower, upper, steps))
    # along output axis 0.
    if not constant_diameter:
        output =  np.stack([circle_binary(diameter) for diameter in np.linspace(lower, upper, steps)])
    # If diameter is constant generate one binary circle array with
    # diameter = upper and stack it steps number of times along output axis 0.
    elif constant_diameter:
        constant_circle = circle_binary(upper)
        output = np.stack([constant_circle for _ in np.arange(steps)])
    return output


def grid_arr3d(spacing_paras, size = 64, centering = "intersection", constant_spacing = False):
    """
    Generate an array of grid patterns with spacing varying according to
    spacing_paras.

    Parameters
    ----------
    spacing_paras : iterable of floats, shape (3), (lower, upper, steps)
        A 3 element iterable containing lower, the grid spacing at output[0];
        upper, the grid spacing at output[steps]; steps, the number of
        intermediate grid spacings between lower and upper, defines the size
        of output in axis 0.
    size : int, optional (default is 64)
         The size of output axes 1 and 2.
    centering : string, optional (default is "intersection")
        If "intersection" a point where 2 grid lines meet is kept at the centre
        of the subarrays along axis 0. If "centre" the centre of a square
        formed by 4 grid lines is kept there. If neither the top left corner
        of the grid is kept at (0, 0).
    constant_spacing : bool, optional (default is False)
        If True the grid spacing will not vary along output axis 0 but will
        always be equal to upper.

    Returns
    -------
    numpy ndarray of ints, shape (steps, size, size)
        A (steps, size, size) array where 2d slices along axis 0 are subarrays
        of 0s with a grid pattern of 1s. Grid spacing varies along axis 0
        according to spacing_paras.

    """
    lower, upper, steps = spacing_paras
    y_size = x_size = size
    # Define yy and xx arrays with shape (y_size, x_size) with each row in yy
    # or column in xx having a constant value ~= np.arange(size) at its index.
    yy, xx = np.mgrid[:y_size, :x_size]
    def grid(spacing):
        """
        Generate a grid pattern array.

        Parameters
        ----------
        spacing : float
            The spacing between grid lines, containing spacing - 1 0s followed
            by 1 1.

        Returns
        -------
        grid : numpy ndarray of ints, shape (y_size, x_size)
            A (y_size, x_size) array of 0s with a grid pattern of 1s.

        """
        y_spacing = x_spacing = spacing
        y_indices = np.arange(0, y_size, y_spacing)
        x_indices = np.arange(0, x_size, x_spacing)
        grid = np.zeros((y_size, x_size))
        # Check if centering is specified at all.
        if centering in ["intersection", "centre"]:
            # If using "intersection" centering, calculate the number of
            # complete cells formed by the grid lines (size//spacing) and round
            # this down to the nearest even int. 
            if centering == "intersection":
                y_cells_rounded = round_even(y_size//y_spacing)
                x_cells_rounded = round_even(x_size//x_spacing)
            # If using "centre" centering, do the same but round to the nearest
            # odd int.
            elif centering == "centre":
                y_cells_rounded = round_odd(y_size//y_spacing)
                x_cells_rounded = round_odd(x_size//x_spacing)
            # Calculate the size of the block of cells.
            y_grid_size = y_cells_rounded*y_spacing
            x_grid_size = x_cells_rounded*x_spacing
            # Find the difference between the centre coordinate of the whole
            # array and the centre coordinate of the block of cells. If number
            # of cells is even this coordinate is an intersection point, if odd
            # it's the centre of a cell. 
            y_shift = y_size//2 - y_grid_size//2
            x_shift = x_size//2 - x_grid_size//2
            # Shift indices along by an amount equal to the difference between
            # array and block of cell centres. Use % to wrap indices that would
            # be out of range.
            y_indices = (y_indices + y_shift) % y_size
            x_indices = (x_indices + x_shift) % x_size
        # Set rows and columns in indices arrays equal to 1.
        grid[y_indices, :] = 1
        grid[:, x_indices] = 1
        return grid
    # If spacing is not constant stack steps number of grid pattern arrays
    # with spacings varying according to np.linspace((lower, upper, steps))
    # along output axis 0.
    if not constant_spacing:
        return np.stack([grid(int(spacing)) for spacing in np.linspace(lower, upper, steps)])
    # If spacing is constant generate one grid pattern array with
    # spacing = upper and stack it steps number of times along output axis 0.
    elif constant_spacing:
        constant_grid = grid(int(upper))
        return np.stack([constant_grid for _ in np.arange(steps)])


def fft_arr3d(input_array):
    """
    Take the 2d FFT of subarrays along axis 0 of a 3d array.

    Parameters
    ----------
    input_array : numpy ndarray of numbers, shape (m, n, k)
        A 3d array made up of 2d subarrays along axis 0. 

    Returns
    -------
    output : numpy ndarray of complex, shape (m, n, k)
        A 3d array made up of 2d subarrays along axis 0, each being the 2d FFT
        of the corresponding subarray in input_array.

    """
    output = np.stack([npft.ifftshift(npft.fft2(npft.fftshift(input_array[n])))
                       for n in np.arange(np.shape(input_array)[0])])
    return output