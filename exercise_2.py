import numpy as np

# EXERCISE #2

## Answer to question 1:
#  - The problem in the exercise is that the x and y coordinates are not being swapped correctly. The function was 
#    mistakenly swapping both the x1 and y1 coordinates with y1 itself, which caused the unexpected output.
#  - The fix is to properly swap the first two coordinates (x1 with y1) and the second two coordinates (x2 with y2).

## Answer to question 2:
#  - After fixing the issue in the swap, it still returned an incorrect result. 
#  - This happened because the wrong columns were being reassigned. After fixing it, the function correctly swaps 
#    the x and y values in each row (x1 with y1, and x2 with y2).

def swap(coords: np.ndarray):
    """
    This method will flip the x and y coordinates in the coords array.

    :param coords: A numpy array of bounding box coordinates with shape [n,5] in format:
        ::

            [[x11, y11, x12, y12, classid1],
             [x21, y21, x22, y22, classid2],
             ...
             [xn1, yn1, xn2, yn2, classid3]]

    :return: The new numpy array where the x and y coordinates are flipped.

    **This method is part of a series of debugging exercises.**
    **Each Python method of this series contains bug that needs to be found.**

    The example demonstrates the issue. The returned swapped_coords are expected to have swapped
    x and y coordinates in each of the rows.
    """

    # coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], = coords[:, 1], coords[:, 1], coords[:, 3], coords[:, 2]
    # - Switching 0, 1 to 1, 0, instead of switching 0, 1 to 1, 1;
    # - So we get : coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], = coords[:, 1], coords[:, 0], coords[:, 3], coords[:, 2]
    # - The correct approach is to use coords[:, [0, 1]] = coords[:, [1, 0]] to swap the first pair of coordinates (x1, y1) and coords[:, [2, 3]] = coords[:, [3, 2]] to swap the second pair (x2, y2)
    # - ":" select all rows and [0,1] selects columns 0 and 1
    coords[:, [0, 1]] = coords[:, [1, 0]]  
    coords[:, [2, 3]] = coords[:, [3, 2]]
    return coords

coords = np.array([[10, 5, 15, 6, 0],
                   [11, 3, 13, 6, 0],
                   [5, 3, 13, 6, 1],
                   [4, 4, 13, 6, 1],
                   [6, 5, 13, 16, 1]])

swapped_coords = swap(coords)

print(swapped_coords)

# Expected output:
#[[ 5 10  6 15  0]
# [ 3 11  6 13  0]
# [ 3  5  6 13  1]
# [ 4  4  6 13  1]
# [ 5  6 16 13  1]]
