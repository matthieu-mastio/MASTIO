"""
This module defines the SquareArea class, which represents the geographical area of the simulation.
"""

import random

class SquareArea:
    """
    Represents a square geographical area for the simulation.
    """
    def __init__(self, width):
        """
        Initializes a SquareArea.

        Args:
            width (float): The width of the square area.
        """
        self.width = width

    def random_point(self):
        """
        Generates a random point within the square area.

        Returns:
            tuple: A tuple containing the (x, y) coordinates of the random point.
        """
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.width)
        return (x, y)
