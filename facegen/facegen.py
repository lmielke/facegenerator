"""
facegen.py
This is an example module for a properly formatted Python facetype project.
It contains the DefaultClass used for facetyping Python projects.
"""

# Standard library imports in alphabetical order
import os
import re
import time
from datetime import datetime as dt
import yaml

# Local application imports in alphabetical order
import facegen.settings as sts

class DefaultClass:
    """
    DefaultClass is a class for facetyping Python projects.
    NOTE: **kwargs are critical in this class to allow for the passing of arguments
    """

    def __init__(self, *args, pg_name: str = None, verbose: int = 0, **kwargs) -> None:
        """
        Initializes DefaultClass with the following attributes.
        """
        self.verbose = verbose
        self.pg_name = pg_name

    def __str__(self, *args, **kwargs) -> str:
        """
        String representation of the DefaultClass instance.

        Returns:
            str: The string representation of the class instance.
        """
        return f"DefaultClass: {self.pg_name = }"
