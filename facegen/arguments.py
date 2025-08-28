"""
    pararses facegen arguments and keyword arguments
    args are provided by a function call to mk_args()
    
    RUN like:
    import facegen.arguments
    kwargs.updeate(arguments.mk_args().__dict__)
"""
import argparse
from typing import Dict


def mk_args():
    parser = argparse.ArgumentParser(description="run: python -m facegen info")
    parser.add_argument(
                            "api", 
                            metavar="api", nargs=None, 
                            help=(
                                    f""
                                    f"see facegen.apis"
                                )
                        )
    parser.add_argument(
        "-i",
        "--infos",
        required=False,
        nargs="+",
        const=None,
        type=str,
        default=None,
        help="list of infos to be retreived, default: all",
    )

    parser.add_argument(
        "-s",
        "--sketch_name",
        required=False,
        nargs="?",
        const="flower",
        type=str,
        default="flower",
        help="name of the sketch preset (default: flower, looks in ~/.facegen/sketches)",
    )

    parser.add_argument(
        "-n",
        "--num_objects",
        required=False,
        nargs="?",
        const=1,
        type=int,
        default=1,
        help="number of objects (default: 1)",
    )

    parser.add_argument(
        "-x",
        "--num_points",
        required=False,
        nargs="?",
        const=255,
        type=int,
        default=255,
        help="number of points per object (default: 255)",
    )

    parser.add_argument(
        "-z",
        "--num_layers",
        required=False,
        nargs="?",
        const=255,
        type=int,
        default=None,
        help="number of layers in extrusion (default: None, no extrusion running)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        nargs="?",
        const=1,
        type=int,
        default=0,
        help="0:silent, 1:user, 2:debug",
    )

    parser.add_argument(
        "-y",
        "--yes",
        required=False,
        nargs="?",
        const=1,
        type=bool,
        default=None,
        help="run without confirm, not used",
    )

    return parser.parse_args()



def get_required_flags(parser: argparse.ArgumentParser) -> Dict[str, bool]:
    """
    Extracts the 'required' flag for each argument from an argparse.ArgumentParser object.

    Args:
        parser (argparse.ArgumentParser): The parser to extract required flags from.

    Returns:
        Dict[str, bool]: A dictionary with argument names as keys and their 'required' status as values.
    """
    required_flags = {}
    for action in parser._actions:
        if isinstance(action, argparse._StoreAction):
            # For positional arguments, the 'required' attribute is not explicitly set,
            # but they are required by default.
            is_required = getattr(action, 'required', True) if action.option_strings == [] else action.required
            # Option strings is a list of option strings (e.g., '-f', '--foo').
            for option_string in action.option_strings:
                required_flags[option_string] = is_required
            if not action.option_strings: # For positional arguments
                required_flags[action.dest] = is_required
    return required_flags

if __name__ == "__main__":
    parser = mk_args()
    required_flags = get_required_flags(parser)
    print(required_flags)
