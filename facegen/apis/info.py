# info.py
import subprocess
import os, sys
import pyperclip, requests
from tabulate import tabulate as tb
from colorama import Fore, Style

import facegen.settings as sts
from facegen.helpers.tree import Tree
from facegen.helpers.import_info import main as import_info
from facegen.helpers.package_info import pipenv_is_active


all_infos = {"python", "package"}


def collect_infos(msg: str, init=False, info_list: list = []) -> list:
    if init: info_list.clear()
    if msg: info_list.append(str(msg))
    return info_list


def get_infos(*args, verbose, infos: set = set(), **kwargs):
    collect_infos('', True)
    if infos:
        for info in infos:
            try:
                getattr(sys.modules[__name__], f"{info}_info")(*args, verbose=verbose, **kwargs)
            except Exception as e:
                print(
                    f"{Fore.RED}ERROR:{Fore.RESET} in {info}_info {e = }. Skipping..."
                )
    collect_infos(
        f"{Fore.YELLOW}\nfor more infos: {Style.RESET_ALL}face info "
        f"{Fore.YELLOW}-i{Style.RESET_ALL} {all_infos} "
        f"{Fore.YELLOW}-v{Style.RESET_ALL} 1-3"
    )
    user_info(*args, **kwargs)
    server_info(*args, **kwargs)

def user_info(*args, **kwargs):
    msg = f"""\n{f" PROTOPY USER info ":#^80}"""
    collect_infos(f"{Fore.GREEN}{msg}{Style.RESET_ALL}")

def server_info(*args, **kwargs):
    msg = f"{Fore.YELLOW}Modify User settings{Fore.RESET}: {sts.user_settings_path}!\n"
    collect_infos(msg)
    msg = f"{Fore.YELLOW}serve:{Fore.RESET} face server {Style.DIM}# port is {sts.port}{Style.RESET_ALL}\n"
    collect_infos(msg)

def python_info(*args, **kwargs):
    collect_infos(f"""\n{Fore.YELLOW}{f" PYTHON info ":#^80}{Style.RESET_ALL}""")
    collect_infos(f"{sys.executable = }\n{sys.version}\n{sys.version_info}")
    with open(os.path.join(sts.project_dir, "Pipfile"), "r") as f:
        collect_infos(f.read())

def package_info(*args, verbose: int = 0, **kwargs):
    collect_infos(f"""\n{Fore.YELLOW}{f" PACKAGE info ":#^80}{Style.RESET_ALL}""")
    collect_infos(f"\n{sts.project_name = }\n{sts.package_dir = }\n{sts.test_dir = }")
    collect_infos(f"\n\n{sts.project_dir = }")
    collect_infos(f"{sts.package_name = }\n")
    collect_infos(
        (
            f"$PWD: {os.getcwd()}\n"
            f"$EXE: {sys.executable} -> {pipenv_is_active(sys.executable) = }\n"
        )
    )
    ignores = sts.ignore_dirs | {"gp", "models"}
    tree = (Tree(*args, verbose=verbose, **kwargs)(sts.project_dir,
                                                                colorized=True,
                                                                ignores=ignores,
                                                                verbose=verbose)
    )
    collect_infos(f"{tree.get('tree')}\n")
    if verbose:
        collect_infos(f"{tree.get('contents')}\n")
    try:
        collect_infos(
            subprocess.run(
                f"face -h".split(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        )
    except Exception as e:
        print(f"{Fore.RED}Error:{Fore.RESET} {e}")
    collect_infos(
        f"Project import structure:\n" f"{import_info(main_file_name='facegen.py', verbose=0, )}"
    )
    with open(os.path.join(sts.project_dir, "Readme.md"), "r") as f:
        collect_infos(f"\n<readme>\n# Readme.md\n{f.read()}\n</readme>\n")
        # package help


def main(*args, clip=None, **kwargs) -> str:
    get_infos(*args, **kwargs)
    out = "\n".join(collect_infos(f"info.main({kwargs})"))
    if clip:
        pyperclip.copy(out)
        print(f"{Fore.GREEN}Copied to clipboard!{Style.RESET_ALL}")
    return out

if __name__ == "__main__":
    main(verbose=2, infos=all_infos, clip=False)