# package_info.py
import os, re, time
import facegen.settings as sts
from facegen.helpers.collections import temp_chdir
from colorama import Fore, Style

ignoreDirs = {
    ".git",
    "build",
    "dist",
    "logs",
    "*.egg-info",
    "__pycache__",
    ".pytest_cache",
    ".tox",
}

start_token, end_token = '<hierarchy>', '</hierarchy>'
dir_symbol_raw = '▼'
dir_symbol = f"{Fore.YELLOW} {dir_symbol_raw} {Style.RESET_ALL}"
dir_conn, file_conn = f"|--", f"|-"
dir_discontinued_raw =  f"|..."
dir_discontinued =  f"{Style.DIM}{Fore.WHITE}{dir_discontinued_raw}{Style.RESET_ALL}"
file_color = lambda file: Fore.BLUE if file.endswith(".py") else ""
indent = "    "

def dirs_to_tree(projectDir, *args, ignores:set=set(), **kwargs):
    global ignoreDirs
    ignoreDirs = ignoreDirs | ignores
    prStruct = f"{start_token}\n"
    baseLevel = projectDir.count(os.sep)
    for root, dirs, files in os.walk(projectDir):
        level = root.count(os.sep) - baseLevel
        subdir = os.path.basename(root)

        # Check if the directory should be ignored
        ignoreDir = any(
            subdir.endswith(igDir.lstrip("*")) if igDir.startswith("*") else subdir == igDir
            for igDir in ignoreDirs
        )
        if ignoreDir:
            prStruct += (
                f"{indent * level}{dir_conn}{dir_symbol}{subdir}\n"
                f"{indent * (level + 1)}{dir_discontinued}\n"
            )
            dirs[:] = []  # Prevent further traversal into this directory
        else:
            prStruct += f"{indent * level}{dir_conn}{dir_symbol}{subdir}\n"
            indentOn = indent * (level + 1)
            logDir = subdir.endswith("log") or subdir.endswith("logs")
            printFiles = False

            for file in files:
                if logDir and printFiles:
                    prStruct += f"{indentOn}{dir_discontinued}\n"
                    break
                prStruct += f"{indentOn}{file_conn}{file_color(file)}{file}{Style.RESET_ALL}\n"
                printFiles = True

            # Update dirs for next iteration, excluding current ignored directory
            if ignoreDir:
                dirs[:] = []
    return prStruct + f"{end_token}\n"

def tree_to_dirs(tree, *args, **kwargs):
    """
    take a tree structure as created by dirs_to_tree and back convert it into a list of 
    tuples. Lines need to be cleaned from color entries.
    
    Example tree:
    |-- ▼ test
    |-testhelper.py
    |-__init__.py
    |-- ▼ data
        |-empty.txt
        |-facegen.yml
    |-- ▼ logs
        |...
    |-- ▼ testopia_logs
        |-2024-01-12-14-26-24-796531_test_facepackage.log
        |...
    |...
    
    Example paths:
    [['test', True],
    ['test\\testhelper.py', False],
    ['test\\__init__.py', False],
    ['test\\data', True],
    ['test\\logs', True],
    ['test\\testopia_logs', True],
    ['test\\testopia_logs\\2024-01-12-14-26-24-796531_test_facepackage.log', False],
    ...,]
    So we should be able to recreate the orignal folder structure by doing:
        for path, is_dir in tree_to_dirs(tree):
            if not os.path.exists(path):
                if is_dir:
                    os.makedirs(path)
                else:
                    with open(path, "w") as f:
                        f.write(f"#{os.path.basename(path)}")
    """

    def _cleanup_line(line, *args, **kwargs):
        """
        take a line from dirs_to_tree and cleanup the line
        """
        line = line.strip()
        is_dir = True if line.startswith(dir_conn) else False
        # remove connectors conn
        line = line.replace(dir_discontinued, "")
        line = line.replace(dir_conn, '').replace(file_conn, '')
        line = line.replace(dir_symbol, "").replace(dir_symbol_raw, "")
        line = _decolorize(line, *args, **kwargs)
        return line.strip(), is_dir

    def _normalize(tree, *args, **kwargs):
        # normalized lines have level 0 indents removed, so level 0 has no indents
        border_line = lambda line: (start_token in line) or (end_token in line)
        lines = [line for line in tree.split("\n") if line and not border_line(line)]
        start_level = lines[0].count(indent)
        normalized = [line.replace(indent * start_level, "", 1) for line in lines]
        return normalized

    def _parse_tree(tree, *args, **kwargs):
        paths, temps = [], []
        # tree might have indents that are not hierarchy but only display indents
        for i, line in enumerate(_normalize(tree, *args, **kwargs), 0):
            level = line.count(indent)
            line, is_dir = _cleanup_line(line)
            if not line or line == dir_discontinued_raw: continue
            # print(f"{i}, {level = }, {len(temps) = } {line = }")
            if len(temps) < level:
                temps.append(line)
            else:
                temps = temps[:level]
                temps.append(line)
            paths.append([os.path.join(*temps), is_dir])
        return paths
    return _parse_tree(tree, *args, **kwargs)

def _decolorize(line, *args, **kwargs):
    line = line.replace(Fore.YELLOW, "").replace(Fore.BLUE, "").replace(Fore.GREEN, "")
    line = line.replace(Style.DIM, "").replace(Fore.WHITE, "").replace(Style.RESET_ALL, "")
    return line

def _decolorize_tree(tree, *args, **kwargs):
    lines = [line for line in tree.split("\n") if line]
    decolorized = '\n'.join([_decolorize(line, *args, **kwargs) for line in lines])
    return decolorized

def mk_dirs_hierarchy(dirs:list[list], tgt_path:str, *args, **kwargs) -> None:
    """
    takes a list of lists (dirs) like ((tgt_path:str, is_dir:bool), ...) 
    and creates the directory structure in tgt_path
    if the tgt_path is a file, it will be created with a comment as temporary content
    """
    with temp_chdir(tgt_path):
        for path, is_dir in dirs:
            if not os.path.exists(path):
                if is_dir:
                    os.makedirs(path)
                else:
                    with open(path, "w") as f:
                        f.write(f"#{os.path.basename(path)}\n\n")

# project environment info
def pipenv_is_active(exec_path, *args, **kwargs):
    """
    check if the environment is active 
    pipenv is active when the package name appears as the basename of the exec_path
    """
    # print(f"{os.path.basename(exec_path.split('Scripts')[0].strip(os.sep)) = }")
    is_active = os.path.basename(exec_path.split('Scripts')[0]\
                    .strip(os.sep)).startswith(sts.project_name)
    return is_active