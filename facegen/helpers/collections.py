# collections.py
import json, os, re, shutil, subprocess, sys, textwrap, time, yaml
from contextlib import contextmanager
from pathlib import Path
from tabulate import tabulate as tb
from datetime import datetime as dt

import facegen.settings as sts

def unalias_path(work_path: str) -> str:
    """
    repplaces path aliasse such as . ~ with path text
    """
    if not any([e in work_path for e in [".", "~", "%"]]):
        return work_path
    work_path = work_path.replace(r"%USERPROFILE%", "~")
    work_path = work_path.replace("~", os.path.expanduser("~"))
    if work_path.startswith(".."):
        work_path = os.path.join(os.path.dirname(os.getcwd()), work_path[3:])
    elif work_path.startswith("."):
        work_path = os.path.join(os.getcwd(), work_path[2:])
    work_path = os.path.normpath(os.path.abspath(work_path))
    return work_path


def _handle_integer_keys(self, intDict) -> dict:
    """
    helper function for api calls
    api endpoint calls are called by providing the relevant api api index
    as an integer. During serialization its converted to string and therefore
    has to be reconverted to int here
    """
    intDict = {int(k) if str(k).isnumeric() else k: vs for k, vs in intDict.items()}
    return intDict


def prep_path(work_path: str, file_prefix=None) -> str:
    work_path = unalias_path(work_path)
    if os.path.exists(work_path):
        return work_path
    # check for extensions
    extensions = ["", sts.eext, sts.fext]
    name, extension = os.path.splitext(os.path.basename(work_path))
    for ext in extensions:
        work_path = unalias_path(f"{name}{ext}")
        if os.path.isfile(work_path):
            return work_path
    return f"{name}{extension}"


def get_sec_entry(d, matcher, ret="key", current_key=None) -> str:
    if isinstance(d, dict):
        for key, value in d.items():
            if key == matcher:
                return current_key if ret == "key" else d[key]
            elif isinstance(value, dict):
                result = get_sec_entry(value, matcher, ret, current_key=key)
                if result is not None:
                    return result
    return None


def load_yml(testFilePath, *args, **kwargs):
    with open(testFilePath, "r") as f:
        return yaml.safe_load(f)


def load_str(testFilePath, *args, **kwargs):
    with open(testFilePath, "r") as f:
        return f.read()


def group_text(text, charLen, *args, **kwargs):
    # performs a conditional group by charLen depending on type(text, list)
    if not text:
        return "None"
    elif type(text) is str:
        text = handle_existing_linebreaks(text, *args, **kwargs)
        # print(f"0: {text = }")
        text = '\n'.join(textwrap.wrap(text, width=charLen))
        # print(f"1: {text = }")
        text = restore_existing_linebreaks(text, *args, **kwargs)
        # print(f"2: {text = }")
        # text = text.replace(' <lb> ', '\n').replace('<lb>', '\n')
        # text = text.replace('<tab>', '\t')
    elif type(text) is list:
        text = "\n".join(textwrap.wrap("\n".join([t for t in text]), width=charLen))
    else:
        print(type(text), text)
    return '\n' + text

def handle_existing_linebreaks(text, *args, **kwargs):
    text = text.replace('\n', ' <lb> ').replace('\t', ' <tab> ')
    text = text.replace('#'*29, '#'*5)
    text = text.replace('-'*29, '-'*5)
    return text

def restore_existing_linebreaks(text, *args, **kwargs) -> str:
    """
    Adds a line break before numbered list items, preserving existing line breaks.

    Args:
        text (str): The text where line breaks and numbered lists will be adjusted.

    Returns:
        str: The text with the modified line breaks.
    """
    # print(re.findall(r'<lb>\s*(\d+\.\s)', text))
    text = re.sub(r'(<lb>\s*)(\d+\.\s)', r'\n\2', text)
    text = re.sub(r'(<lb>\s*)(-\s*)', r'\n\2', text)
    text = re.sub(r'(<lb>\s*)(<code_block_\d+>\s*)(<lb>\s*)', r'\n\2\n', text)
    text = re.sub(r'\n(\n\d+\.\s|\n-\s|\n<code_block_\d+\s)', r'\1', text)
    # print(f"restore_existing_linebreaks: \n{text = }")
    text = re.sub(r'\n<lb>\s*', r'\n', text)
    text = re.sub(r'\s*<lb>\s*', r' ', text)
    return text

def collect_ignored_dirs(source, ignore_dirs, *args, **kwargs):
    """
    Uses os.walk and regular expressions to collect directories to be ignored.

    Args:
        source (str): The root directory to start searching from.
        ignore_dirs (list of str): Regular expressions for directory paths to ignore.

    Returns:
        set: A set of directories to be ignored.
    """
    ignored = set()
    regexs = [re.compile(d) for d in ignore_dirs]

    for root, dirs, _ in os.walk(source, topdown=True):
        for dir in dirs:
            dir_path = os.path.join(root, dir).replace(os.sep, '/')
            if any(regex.search(dir_path) for regex in regexs):
                ignored.add(os.path.normpath(dir_path))
    return ignored

def custom_ignore(ignored):
    """
    Custom ignore function for shutil.copytree.

    Args:
        ignored (set): Set of directory paths to ignore.

    Returns:
        callable: A function that shutil.copytree can use to determine what to ignore.
    """
    def _ignore_func(dir, cs):
        return set(c for c in cs if os.path.join(dir, c) in ignored)
    return _ignore_func


@contextmanager
def temp_chdir(target_dir: str) -> None:
    """
    Context manager for temporarily changing the current working directory.

    Parameters:
    target_dir (str): The target directory to change to.

    Yields:
    None
    """
    original_dir = os.getcwd()
    try:
        os.chdir(target_dir)
        yield
    finally:
        os.chdir(original_dir)

def ppm(msg, *args, **kwargs):
    """
    Pretty print messages
    """
    # contents a printed without headers and table borders
    tbl_params = {'tablefmt': 'plain', 'headers': ''}
    msg["content"] = pretty_print_messages([msg["content"]], *args, **tbl_params, **kwargs)
    return pretty_print_messages([msg], *args, **kwargs)

def pretty_print_messages(messages, *args, verbose:int=0, clear=True, save=True, **kwargs):
    """
    Takes self.messages and prints them as a tabulate table with two columns 
    (role, content)
    """
    tbl = to_tbl(messages, *args, verbose=verbose, **kwargs)
    # use subprocess to clear the terminal
    if clear: subprocess.run(["cmd.exe", "/c", "cls"])
    # print(printable)
    if save: save_table(tbl, *args, **kwargs)
    return tbl

def to_tbl(data, *args, verbose:int=0, headers=['EXPERT', 'MESSAGE'], tablefmt='simple', **kwargs):
    tbl = []
    for m in data:
        name, content = m.get('name'), m.get('content', m.get('text'))
        role, mId = m.get('role'), m.get('mId')
        # content = hide_tags(content, *args, verbose=verbose, **kwargs)
        tbl.append((f"{color_expert(name, role)}\n{mId}", content))
    return tb(tbl, headers=headers, tablefmt=tablefmt)

def color_expert(name, role, *args, **kwargs):
    try:
        color = sts.experts.get(name.lower(), {}).color_code
    except AttributeError:
        color = Fore.RED
    name = f"{sts.watermark if role == 'agent' else ''} {color}{name}:{Style.RESET_ALL}"
    return name

def colorize_code_blocks(code_blocks:dict, *args, **kwargs):
    # colorize code blocks
    colorized = {}
    for name, (language, block) in code_blocks.items():
        block = f"{sts.code_color}{block}{Style.RESET_ALL}"
        block = '\t' + block.replace('\n', '\t\t')
        language = f"{sts.language_color}{language}{Style.RESET_ALL}"
        colorized[name] = '´´´' + language + '\n\n' + block + '\n\n´´´'
    return colorized

def _decolorize(line, *args, **kwargs):
    line = line.replace(Fore.YELLOW, "").replace(Fore.BLUE, "").replace(Fore.GREEN, "")
    line = line.replace(Fore.RED, "").replace(Fore.CYAN, "").replace(Fore.WHITE, "")
    line = line.replace(sts.DIM, "").replace(Fore.RESET, "").replace(Style.RESET_ALL, "")
    return line

def save_table(tbl, *args, **kwargs):
    table_path = os.path.join(sts.chat_logs_dir, f"{sts.session_time_stamp}_chat.log")
    tbl = '\n'.join([_decolorize(l) for l in tbl.split('\n')])
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(tbl)


def hide_tags(text, *args, verbose:int=0, **kwargs):
    """
    Takes a string and removes all tag enclosed contents from the string

    Args:
        text (str): The text to remove tags from.
        tags (list, optional): The tags to remove. Defaults to None.
            Options:
                ['pg_info', ]: removes everything between <pg_info> and </pg_info>
    """
    for start, end in sts.tags.values():
        # the replacement term depends on verbosity. If verbose, add a newline
        # otherwise remove the tag completely.
        if verbose == 0:
            start, replacement = r"\s*" + start, r''
        elif verbose <= 1:
            replacement = f"{Fore.CYAN}{start}...{end}{Style.RESET_ALL}"
        else:
            replacement = f'\n{Fore.CYAN}\\1{Style.RESET_ALL}\n'
        # replacement = r'' if verbose < 1 else f"{start}...{end}" if verbose < 2 else r'\n\1\n'
        # flags must be set to re.DOTALL to match newline characters and multiline strings
        text = re.sub(fr'({start}.*?{end})', replacement, text, flags=re.DOTALL)
    return '\n' + text

def prettyfy_instructions(instructs, tag:str='instructs', *args, verbose:int=1, **kwargs):
    # instructions are cyan
    if verbose >= 2:
        prettyfied = group_text(instructs, 70).replace('\n', '\n\t')
    elif verbose >= 1:
        prettyfied = group_text(instructs, 70).replace('\n', '\n\t')[:70]
        prettyfied += f"\n\t...[hidden {len(instructs) - 70} characters], verbose={verbose}"
    else:
        return ''
    return add_tags('\t' + prettyfied, tag)


def add_tags(content, tag, *args, **kwargs):
    """
    Adds tags to the message content.
    """
    start, end, color = sts.tags.get( tag, ('', '', '') )
    return f"{color}{start}{Fore.RESET}{content}\n{color}{end}{Fore.RESET}"

def strip_ansi_codes(text: str) -> str:
    """
    Strip ANSI escape sequences from a text string.
    Args:
        text (str): Text containing ANSI escape codes.
    Returns:
        str: Text with ANSI codes removed.
    """
    ansi_escape = re.compile(r'\x1b\[([0-9]+)(;[0-9]+)*m')
    return ansi_escape.sub('', text)
