# entry_point.py
# this is an example api for facegen

from facegen import settings as sts
from facegen.facegen import DefaultClass

def entry_point_function(*args, **kwargs):
    inst = DefaultClass(*args, **kwargs)
    return inst

def main(*args, **kwargs):
    """
    All entry points must contain a main function like main(*args, **kwargs)
    """
    return entry_point_function(*args, pg_name=sts.package_name, **kwargs)