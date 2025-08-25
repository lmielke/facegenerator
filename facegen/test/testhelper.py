"""
When testing temporary objects are needed to test creatable objects.
This module contains context managers to create temporary objects

Example:

@helpers.test_setup(temp_file=None, temp_chdir='temp_file', temp_pw=None) 
def test_my_fuction(self, *args, **kwargs):
    run test 
    ...

temp_file:      creates a temporary folder/file in the test_data_dir
                folder is named lke the test function here it would be (test_my_fuction)
temp_chdir:     changes the current working directory to the provided directory
                if string 'temp_file' is provided then, temp_chdir = os.path.basename(temp_file)
temp_pw:        sets the environment variables DATASAFEKEY and DATAKEY to empty strings
                this is currently not implemented
"""


import json, os, shutil, sys, time, yaml


from contextlib import contextmanager
import functools
from pathlib import Path

import facegen.settings as sts


@contextmanager
def temp_secret(j, *args, secretsFilePath: str, entryName: str, **kwargs) -> None:
    """
    temporaryly renames files in .ssp for upload to bypass files
    secretsFilePath: full path to secretsfile.json
    creds: facegen params to get secret
            {entryName: secretToWrite}
    """
    fType = os.path.splitext(secretsFilePath)[-1]
    try:
        secrets = j.secrets.get(entryName)
        with open(secretsFilePath, "w") as f:
            if fType == ".json":
                json.dump(secrets, f)
            elif fType == "sts.fext":
                yaml.dump(secrets, f)
            else:
                raise Exception(f"Invalid file extension: {fType}, use [.json, sts.fext]")
        while not os.path.exists(secretsFilePath):
            continue
        yield
    except Exception as e:
        print(f"oamailer.secrets_loader Exception: {e}")
    finally:
        if os.path.exists(secretsFilePath):
            os.remove(secretsFilePath)


@contextmanager
def temp_chdir(tempDataPath, *args, temp_chdir: Path = None, **kwargs) -> None:
    """Sets the cwd within the context

    Args:
        temp_chdir (Path): The temp_chdir to the cwd

    Yields:
        None
    """
    origin = os.getcwd()
    if temp_chdir == 'temp_file':
        if os.path.isfile(tempDataPath):
            temp_chdir = os.path.dirname(tempDataPath)
        elif os.path.isdir(tempDataPath):
            temp_chdir = tempDataPath
        else:
            raise Exception(f"testhelper.temp_chdir.tempDataPath: {tempDataPath} not found")
    temp_chdir = temp_chdir if temp_chdir is not None else os.getcwd()
    try:
        os.chdir(temp_chdir)
        yield
    finally:
        os.chdir(origin)


@contextmanager
def temp_ch_host_name(hostName: str) -> None:
    """Sets the cwd within the context
    Args:
        host (Path): The host to the cwd
    Yields:
        None
    """
    origin = os.environ.get("HOSTNAME", "")
    try:
        os.environ["HOSTNAME"] = hostName
        yield
    finally:
        os.environ["HOSTNAME"] = origin


@contextmanager
def temp_safe_rename(*args, safeName: str, prefix: str = "#", **kwargs) -> None:
    """
    temporaryly renames files in .ssp for upload to bypass files
    """
    # rename fileName by adding prefix
    fileName = f"{safeName.lower()}sts.fext"
    currPath = os.path.join(sts.encryptDir, fileName)
    tempPath = os.path.join(sts.encryptDir, f"{prefix}{fileName}")
    try:
        if os.path.exists(currPath):
            os.rename(currPath, tempPath)
        yield
    finally:
        if os.path.exists(tempPath):
            if os.path.exists(currPath):
                os.remove(currPath)
            time.sleep(0.1)
            os.rename(tempPath, currPath)
            time.sleep(0.1)


def test_setup(*args, temp_pass:str=None, **kwargs):
    """
    A decorator for setting up a test environment with a temporary file, a specific
    working/execution directory and temporay passwords for encryption/decryption.
    This decorator uses two context managers:
        temp_test_file and temp_chdir. temp_test_file creates a
    temporary file for the test, and temp_chdir temporarily changes the
    current working directory.
    Args:
        *outer_args: Arguments for the temp_test_file context manager.
                     The first argument is expected to be the path for temp_chdir.
        **outer_kwargs: Keyword arguments for the temp_test_file context manager.
    Returns:
        The decorated test function.
    """

    def decorator(test_func):
        # The actual decorator that wraps the test function
        @functools.wraps(test_func)
        def wrapper(self, *inner_args, **inner_kwargs):
            # Wrapper function that sets up the test environment and executes the test
            # Use temp_test_file to create a temporary file
            with temp_test_file(test_func.__name__, *args, **kwargs) as tempDataPath:
                # Use temp_chdir to change the current working directory
                with temp_chdir(tempDataPath, *args, **kwargs):
                    # use temporary testing passwords for encryption
                    # NOTE: this was formally included when calling unittest.main() on the
                    # test module. This did not always work however. So it was moved here.
                    with temp_password(*args, **kwargs):
                        # Execute the test function with the path to the temporary files.
                        return test_func(self, tempDataPath, *inner_args, **inner_kwargs)

        return wrapper

    return decorator


@contextmanager
def temp_test_file(temp_dir_name: str, *args, temp_file: str, **kwargs) -> None:
    """
    Creates a temp dir named like test_func_name_to_be_tested and copies test data to it
    returns the full path to the copied file. The tempDir is removed when context is left.
    Example:
    Use this to create an isolated test data source named like the test function.
    The test file can then be read, changed, destroyed.
    temp_file, temp_dir_name = 'safe_one.json', 'test__prep_api_params'
    results in:
        1. created tempDir like sts.test_data_dir/temp_dir_name
            tempDir = sts.test_data_dir/test__prep_api_params/
        2. copies sts.test_data_dir/safe_one.json to tempDir/safe_one.json
            tempPath = sts.test_data_dir/test__prep_api_params/safe_one.json
    use like Example:
    with helpers.temp_test_file('test__prep_api_params', 'safe_one.json') as tempPath:
        run_any_test_using(tempPath)
    """
    # no file is needed, but this context manager still has to run
    if temp_file is None: temp_file = 'empty.txt'
    sourcePath = os.path.join(sts.test_data_dir, temp_file)
    assert os.path.isfile(sourcePath), f"file {sourcePath} not found"
    try:
        tempDir = os.path.join(sts.test_data_dir, temp_dir_name)
        tempPath = os.path.join(tempDir, temp_file)
        if not os.path.isdir(tempDir):
            os.makedirs(tempDir)
        shutil.copyfile(sourcePath, tempPath)
        yield tempPath
    finally:
        if os.path.exists(tempDir):
            shutil.rmtree(tempDir, ignore_errors=False, onerror=None)


@contextmanager
def temp_password(*args, temp_pw: str = None, **kwargs) -> None:
    try:
        os.environ["DATASAFEKEY"] = ''
        os.environ["DATAKEY"] = ''
        yield
    finally:
        os.environ["DATASAFEKEY"] = ''
        os.environ["DATAKEY"] = ''


# some test helper functions
def rm_test_dir(tempDir, *args, **kwargs):
    try:
        shutil.rmtree(tempDir, ignore_errors=False, onerror=None)
        # pass
    except Exception as e:
        print(f"UnitTest, tearDownClass, e: {e}")


def mk_test_dir(temp_dir_name, *args, **kwargs):
    """
    test files to be encrypted are created on the fly inside a temp directory
    """
    tempDataDir = os.path.join(sts.test_data_dir, temp_dir_name)
    if not os.path.isdir(tempDataDir):
        os.makedirs(tempDataDir)
    time.sleep(0.1)
    return tempDataDir


def copy_test_data(temp_dir_name, testFileName, *args, targetName=None, **kwargs):
    print(testFileName)
    target = os.path.join(temp_dir_name, testFileName if targetName is None else targetName)
    shutil.copyfile(os.path.join(sts.test_data_dir, testFileName), target)
    return target


# helper functions for unittest setup and teardown
def mk_test_file(tempDataDir, fileName, testDataStr=None, *args, **kwargs):
    """
    test files to be encrypted are created on the fly inside a temp directory
    """
    testFilePath = os.path.join(tempDataDir, fileName)
    testDataStr = sts.cryptonizeDataStr if testDataStr is None else testDataStr
    if testFilePath.endswith(".yml"):
        if not os.path.isfile(testFilePath):
            with open(testFilePath, "w") as f:
                f.write(yaml.dump(sts.testDataDict))
    elif testFilePath.endswith(".json"):
        if not os.path.isfile(testFilePath):
            with open(testFilePath, "w+") as f:
                json.dump(json.dumps(testDataStr, ensure_ascii=False), f)
    return testFilePath
