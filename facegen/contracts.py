# contracts.py
import facegen.settings as sts
import os, sys
import facegen.arguments as arguments
from colorama import Fore, Style


def checks(*args, **kwargs):
    kwargs = clean_kwargs(*args, **kwargs)
    check_missing_kwargs(*args, **kwargs)
    check_env_vars(*args, **kwargs)
    return kwargs

def check_env_vars(*args, **kwargs):
    """
    Some processes like pm2 run the application without environment variables.
    This function checks if the required environment variables are set.
    If not it adds them by loading the .env file.
    """
    if os.environ.get('pg_alias') is None:
        from dotenv import load_dotenv
        env_file = os.path.join(sts.project_dir, ".env")
        load_dotenv(env_file)


def clean_kwargs(*args, **kwargs):
    # kwargs might come from a LLM api and might be poluted with whitespaces ect.
    cleaned_kwargs = {}
    for k, vs in kwargs.items():
        if isinstance(vs, str):
            cleaned_kwargs[k.strip()] = vs.strip().strip("'")
        else:
            cleaned_kwargs[k.strip()] = vs
    return cleaned_kwargs

def check_missing_kwargs(*args, api,  **kwargs):
    """
    Uses arguments to check if all required kwargs are provided
    """
    missings = set()
    requireds = {}
    for k, v in requireds.items():
        if k not in kwargs.keys():
            missings.add(k)
    if missings:
        print(f"{Fore.RED}Missing required arguments: {missings}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Required arguments are: {requireds}{Style.RESET_ALL}")
        exit()
