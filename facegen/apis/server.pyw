# C:\Users\lars\python_venvs\packages\facegenerator\facegen\apis\server.pyw

import http.server
import socketserver
import os
import logging
import importlib
from urllib.parse import urlparse, parse_qs
import io
import contextlib
import pyttsx3
import facegen.settings as sts

# This server now acts as a dynamic control endpoint for the facegen package.

def _speak_message(message: str, *args, **kwargs):
    """Uses pyttsx3 to speak a given message."""
    try:
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        logging.error(f"Text-to-speech failed: {e}")


class FaceControlHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom request handler that discovers and runs available API modules.
    """
    # Class attribute to hold the discovered API modules
    available_apis = {}

    @classmethod
    def load_apis(cls, *args, **kwargs):
        """
        Scans the 'apis' directory, imports all valid .py modules,
        and stores them in the available_apis dictionary.
        """
        cls.available_apis = {}
        apis_dir = os.path.dirname(__file__)
        current_module_name = os.path.splitext(os.path.basename(__file__))[0]
        for filename in os.listdir(apis_dir):
            if not filename.endswith(".py") or filename.startswith(("_", "#")):
                continue
            
            api_name = os.path.splitext(filename)[0]
            if api_name == current_module_name:
                continue # Don't import self
            try:
                module_path = f"facegen.apis.{api_name}"
                module = importlib.import_module(module_path)
                if hasattr(module, "main"):
                    cls.available_apis[api_name] = module
                    logging.info(f"Successfully loaded API: '{api_name}'")
                else:
                    logging.warning(f"Module '{module_path}' has no main() function, skipping.")
            except Exception as e:
                logging.error(f"Failed to load API '{api_name}': {e}")

    def do_GET(self, *args, **kwargs):
        """
        Handles GET requests. If the path matches a loaded API, it runs it.
        Otherwise, it lists available APIs.
        """
        parsed_url = urlparse(self.path)
        api_name = parsed_url.path.strip("/")
        target_api_module = self.available_apis.get(api_name)
        if target_api_module:
            try:
                api_response = self.run_api_command(
                    *args,
                    api_module=target_api_module,
                    parsed_url=parsed_url,
                    **kwargs
                )
                self._send_ok_response(api_response, *args, **kwargs)
            except Exception as e:
                self.send_error(500, f"Error executing API '{api_name}': {e}")
                logging.error(f"Failed to run API command for '{api_name}': {e}")
        else:
            # List available APIs if the requested one is not found
            available_list = list(self.available_apis.keys())
            content = f"API '{api_name}' not found.\nAvailable APIs: {available_list}"
            self._send_ok_response(content, *args, **kwargs)

    def _run_api(self, *args, api_module, prepared_kwargs, **kwargs) -> str:
        """Runs the API and captures its returned output."""
        # The API's printed output will be ignored, only the return value is used.
        with contextlib.redirect_stdout(io.StringIO()):
            return_value = api_module.main(*args, **prepared_kwargs)
        return return_value if isinstance(return_value, str) else ""

    def _send_ok_response(self, content: str, *args, **kwargs):
        """Sends a 200 OK response with the provided content."""
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def run_api_command(self, *args, api_module, parsed_url, **kwargs) -> str:
        """
        Orchestrates running an API command and returns its output as a string.
        Raises exceptions on failure.
        """
        query_params = parse_qs(parsed_url.query)
        prepared_kwargs = self._prepare_kwargs(*args, query_params=query_params, **kwargs)
        
        api_output = self._run_api(
            *args,
            api_module=api_module,
            prepared_kwargs=prepared_kwargs,
            **kwargs
        )
        return api_output

    def _prepare_kwargs(self, *args, query_params: dict, **kwargs) -> dict:
        """
        Converts parsed query string dict to a clean kwargs dict for the API.
        """
        prepared_kwargs = {}
        for key, value_list in query_params.items():
            if not value_list:
                continue
            
            if key == 'infos':
                prepared_kwargs[key] = value_list
                continue
            val = value_list[0]
            
            if val.isdigit():
                prepared_kwargs[key] = int(val)
            elif val.lower() in ['true', 'false']:
                prepared_kwargs[key] = val.lower() == 'true'
            else:
                prepared_kwargs[key] = val
        
        if 'verbose' not in prepared_kwargs:
            prepared_kwargs['verbose'] = 0
            
        return prepared_kwargs

def run_server(*args, verbose:int=1, **kwargs):
    """Sets up and runs the HTTP server indefinitely."""
    FaceControlHandler.load_apis(*args, **kwargs)
    handler = FaceControlHandler
    with socketserver.TCPServer(("", sts.port), handler) as httpd:
        startup_message = f"{sts.package_name} control server starting on port {sts.port}"
        logging.info(startup_message)
        logging.info(f"Available API endpoints: {list(handler.available_apis.keys())}")        
        if verbose >= 1:
            _speak_message(startup_message, *args, **kwargs)
        httpd.serve_forever()

def main(*args, **kwargs):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_server(*args, **kwargs)

if __name__ == "__main__":
    main()
