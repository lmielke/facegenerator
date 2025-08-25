import ast, inspect, json, os, re, sys, textwrap, types
from copy import deepcopy
from datetime import datetime as dt
from functools import wraps
from typing import Any, Callable, Dict, List

from colorama import Fore, Style
from tabulate import tabulate as tb

import facegen.settings as sts
from dataclasses import dataclass, asdict


@dataclass
class BaseSchema:
    name: str = ""
    description: str = ""
    import_path: str = ""
    module_path: str = ""
    parameters: Dict[str, Any] = None
    body: str = ""
    returns: str = ""
    example: str = ""

    @classmethod
    def set_fields(cls, main_meth: Callable, test_meth: Callable, parsed_props: Dict[str, Any]
    ) -> "BaseSchema":
        """Set schema fields from the target function."""
        return cls(
                    name=main_meth.__qualname__,
                    description=main_meth.__doc__,
                    import_path=main_meth.__module__,
                    module_path=inspect.getmodule(main_meth).__file__,
                    parameters={"type": "object", "properties": parsed_props},
                    body=cls.get_function_code(main_meth),
                    returns=cls.handle_returns_inspect(main_meth),
                    example=cls.get_function_code(test_meth),
        )

    @staticmethod
    def get_function_code(func: Callable) -> str:
        lines, _ = inspect.getsourcelines(func)
        docStrRegex = "'''|\"\"\""
        ixs = sorted([ix.start() for ix in re.finditer(docStrRegex, ''.join(lines))])
        if len(ixs) >= 2:
            body_lines = [line for ix, line in enumerate(lines) if ix < ixs[0] or ix > ixs[1]]
        else:
            body_lines = lines
        return ''.join(body_lines).strip()

    @staticmethod
    def handle_returns_inspect(func: Callable) -> str:
        """
        Map the Python return-annotation to a JSON-Schema primitive.
        Falls back to ``object`` or the raw string if no match is found.
        """
        py2json: dict[type, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }
        ann = inspect.signature(func).return_annotation
        if ann is inspect._empty:
            return "object"
        return py2json.get(ann, str(ann))

    def to_dict(self) -> dict:
        return self.__dict__

# ──────────────────────────────────── OpenaiSchema ────────────────────────────────────
@dataclass
class OpenaiSchema:
    """
    Minimal schema for OpenAI tool-calling.
    * `description` now carries the *entire* function source (signature + body).
    * Field `example` has been removed – it stays only in `BaseSchema`.
    """
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] | None = None
    returns: str = ""

    # --------------------------------------------------------------------- #
    @classmethod
    def set_fields(
        cls,
        main_meth: Callable,
        test_meth: Callable,
        parsed_props: Dict[str, Any],
        *args, **kwargs,
    ) -> "OpenaiSchema":
        """
        Build a minimal, schema-compliant tool definition for OpenAI.
        Strips per-property “required” flags and puts them in the top-level array.
        """
        # collect required names before mutating props
        required_names = [
            name for name, meta in parsed_props.items()
            if meta.get("required", False) and meta.get("type") != "object"
        ]

        # add enum options from docstring
        for p, d in FunctionToJson.parse_docstring(main_meth.__doc__).items():
            if "options" in d and p in parsed_props:
                parsed_props[p]["enum"] = d["options"] or []

        # remove the invalid key from property dicts
        clean_props: Dict[str, Dict[str, Any]] = {
            name: {k: v for k, v in meta.items() if k != "required"}
            for name, meta in parsed_props.items()
        }

        full_src = textwrap.dedent(inspect.getsource(main_meth).strip())

        return cls(
            name=main_meth.__qualname__,
            description=full_src,
            parameters={
                "type": "object",
                "properties": clean_props,
                "required": required_names,
            },
            returns=BaseSchema.handle_returns_inspect(main_meth),
        )


    # --------------------------------------------------------------------- #
    def to_dict(self) -> dict:
        """Serialize for `json.dumps`."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
        }
# ───────────────────────────────────────────────────────────────────────────────────────


@dataclass
class ExecutionInfo:
    import_path: str
    module_path: str

    @classmethod
    def set_fields(cls, main_meth: Callable, *args) -> "ExecutionInfo":
        return cls(
            import_path=main_meth.__module__,
            module_path=inspect.getmodule(main_meth).__file__,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JoSchema:
    api: str
    response: str
    path: str

    @classmethod
    def set_fields(cls, main_meth: Callable, *args) -> "JoSchema":
        return cls(
            api=main_meth.__name__,
            response=BaseSchema.handle_returns_inspect(main_meth),
            path=inspect.getmodule(main_meth).__file__,
        )

    def to_dict(self) -> dict:
        return asdict(self)


class FunctionToJson:

    def __init__(self, *dec_args, schemas: set = set(), file_name: str = None,
                 write: bool = True, verbose: int = 0, **kwargs):
        self.schemas = schemas
        self.file_name = file_name
        self.asts = {}

    def __call__(self, test_meth: Callable) -> Callable:

        @wraps(test_meth)
        def wrapper(*args, **kwargs):
            main_meth = self._find_method(*self._get_object_names(test_meth))
            self.get_asts(main_meth, test_meth)
            self.dump_to_json(main_meth.__module__, main_meth.__name__, *args)
            return test_meth(*args, **kwargs)
        return wrapper

    def _get_object_names(self, test_meth: Callable) -> tuple:
        try:
            cl_name, mth_name = test_meth.__qualname__.split('.')
            if cl_name and mth_name:
                return cl_name, mth_name, inspect.getmodule(test_meth)
        except:
            raise ValueError(   f"FunctionToJson._get_object_names ERROR! "
                                f"Invalid method name '{test_meth.__qualname__}'.")

    def _find_method(self, cl_name: str, mth_name: str, module: object) -> Callable:
        found_class = getattr(module, cl_name.replace('Test_', ''), None)
        if found_class:
            return getattr(found_class, mth_name.replace('test_', ''), None)
        else:
            raise AttributeError(f"Class '{cl_name}' not found in '{module.__name__}'.")
        return None

    def get_asts(self, *args, **kwargs) -> Dict[str, dict]:
        meth_props = self.read_signature(*args)
        self.asts['base'] = BaseSchema.set_fields(*args, meth_props).to_dict()
        self.asts['execution'] = ExecutionInfo.set_fields(*args).to_dict()
        for schema in self.schemas:
            try:
                sch = getattr(sys.modules[__name__], f"{schema.capitalize()}Schema")
                self.asts[schema] = sch.set_fields(*args, deepcopy(meth_props) ).to_dict()
            except AttributeError:
                raise AttributeError(f"Schema class '{schema}' not found in module.")

    @staticmethod
    def read_signature(func: Callable, *args) -> Dict[str, Dict[str, object]]:
        """
        Build a JSON-Schema-ready properties map.
        Skips implicit parameters and *args/**kwargs.
        """
        PY2JSON = {str: "string", int: "integer", float: "number",
                   bool: "boolean", list: "array", dict: "object"}
        props: Dict[str, Dict[str, object]] = {}
        for name, p in inspect.signature(func).parameters.items():
            if name in {"self", "cls"} or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            ann = p.annotation if p.annotation is not inspect._empty else object
            json_type = PY2JSON.get(ann, "object")
            props[name] = {
                "type": json_type,
                "default": None if p.default is inspect._empty else p.default,
                "required": p.default is inspect._empty,
            }
        return props

    def dump_to_json(self, dot_import: str, m_name: str, write: bool) -> None:
        def default_serializer(obj):
            if isinstance(obj, (type, types.ModuleType)):
                return f"<{obj.__name__}>"
            return str(obj)
        if write:
            file_name = f"{dot_import}.{m_name}" if self.file_name is None else self.file_name
            if not file_name.endswith(".json"):
                file_name += ".json"
            with open(os.path.join(sts.apis_json_dir, file_name), "w") as f:
                f.write(json.dumps(self.asts, indent=4, default=default_serializer))

    @staticmethod
    def parse_docstring(docstring):
        args_section = re.search(r'Args:\n\s+(.*?)(\n\n|\Z)', docstring or '', re.DOTALL)
        if not args_section:
            return {}
        args_text = args_section.group(1)
        args = {}
        current_arg = None
        options_found = False
        for line in args_text.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                options_found = True
                if current_arg:
                    args[current_arg].setdefault('options', []).append(line[2:])
            else:
                arg_match = re.match(r'(\w+) \((.+?)\): (.+)', line)
                if arg_match:
                    if current_arg and not options_found:
                        args[current_arg]['options'] = None
                    current_arg = arg_match.group(1)
                    options_found = False
                    args[current_arg] = {
                        'type': arg_match.group(2),
                        'description': arg_match.group(3),
                    }
                elif current_arg:
                    args[current_arg]['description'] += ' ' + line
        return args
