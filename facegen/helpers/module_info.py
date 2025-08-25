"""
module_structure.py
module creation and maintenance plus sample module style template
"""

# standard imports are inline in alpabetical order
import os, re, time
# settings import is almost always required
import facegen.settings as sts


class Module:

    def __init__(self, *args, module_path:str, **kwargs):
        """
        Module generator class to create and maintain python modules
        
        Args:
            module_path: (str) abspath to module including file_name 
                Options:
                    - os.path.abspath(__file__)
        """
        self.module_path = module_path
        self.module_doc_str = None
        self.coding = None
        self._mk_asserts(*args, **kwargs)

    def create_module(self, *args, **kwargs):
        self.set_module_doc_string(*args, **kwargs)
        self.lint_python_code(*args, **kwargs)
        self.save_module_to_file(*args, **kwargs)


    def _mk_asserts(self, *args, **kwargs):
        assert os.path.exists(os.path.dirname(self.module_path)), (
                                f"Path does not exist: {os.path.dirname(self.module_path)}"
                                )


    def save_module_to_file(self, *args, module_path:str, contents:str, **kwargs) -> None:
        '''
        Creates an executable module.py file with contents at the module_path location. 
        If the module.py already exists it will be overwritten.
        
        Args:
            module_path: (str) Module module_path is a os.path.abspath to the module.py file
                Options:
                    - any file path to a module.py file
            contents: (str) Module contents to be placed below the module_doc_str
                Options:
                    -```python

                        PEP conform python module starting with a module doc string 
                        followed by PEP conform python code code
                    ```

        Returns:
            - None
        '''
        with open(module_path, "w") as f:
            f.write(contents)


    def set_module_doc_string(self, *args, module_doc_str:str, **kwargs) -> str:
        """
        The module module_doc_str is placed at the beginning of the module.py file.
        It starts with the file_name and continues with a short description of 
        the module. If no module_doc_str is provided a default is used.
        As an example, this module_doc_str can be used.

        Args:
            module_doc_str: (str) module_doc_str to be placed at the top of the module
                Options:
                    - python module_doc_str
        Returns:
            - tripple quoted module_doc_str
        """
        if module_doc_str:
            module_doc_str = module_doc_str.strip('"').strip("'")
        else:
            module_doc_str = f'# {os.path.basename(self.module_path)}\n'

        def allign_file_name(module_doc_str: str, module_name: str) -> str:
            pattern = r'^\s*#?\s*' + re.escape(module_name)
            replacement = f'\n{module_name}'
            formatted_doc_str = re.sub(pattern, replacement, module_doc_str, flags=re.MULTILINE)
            return formatted_doc_str
        module_doc_str = allign_file_name(module_doc_str, os.path.basename(self.module_path))
        # remove trailing newlines
        module_doc_str = re.sub(r'\n*$', '\n', module_doc_str, flags=re.MULTILINE)
        self.module_doc_str = f'"""{module_doc_str}"""'
        return self.module_doc_str


    def lint_python_code(self, *args, coding:str, **kwargs) -> str:
        """
        Takes the provided python code and formats it according to the requirements
        
        Args:
            self.module_doc_str: (str) module_doc_str to be placed at the top of the module
                Options:
                    - python module_doc_str
            coding: (str) coding to be placed below the module_doc_str
                Options:
                    - python coding (classes, functions)
        Returns:
            - contents: (str) module contents to be saved in the module.py file
        """
        if not coding:
            coding, module_doc_str = f"class Default:\n    pass\n", ""
        elif coding.strip().startswith('"""'):
            # then we split coding into module_doc_str and rest
            module_doc_str, coding = coding.split('"""', 2)[1:]
            # print(f"module_doc_str: {module_doc_str}")
        elif coding.strip().startswith("'''"):
            # then we split coding into module_doc_str and rest
            module_doc_str, coding = coding.split("'''", 2)[1:]
        else:
            module_doc_str = None
        module_doc_str = self.set_module_doc_string(*args, module_doc_str=module_doc_str, **kwargs)
        self.contents = f"{module_doc_str}\n\n{coding.strip()}"
        return self.contents


