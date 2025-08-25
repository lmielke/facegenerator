import ast
import os
import graphviz
import argparse

ignore_dirs = {
                ".git",
                "build",
                "gp",
                "dist",
                "models",
                "*.egg-info",
                "__pycache__",
                ".pytest_cache",    
                ".tox",
}

class PackageInfo:
    def __init__(self, main_file: str, *args, **kwargs):
        self.root_dir, self.package_name = self.find_root_dir(*args, **kwargs)
        if not self.root_dir:
            raise RuntimeError("Root directory not found.")
        self.main_file = main_file
        self.graph = graphviz.Digraph(comment='Package Dependency Graph')
        self.visited_files = set()
        self.incoming_edges = {}  # Track incoming edges for each node
        # Set default styles for the graph
        self.graph.attr('node', style='filled', fillcolor='white')
        self.graph.attr('edge', fontsize='10')  # Smaller font size for edges

    def find_root_dir(self):
        """
        Determine the root directory of a Python project by locating __main__.py.
        """
        for root, dirs, files in os.walk(os.getcwd()):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            if '__main__.py' in files:
                return os.path.split(root)
        return None

    def build_graph(self, filepath):
        filename = os.path.basename(filepath)
        if filename in self.visited_files:
            return
        self.visited_files.add(filename)

        # Initialize incoming edges count if not already
        if filename not in self.incoming_edges:
            self.incoming_edges[filename] = 0

        imports = self.parse_imports(filepath)
        for imp, origin in imports:
            next_file = self.resolve_module_path_to_file(imp)
            if next_file:
                next_filename = os.path.basename(next_file)
                # Increment the incoming edge count for the target node
                if next_filename not in self.incoming_edges:
                    self.incoming_edges[next_filename] = 0
                self.incoming_edges[next_filename] += 1

                self.graph.edge(filename, next_filename, label=imp)
                self.build_graph(next_file)

    def finalize_graph(self):
        # Determine the maximum number of incoming edges for scaling
        max_edges = max(self.incoming_edges.values(), default=1)
        
        # Set node attributes based on incoming edges
        for node, count in self.incoming_edges.items():
            # Scale the fontsize according to the number of incoming edges
            fontsize = '12' if node == self.main_file else str(10 + min(count * 2, 10))
            
            # Calculate the intensity of the red color based on incoming edges
            if max_edges > 0:
                intensity = int(255 * (1 - (count / max_edges)))  # Adjust for a proper scale
            else:
                intensity = 255  # No edges lead to lightest color
            
            fillcolor = f'#{255-intensity:02x}{intensity//1:02x}{intensity//1:02x}'  # Adjusting for lighter shades
            
            # Adjusting the fill color and font size
            self.graph.node(node, fontsize=fontsize, fillcolor=fillcolor, style='filled')
            self.graph.node(node, fontsize=fontsize, fillcolor='lightblue' if node == self.main_file else fillcolor)

    def create_graph(self):
        """
        Start the graph creation process from the specified main file.
        """
        main_path = self.locate_file(self.main_file, self.root_dir)
        if not main_path:
            raise FileNotFoundError(f"{self.main_file} not found in {self.root_dir}")
        
        self.build_graph(main_path)
        self.finalize_graph()  # Apply final node styles based on connectivity
        return self.graph

    def parse_imports(self, filepath):
        """
        Parse a Python file and extract all local import statements relevant to the package.
        """
        with open(filepath, 'r') as file:
            tree = ast.parse(file.read(), filepath)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name.startswith(self.package_name):
                        imports.append((alias.name, os.path.relpath(filepath, self.root_dir).replace(os.sep, '.')))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(self.package_name):
                    for alias in node.names:
                        full_import_path = f"{node.module}.{alias.name}"
                        imports.append((full_import_path, os.path.relpath(filepath, self.root_dir).replace(os.sep, '.')))
        return imports

    def resolve_module_path_to_file(self, module_path):
        """
        Resolve a dot-separated module path to a file path by checking each segment of the path.
        This method iteratively shortens the path from the rightmost segment until a valid file is found.
        """
        module_parts = module_path.split('.')
        for i in range(len(module_parts), 0, -1):
            potential_path = os.path.join(self.root_dir, *module_parts[:i]) + '.py'
            potential_rel = os.path.relpath(potential_path, self.root_dir)
            if os.path.exists(potential_path):
                return potential_path
            else:
                pass
        return None

    def locate_file(self, filename, search_dir):
        """
        Recursively locate a specific file within a given directory.
        """
        for root, dirs, files in os.walk(search_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

def set_params(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Analyze Python package structure and visualize import relationships.")
    parser.add_argument('main_file_name', type=str, help='The name of the main Python file to trace imports from.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbose mode. If set to 1 or higher, the graph is displayed.')
    return parser.parse_args().__dict__

def main(*args, verbose=1, **kwargs):
    if not kwargs:
        kwargs = set_params(*args, **kwargs)
    package_info = PackageInfo(kwargs['main_file_name'])
    graph = package_info.create_graph()
    dot_source = graph.source
    if verbose:
        graph.view()  # This will open the graph using the default viewer
    return dot_source

if __name__ == '__main__':
    main()
