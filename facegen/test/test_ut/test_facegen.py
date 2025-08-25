# test_facegen.py
# C:\Users\lars\python_venvs\packages\facegenerator\facegen\test\test_ut\test_facegen.py

import logging
import os
import unittest
import yaml

from facegen.facegen import DefaultClass
from facegen.helpers.function_to_json import FunctionToJson
import facegen.settings as sts

class Test_DefaultClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verbose = 0

    @classmethod
    def tearDownClass(cls):
        pass

    @FunctionToJson(schemas={"openai"}, write=True)
    def test___str__(self):
        pc = DefaultClass(pr_name="facegenerator", pg_name="facegen", py_version="3.7")
        expected = "DefaultClass: self.pg_name = 'facegen'"
        self.assertEqual(str(pc), expected)
        logging.info("Info level log from the test")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
