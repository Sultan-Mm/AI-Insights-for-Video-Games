import pytest
import os


from project.params import *


class TestParams():
    def test_MODEL_TARGET(self):

        assert os.getenv("MODEL_TARGET") , "MODEL_TARGET environment variable not defined"
