import unittest

from subspaces import VectorSpace
from subspaces.generators import IdentityGenerator


# --- unittests
class TestIdentityGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = IdentityGenerator()

    def mock_test(self):
        pass


if __name__ == "__main__":
    unittest.main()
