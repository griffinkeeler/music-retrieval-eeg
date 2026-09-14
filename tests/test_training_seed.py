import random
import unittest

import numpy as np
import torch

from scripts.train import seed_training


class TrainingSeedTests(unittest.TestCase):
    def test_seed_training_reproduces_python_numpy_and_torch_randomness(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        try:
            seed_training(17)
            first = (
                random.random(),
                np.random.random(),
                torch.rand(3),
            )

            seed_training(17)
            second = (
                random.random(),
                np.random.random(),
                torch.rand(3),
            )

            self.assertEqual(first[0], second[0])
            self.assertEqual(first[1], second[1])
            torch.testing.assert_close(first[2], second[2])
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)


if __name__ == "__main__":
    unittest.main()
