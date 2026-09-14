import unittest

import torch

from src.encoders.eeg_encoder import SubjectLayer


class SubjectLayerTests(unittest.TestCase):
    def test_initial_transform_is_identity_for_every_subject(self):
        layer = SubjectLayer(n_subjects=3, channels=4)
        inputs = torch.randn(3, 4, 7)
        subject_ids = torch.tensor([0, 1, 2])

        outputs = layer(inputs, subject_ids)

        torch.testing.assert_close(outputs, inputs)

    def test_adamw_keeps_unseen_subject_transform_at_identity(self):
        layer = SubjectLayer(n_subjects=2, channels=3)
        optimizer = torch.optim.AdamW(
            layer.parameters(),
            lr=3e-4,
            weight_decay=0.01,
        )
        seen_inputs = torch.randn(2, 3, 5)

        loss = layer(seen_inputs, torch.zeros(2, dtype=torch.long)).sum()
        loss.backward()
        optimizer.step()

        torch.testing.assert_close(
            layer.delta[1],
            torch.zeros_like(layer.delta[1]),
        )
        unseen_input = torch.randn(1, 3, 5)
        unseen_output = layer(unseen_input, torch.tensor([1]))
        torch.testing.assert_close(unseen_output, unseen_input)


if __name__ == "__main__":
    unittest.main()
