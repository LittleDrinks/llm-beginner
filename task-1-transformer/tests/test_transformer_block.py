import unittest
import torch
import torch.nn as nn
from src.attention import MultiHeadAttention
from src.block import TransformerBlock, LayerNorm


class TestTransformerBlock(unittest.TestCase):
    def test_mha_output_shape(self):
        B, T, H, D = 2, 3, 2, 8
        mha = MultiHeadAttention(D, H)
        X = torch.randn(B, T, D)
        O = mha(X)
        same = O.shape == (B, T, D)
        return {"test": "mha_output_mask", "pass": same}

    def test_forward_preserve_shape(self):
        torch.manual_seed(8)
        block = TransformerBlock(d_model=8, num_heads=2)
        x = torch.randn(2, 4, 8)
        y = block(x)
        self.assertEqual(x.shape, y.shape)

    def test_output_is_independent_of_other_batch(self):
        torch.manual_seed(8)
        block = TransformerBlock(8, 2)
        a = torch.randn(1, 4, 8)
        b = torch.randn(1, 4, 8) * 20 + 100
        with torch.no_grad():
            x = block(a)
            y = block(torch.cat([a, b], dim=0))
        torch.testing.assert_close(x, y[:1], atol=1e-6, rtol=1e-5)

    def test_layer_norm(self):
        torch.manual_seed(8)
        x = torch.randn(2, 4, 8)
        LN = LayerNorm(8)
        torch.testing.assert_close(
            LN(x),
            nn.functional.layer_norm(x, (x.shape[-1],), eps=1e-5),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_layer_norm_0var(self):
        x = torch.full((2, 4, 8), 3.0)
        LN = LayerNorm(8)
        torch.testing.assert_close(
            LN(x),
            nn.functional.layer_norm(x, (x.shape[-1],), eps=1e-5),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_layer_norm_affine_parameters(self):
        LN = LayerNorm(4)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        gamma = torch.tensor([2.0, 1.0, -0.5, 3.0])
        beta = torch.tensor([1.0, -2.0, 0.5, 4.0])

        with torch.no_grad():
            LN.gamma.copy_(gamma)
            LN.beta.copy_(beta)

        torch.testing.assert_close(
            LN(x),
            nn.functional.layer_norm(
                x, (x.shape[-1],), weight=gamma, bias=beta, eps=1e-5
            ),
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
