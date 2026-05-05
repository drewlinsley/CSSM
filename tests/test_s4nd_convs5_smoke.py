"""Forward-pass smoke tests for the new S4ND and ConvS5 CSSM variants.

Runs on CPU. Confirms both variants:
  - instantiate correctly through SimpleCSSM
  - accept 5D (B, T, H, W, C) input at T=1 (pathfinder-style) and T=4 (video)
  - return (B, num_classes) logits
"""

import jax
import jax.numpy as jnp
import pytest

from src.models.simple_cssm import SimpleCSSM


@pytest.mark.parametrize("cssm_type", ["s4nd", "convs5"])
@pytest.mark.parametrize("T", [1, 4])
def test_forward_shape(cssm_type, T):
    model = SimpleCSSM(
        num_classes=2,
        embed_dim=16,
        depth=1,
        cssm_type=cssm_type,
        seq_len=T,
        max_seq_len=T,
        stem_mode='pathtracker',
        pos_embed='none',
        pool_type='max',
        frame_readout='last',
        s4nd_d_state=16,
        convs5_state_dim=8,
    )
    x = jnp.zeros((2, T, 32, 32, 3))
    params = model.init(jax.random.PRNGKey(0), x, training=False)
    y = model.apply(params, x, training=False)
    assert y.shape == (2, 2), f"{cssm_type} T={T}: expected (2, 2), got {y.shape}"


def test_s4nd_bidirectional_flag():
    """Both directions of the bidi flag must build and run."""
    for bidi in (True, False):
        model = SimpleCSSM(
            num_classes=2, embed_dim=16, depth=1, cssm_type='s4nd',
            seq_len=1, max_seq_len=1,
            stem_mode='pathtracker', pos_embed='none', pool_type='max',
            frame_readout='last',
            s4nd_d_state=16, s4nd_bidirectional=bidi,
        )
        x = jnp.zeros((1, 1, 32, 32, 3))
        params = model.init(jax.random.PRNGKey(0), x, training=False)
        y = model.apply(params, x, training=False)
        assert y.shape == (1, 2)


def test_convs5_kernel_sizes():
    """ConvS5 should build for different kernel sizes."""
    for ks in (3, 5):
        model = SimpleCSSM(
            num_classes=2, embed_dim=16, depth=1, cssm_type='convs5',
            seq_len=4, max_seq_len=4,
            stem_mode='pathtracker', pos_embed='none', pool_type='max',
            frame_readout='last',
            kernel_size=ks, convs5_state_dim=8,
        )
        x = jnp.zeros((1, 4, 32, 32, 3))
        params = model.init(jax.random.PRNGKey(0), x, training=False)
        y = model.apply(params, x, training=False)
        assert y.shape == (1, 2)
