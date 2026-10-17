"""The Vision Transformer: shapes, maths, and the standard neural-network sanity checks."""

import math

import pytest
import torch
import torch.nn.functional as F
from conftest import tiny_model_config

from tiny_vit.model import Attention, PatchEmbedding, VisionTransformer, attention_rollout, patchify


def images(batch: int = 2, size: int = 16, seed: int = 0) -> torch.Tensor:
    return torch.randn(batch, 3, size, size, generator=torch.Generator().manual_seed(seed))


# --- patches ----------------------------------------------------------------------


def test_patchify_cuts_row_major_patches() -> None:
    x = images(1, 8)
    patches = patchify(x, 4)  # (1, 4, 4*4*3)
    assert patches.shape == (1, 4, 48)
    # patch 1 = top-right corner; its pixels flattened as (row, col, channel)
    expected = x[0, :, 0:4, 4:8].permute(1, 2, 0).reshape(-1)
    torch.testing.assert_close(patches[0, 1], expected)


def test_conv_and_linear_patch_embeddings_compute_the_same_map() -> None:
    """A stride-P conv IS a linear layer applied to each patch (same patch order)."""
    config = tiny_model_config(patch_embed="conv")
    conv = PatchEmbedding(config).proj  # Conv2d(3, D, 4, stride=4)
    linear = torch.nn.Linear(4 * 4 * 3, config.d_model)
    # conv weight (D, C, P, P) -> linear weight over (P, P, C)-flattened patches
    linear.weight.data = conv.weight.data.permute(0, 2, 3, 1).reshape(config.d_model, -1)
    linear.bias.data = conv.bias.data
    x = images()
    from_conv = conv(x).flatten(2).transpose(1, 2)
    torch.testing.assert_close(from_conv, linear(patchify(x, 4)), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("patch_embed", ["linear", "conv"])
@pytest.mark.parametrize("pool", ["cls", "mean"])
def test_forward_shapes(patch_embed: str, pool: str) -> None:
    model = VisionTransformer(tiny_model_config(patch_embed=patch_embed, pool=pool, num_classes=5))
    assert model(images(3)).shape == (3, 5)
    tokens, attentions = model.forward_features(images(3), return_attention=True)
    assert tokens.shape == (3, 16 + 1, 32)  # 16 patches + [CLS]
    assert len(attentions) == 2 and attentions[0].shape == (3, 4, 17, 17)


def test_wrong_image_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="16x16"):
        VisionTransformer(tiny_model_config())(images(1, 20))


# --- attention --------------------------------------------------------------------


def test_explicit_attention_equals_the_fused_kernel() -> None:
    attention = Attention(tiny_model_config()).eval()
    x = torch.randn(2, 17, 32)
    fused, none = attention(x)
    explicit, weights = attention(x, return_attention=True)
    assert none is None
    torch.testing.assert_close(fused, explicit, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 17))  # rows are distributions


def test_attention_rollout_is_a_distribution_over_patches() -> None:
    model = VisionTransformer(tiny_model_config()).eval()
    rollout = attention_rollout(model.attention_maps(images(2)))
    assert rollout.shape == (2, 16)
    assert (rollout >= 0).all()
    torch.testing.assert_close(rollout.sum(-1), torch.ones(2))


def test_rollout_edge_cases() -> None:
    eye = torch.eye(5).expand(1, 2, 5, 5)  # every token attends only to itself
    torch.testing.assert_close(attention_rollout([eye, eye]), torch.zeros(1, 4))  # no NaN
    uniform = torch.full((1, 2, 5, 5), 0.2)
    torch.testing.assert_close(attention_rollout([uniform]), torch.full((1, 4), 0.25))


# --- sanity checks ----------------------------------------------------------------


def test_initial_loss_is_log_num_classes() -> None:
    """Zero-initialised head: all classes equally likely, loss = ln(C) exactly."""
    model = VisionTransformer(tiny_model_config(num_classes=7))
    loss = F.cross_entropy(model(images(8)), torch.randint(0, 7, (8,)))
    assert loss.item() == pytest.approx(math.log(7), abs=1e-6)


def test_every_parameter_gets_a_gradient_and_is_updated() -> None:
    torch.manual_seed(0)
    model = VisionTransformer(tiny_model_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(2):  # step 1 makes the zero head non-zero, so step 2 reaches every layer
        optimizer.zero_grad()
        F.cross_entropy(model(images(4)), torch.tensor([0, 1, 2, 0])).backward()
        optimizer.step()
    for name, param in model.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, f"{name} gets no gradient"
        assert not torch.equal(param, before[name]), f"{name} was not updated"


def test_samples_of_a_batch_do_not_interact() -> None:
    model = VisionTransformer(tiny_model_config()).eval()
    _shake_head(model)
    batch = images(4)
    alone = model(batch[:1])
    changed = batch.clone()
    changed[1:] = torch.randn_like(changed[1:])
    torch.testing.assert_close(model(changed)[:1], alone, atol=1e-5, rtol=1e-5)


def test_position_embeddings_make_patch_order_matter() -> None:
    torch.manual_seed(0)
    model = VisionTransformer(tiny_model_config()).eval()
    _shake_head(model)
    x = images(1)
    flipped = torch.cat([x[..., 8:, :], x[..., :8, :]], dim=-2)  # swap top and bottom halves
    assert not torch.allclose(model(x), model(flipped), atol=1e-4)


def test_dropout_is_active_in_train_mode_only() -> None:
    model = VisionTransformer(tiny_model_config(dropout=0.5))
    _shake_head(model)
    x = images(2)
    model.eval()
    torch.testing.assert_close(model(x), model(x))  # deterministic
    model.train()
    assert not torch.allclose(model(x), model(x))  # random masks


def test_gradients_match_finite_differences() -> None:
    torch.manual_seed(0)
    model = VisionTransformer(tiny_model_config(d_model=8, n_heads=2, n_layers=1)).double()
    _shake_head(model)
    x = images(2).double()
    head = model.head.weight

    def loss_of(weight: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            torch.func.functional_call(model, {"head.weight": weight}, (x,)), torch.tensor([0, 2])
        )

    assert torch.autograd.gradcheck(loss_of, (head.detach().clone().requires_grad_(),))
    # and through the whole network, w.r.t. the input pixels
    assert torch.autograd.gradcheck(lambda inp: model(inp).sum(), (x[:1].requires_grad_(),))


def test_residual_stream_stays_well_scaled_through_depth() -> None:
    torch.manual_seed(0)
    model = VisionTransformer(tiny_model_config(n_layers=12, d_model=64)).eval()
    x = model.patch_embed(images(4))
    x = torch.cat([model.cls_token.expand(4, -1, -1), x], 1) + model.pos_embed
    stds = []
    for block in model.blocks:
        x, _ = block(x)
        stds.append(x.std().item())
    assert max(stds) / min(stds) < 3, f"activations drift with depth: {stds}"


def test_parameter_groups_cover_every_parameter_once() -> None:
    model = VisionTransformer(tiny_model_config())
    decay, no_decay = model.parameter_groups(0.05)
    ids = [id(p) for p in decay["params"] + no_decay["params"]]
    assert sorted(ids) == sorted(id(p) for p in model.parameters())
    assert all(p.ndim >= 2 for p in decay["params"])
    assert id(model.cls_token) in map(id, no_decay["params"])
    assert id(model.pos_embed) in map(id, no_decay["params"])
    assert no_decay["weight_decay"] == 0.0


def test_overfits_one_batch() -> None:
    torch.manual_seed(0)
    model = VisionTransformer(tiny_model_config(num_classes=4))
    x, y = images(8), torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(80):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05
    assert (model(x).argmax(-1) == y).all()


def test_bf16_forward_is_close_to_fp32() -> None:
    model = VisionTransformer(tiny_model_config()).eval()
    _shake_head(model)
    x = images(2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        low = model(x).float()
    torch.testing.assert_close(low, model(x), atol=0.1, rtol=0.1)


def _shake_head(model: VisionTransformer) -> None:
    """The head starts at zero (constant output); give it weights so outputs differ."""
    torch.nn.init.normal_(model.head.weight, std=1.0)
