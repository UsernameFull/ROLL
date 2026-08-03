import pytest
import torch

from roll.third_party.megatron.attention_mask import build_causal_padding_mask


def test_build_causal_padding_mask_for_all_valid_tokens():
    token_mask = torch.ones((1, 3), dtype=torch.long)

    result = build_causal_padding_mask(token_mask)

    expected = torch.tensor([[[[False, True, True], [False, False, True], [False, False, False]]]])
    assert result.dtype == torch.bool
    assert torch.equal(result, expected)


def test_build_causal_padding_mask_blocks_padding_keys():
    token_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    result = build_causal_padding_mask(token_mask)

    assert result.shape == (2, 1, 3, 3)
    assert result[0, 0, :, 2].all()
    assert result[1, 0, :, 1:].all()
    assert not result[0, 0, 1, 0]
    assert not result[1, 0, 2, 0]


def test_build_causal_padding_mask_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        build_causal_padding_mask(torch.ones(3))
