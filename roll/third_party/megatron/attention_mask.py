import torch


def build_causal_padding_mask(token_attention_mask: torch.Tensor) -> torch.Tensor:
    """Convert a token-validity mask into Megatron's boolean attention mask.

    ``token_attention_mask`` uses ``True``/``1`` for valid tokens, while
    Megatron's four-dimensional boolean mask uses ``True`` for blocked
    attention positions.
    """
    if token_attention_mask.ndim != 2:
        raise ValueError(
            "token_attention_mask must be two-dimensional [batch, sequence], "
            f"got shape {tuple(token_attention_mask.shape)}"
        )

    valid_tokens = token_attention_mask.bool()
    sequence_length = valid_tokens.shape[1]
    causal_mask = torch.triu(
        torch.ones(
            (1, 1, sequence_length, sequence_length),
            dtype=torch.bool,
            device=valid_tokens.device,
        ),
        diagonal=1,
    )
    padding_mask = ~valid_tokens[:, None, None, :]
    return causal_mask | padding_mask
