import json
import os
import random

from roll.configs import ModelArguments, DataArguments, TrainingArguments

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from tqdm import tqdm

from roll.models.model_providers import default_actor_model_provider
from roll.platforms import current_platform
from tests.models.load_utils import get_mock_dataloader, get_model_input_device


def test_load_generate():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    test_batch_size = int(os.environ.get("ROLL_TEST_MODEL_BATCH_SIZE", "1"))
    test_max_batches = int(os.environ.get("ROLL_TEST_MAX_MODEL_BATCHES", "1"))
    test_max_new_tokens = int(os.environ.get("ROLL_TEST_MAX_NEW_TOKENS", "8"))

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    data_filename = "data/comparison_gpt4_data_zh.json"

    attn_implementation = "sdpa" if current_platform.is_npu() else "fa2"
    model_args: ModelArguments = ModelArguments(
        model_name_or_path=model_name,
        attn_implementation=attn_implementation,
        dtype="bf16",
    )
    data_args: DataArguments = DataArguments(
        template="qwen2_5",
        file_name=data_filename,
        prompt="instruction",
    )

    dataloader, tokenizer = get_mock_dataloader(model_args=model_args, data_args=data_args, batch_size=test_batch_size)

    model = default_actor_model_provider(tokenizer, model_args, TrainingArguments(), False)

    results = []
    for batch in tqdm(dataloader):
        input_device = get_model_input_device(model)
        input_ids = batch["input_ids"].to(input_device)
        attention_mask = batch["attention_mask"].to(input_device)
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=test_max_new_tokens,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

        output_str = tokenizer.batch_decode(output, skip_special_tokens=False)
        results.append(output_str)

        with open("generate_res.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        if len(results) >= test_max_batches:
            break


if __name__ == "__main__":
    test_load_generate()
