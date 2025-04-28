GPUMEM = 80 * 1024 * 1024 * 1024 #80GB A100/H100

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
import tiktoken

#FP16
model_weights ={
    "llama3_1_8B": 16 * 1024 * 1024 * 1024 ,
    "llama2_13B" : 26 * 1024 * 1024 * 1024,
    "qwq_32B": 64 * 1024 * 1024 * 1024 ,
    "llama3_1_70B": 140 * 1024 * 1024 * 1024 , #FIXME(Xiao): may add 30B 
}

KVC_THRESHOLD = 4000 
MAX_OFFLOAD_CALL = 20 

def _num_token_from_text(text: str, model: str = "gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text, allowed_special={'<|endoftext|>'}))


def _num_token_from_messages(messages: Union[List, Dict], model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages.

    Retrieved from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb/
    """
    if isinstance(messages, dict):
        messages = [messages]

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if "gpt-4" in model or "gpt-3.5" in model:
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""_num_token_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value is None:
                continue

            # function calls
            if not isinstance(value, str):
                try:
                    value = json.dumps(value)
                except TypeError:
                    continue
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_tokens(input: Union[str, List, Dict], model: str = "gpt-4") -> int:
    """Count number of tokens used by an OpenAI model.
    Args:
        input: (str, list, dict): Input to the model.
        model: (str): Model name.

    Returns:
        int: Number of tokens from the input.
    """
    if isinstance(input, str):
        return _num_token_from_text(input, model=model)
    elif isinstance(input, list) or isinstance(input, dict):
        return _num_token_from_messages(input, model=model)
    else:
        raise ValueError(f"Input must be str, list or dict, but received {type(input)}")
    
def TODO(func_name):
    assert False, f"TODO:{func_name}"