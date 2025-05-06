from typing import List
import random
import math

LLM_STEP = 0.0125 # llm step time in ms
class GenericCall:
    arrival_time: float = 0
    idx: int = 0
    program = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)

class LLMCall(GenericCall):
    id: str
    input: str
    output: str
    
    # With prefix caching
    prefill_tokens: int = 0
    # Without prefix caching (entire conv history)
    total_prefill_tokens: int = 0
    decode_tokens: int = 0
    current_decode_token: int = 0 #Xiao: how many token has generated 
    model: str = 'Llama-7B'
    gpu_kvc = 0 #kv cache in the gpu
    cpu_kv = 0 #kv cache in the cpu
    chunk_kvcache = False #Xiao: whether the decode is chunked
        
    # Simulator variables.
    arrival_time: float = 0
    waiting_time: float = 0
    execution_time: float = 0
    finish_time: float = 0
    chunk_kv_time = 0 #Xiao: the time to offload the kv cache
    prefill_finish_time: float = 0 #Xiao: the time to finish the prefill
        
    # Total aggregate statistics
    total_waiting_time: float = None
    total_execution_time: float = None
    total_completion_time: float = None
    kv_cache_is_swap: bool = False # Xiao: when the kv cache is too long, it will swap out
    pcie_budget:int = 0 #: the pcie budget
    prefill_done: bool = False #Note: whether the prefill is done

    # Actual timing
    actual_arrival_time: int = 0
    actual_completion_time: int = 0
    
    
    # Make all arguments above init arguments
    def __init__(self, id: str, input: str, output: str, prefill_tokens: int, decode_tokens: int, total_prefill_tokens: int = -1):
        self.id = id
        self.input = input
        self.output = output
        self.prefill_tokens = prefill_tokens
        self.decode_tokens = decode_tokens
        self.total_prefill_tokens = total_prefill_tokens
        self.current_decode_token = 0 
        self.execution_time = 0
        self.waiting_time = 0
        self.total_execution_time = 0
        self.total_waiting_time = 0
        self.total_completion_time = 0
        self.iteration_time_list = []
        self.prefill_finish_time = 0

    def set_model_name(self, model_name: str):
        self.model = model_name
    
    def decode(self):
        self.current_decode_token = self.current_decode_token + 1

    def calculate_total_waiting_time(self):
        return self.waiting_time

    def calculate_total_execution_time(self):
        return self.execution_time
    
    def calculate_total_completion_time(self):
        return self.waiting_time + self.execution_time
    
    def calculate_decode_tokens(self):
        return self.decode_tokens

    def total_tokens(self):
        return self.prefill_tokens + self.decode_tokens
        
    def __repr__(self):
        return f"LLMCall(\nid={self.id}\nprefill={self.prefill_tokens},\ndecode={self.decode_tokens},\ntotal_prefill={self.total_prefill_tokens},\nwaiting_time={self.waiting_time},\nexecution_time={self.execution_time},\narrival_time={self.arrival_time},\nprefill_finish_time={self.prefill_finish_time},\nfinish_time={self.finish_time})\n"
    