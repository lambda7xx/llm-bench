import json
import os
from typing import List, Dict
import math
import numpy as np
import copy
import uuid
from typing import List, Tuple
import json
import random
import os, sys
import argparse
import tqdm
import pandas as pd
import numpy as np
import dill as pickle
import sys

from llm_bench.llm_calls import LLMCall
from llm_bench.util import count_tokens

import uuid 

sys.setrecursionlimit(10000)


SHAREGPT_PKL = 'sharegpt.pkl'
LEVAL_PKL = 'leval.pkl'
LOAD_DATA_COLLECTIONS = 'long_data_collections.pkl'
ARXIV_SUMMARY="arxiv_summary.pkl"


def calculate_std(data):
    n = len(data)
    avg = sum(data) / n
    variance = sum((x - avg) ** 2 for x in data) / n
    std = math.sqrt(variance)
    return std


#git lfs clone https://huggingface.co/datasets/ccdv/arxiv-summarization/
def load_arxiv_summary():  
    if os.path.exists(ARXIV_SUMMARY):
        with open(ARXIV_SUMMARY, 'rb') as f:
            return pickle.load(f)
    files = []
    dataset_path = "/root/autodl-tmp/data/arxiv-summarization/document" #Note(Xiao): this is hacking way to get the path
    for root, dirs, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.startswith("train"):
                files.append(os.path.join(root, filename))
    dataset_path = "/root/autodl-tmp/data/arxiv-summarization/section" #Note(Xiao): this is hacking way to get the path

    for root, dirs, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.startswith("train"):
                files.append(os.path.join(root, filename))

    calls = []
    inputs = []
    outputs = []
    small = 2000 
    large = 20000
    #NOTE(Xiao):this can change
    decode_small = 100 
    decode_large = 2000
    prefill1 = 4000
    prefill2 = 7000
    prefill1_list = [] 
    prefill2_list = []
    for file in files:
        df = pd.read_parquet(file)
        for index, row in df.iterrows():
            document = row['article']
            abstract = row['abstract']
            prompt = document
            prefill_token = count_tokens(prompt)
            decode_tokens = count_tokens(abstract)
            #prefill 2000 ~ 10000 or decode 2000~10000
            if not (( small <= prefill_token <= large)):
                continue
            if not ((decode_small <= decode_tokens <= decode_large)):
                continue   
            if small <= prefill_token <= prefill1:
                prefill1_list.append(prefill_token)
            if prefill1 < prefill_token <= prefill2:
                prefill2_list.append(prefill_token)
            print(f"prefill_token: {prefill_token} and decode_tokens: {decode_tokens}")
            inputs.append(prefill_token)
            outputs.append(decode_tokens)
            calls.append(
                LLMCall(
                    id = str(uuid.uuid4()),
                    prefill_tokens = prefill_token,
                    decode_tokens = decode_tokens,
                    input = prompt,
                    output = ""
                )
            )
    print(f"len(prefill1_list)(2000~4000): {len(prefill1_list)} and len(prefill2_list)(4000~7000): {len(prefill2_list)} and len(calls):{len(calls)}", flush=True)
    inputs = sorted(inputs)
    outputs = sorted(outputs)
    std_input  = calculate_std(inputs)
    avg_input = np.mean(inputs)
    avg_output = np.mean(outputs)
    p50_input = np.percentile(inputs, 50)
    p50_output = np.percentile(outputs, 50)
    p90_input = np.percentile(inputs, 90)
    p90_output = np.percentile(outputs, 90)
    std_output = calculate_std(outputs)
    print(f"std_input: {std_input} and std_output: {std_output}")
    print(f"avg_input: {avg_input} and avg_output: {avg_output}")
    print(f"p50_input: {p50_input} and p50_output: {p50_output}")
    with open(ARXIV_SUMMARY, 'wb') as f:
        pickle.dump(calls, f)
    return calls

# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
def load_sharegpt_traces(trace_file: str = 'ShareGPT_V3_unfiltered_cleaned_split.json', model_name=None) -> List[Dict]:
    trace_file = "./ShareGPT_V3_unfiltered_cleaned_split.json" #Xiao: this is hacking way to get the path
    trace_file = os.path.expanduser(trace_file)
    if os.path.exists(SHAREGPT_PKL):
        with open(SHAREGPT_PKL, 'rb') as f:
            return pickle.load(f)
    with open(trace_file, 'r') as f:
        traces = json.load(f)
    calls = []
    inputs = []
    outputs = []
    while traces:
        trace = traces.pop()
        id = trace['id'].split('_')[0]
        conversations = trace["conversations"]
        while traces and traces[0]['id'].split('_')[0] == id:
            trace = traces.pop(0)
            conversations.extend(trace['conversations'])
        if len(conversations) < 2:
            continue
        human_input = conversations.pop(0)
        gpt_output = conversations.pop(0)
        prefill_tokens = count_tokens(human_input['value'])
        decode_tokens = count_tokens(gpt_output['value'])
        if prefill_tokens >= 1500 or decode_tokens >= 500:
            continue 
        if prefill_tokens <= 200 or decode_tokens <=10:
            continue 
        inputs.append(prefill_tokens)
        outputs.append(decode_tokens)
        print(f"prefill_tokens:{prefill_tokens} and decode_tokens:{decode_tokens}")
        if model_name is not None:
            if (prefill_tokens + decode_tokens) >= 50000:
                continue 
        if  decode_tokens <=1:
            continue 
        llm_call = LLMCall(
            id = id,
            prefill_tokens = prefill_tokens,
            decode_tokens = decode_tokens,
            input = human_input['value'],
            output = gpt_output['value'],
        )
        calls.append(llm_call)
    inputs = sorted(inputs)
    outputs = sorted(outputs)
    std_input  = calculate_std(inputs)
    std_output = calculate_std(outputs)
    metadata = []
    metadata.append((np.mean(inputs), std_input, np.percentile(inputs, 90), np.percentile(inputs, 99), np.mean(outputs), std_output, np.percentile(outputs, 90), np.percentile(outputs, 99)))


    #Save to file programs
    with open(SHAREGPT_PKL, 'wb') as f:
        pickle.dump(calls, f)
    return calls

#Xiao: long dataset, prompt:5k~30k or decode token 5k~30k
#https://github.com/LoongServe/LoongServe/blob/main/test/longserve/4-preprocess-dataset.py
def load_lonngserve_dataset(
    name: str = "leval"):  
    dataset_path = "./LEval/LEval-data" #Xiao: this is hacking way to get the path
    dataset_path = os.path.expanduser(dataset_path)
    if name == "leval":
        if os.path.exists(LEVAL_PKL):
            with open(LEVAL_PKL, 'rb') as f:
                return pickle.load(f)
        files = []
        for root, dirs, filenames in os.walk(dataset_path):
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    files.append(os.path.join(root, filename))
        num_lines = sum([
            sum([
                len(json.loads(line)["instructions"])
                for line in open(filename, "r").readlines()
            ])
            for filename in files
        ])
        calls  = []
        pbar = tqdm.tqdm(total=num_lines)
        inputs = []
        outputs = []
        small = 4000
        large = 10000
        decode_threshold1= 100
        decode_threshold2 = 500
        for file in files:
            with open(file, "r") as f:
                for line in f.readlines():
                    if line.strip() == "": continue
                    data = json.loads(line)
                    input = data["input"]
                    for (instruction, output) in zip(data["instructions"], data["outputs"]):
                        prompt = input + instruction
                        prefill_token = count_tokens(prompt)
                        decode_token = count_tokens(output)
                        #prefill 2000 ~ 10000 or decode 2000~10000
                        if not ( small <= prefill_token <= large and decode_threshold1 <= decode_token <= decode_threshold2):
                            continue
                        inputs.append(prefill_token)
                        outputs.append(decode_token)
                        calls.append(LLMCall(
                            id = str(uuid.uuid4()),
                            prefill_tokens = prefill_token,
                            decode_tokens = decode_token,
                            input = prompt,
                            output = output
                        ))
                        pbar.update(1)
        pbar.close()
        inputs = sorted(inputs)
        outputs = sorted(outputs)
        std_input  = calculate_std(inputs)
        std_output = calculate_std(outputs)
        metadata = []
        metadata.append((np.mean(inputs), std_input, np.percentile(inputs, 90), np.percentile(inputs, 99), np.mean(outputs), std_output, np.percentile(outputs, 90), np.percentile(outputs, 99)))
        with open(LEVAL_PKL, 'wb') as f:
            pickle.dump(calls, f)
        return calls 

#wget https://huggingface.co/datasets/togethercomputer/Long-Data-Collections/resolve/main/fine-tune/natural_questions_10_200_docs.jsonl.zst
# zstd -d natural_questions_10_200_docs.jsonl.zst
#Xiao: long dataset, prompt:5k~30k or decode token 5k~30k
#Long-Data-Collections: avg input:5545.622961428941 and std_input:1421.5441673262058 and p90 input:7489.0 and p99 input:7950.0 
#Long-Data-Collections: avg output:173.623996893606 and std_output:175.06535842152365 and p90 output:278.0 and p99 output:1066.0
def load_long_data_collections():
    if os.path.exists(LOAD_DATA_COLLECTIONS):
        with open(LOAD_DATA_COLLECTIONS, 'rb') as f:
            return pickle.load(f)
    path = "/root/autodl-tmp/natural_questions_10_200_docs.jsonl" #Xiao: this is hacking way to get the path
    with open(path, "r") as f:
        lines = f.readlines()
    calls = []
    inputs = []
    outputs = []
    metadata = [] 
    small = 4000
    large = 10000
    decode_threshold1= 100
    decode_threshold2 = 500
    for line in lines:
        line = json.loads(line)
        assert isinstance(line, dict)
        text = line["text"]
        prompt = line["prompt"] 
        output = line["completion"]
        prompt = text + prompt
        prefill_token = count_tokens(prompt)
        decode_token = count_tokens(output)
        if decode_token == 0:
            continue
        #prefill 2000 ~ 10000 or decode 2000~10000
        if not ( small <= prefill_token <= large and decode_threshold1 <= decode_token <= decode_threshold2):
            continue

        inputs.append(prefill_token)
        outputs.append(decode_token)
        print(f"prefill_token:{prefill_token} and decode_token:{decode_token}", flush=True)
        calls.append(LLMCall(
                id = str(uuid.uuid4()),
                prefill_tokens = prefill_token,
                decode_tokens = decode_token,
                input = prompt,
                output = output
            )
        )
    with open(LOAD_DATA_COLLECTIONS, 'wb') as f:
        pickle.dump(calls, f)

    inputs = sorted(inputs)
    outputs = sorted(outputs)
    std_input  = calculate_std(inputs)
    std_output = calculate_std(outputs)
    metadata.append((np.mean(inputs), std_input, np.percentile(inputs, 90), np.percentile(inputs, 99), np.mean(outputs), std_output, np.percentile(outputs, 90), np.percentile(outputs, 99)))

#generate short reqs(prefill + decodes: hundreds of tokens), from sharegpt 
#long reqs:(prefil: thousands of tokens, decode:  thousands of tokens)
#long reqs are the translation tasks
def mixserve_generate_mixed_reqs_from_sharegpt_leval_long_data_collect(
        arrival_rate = 1.0,
        cv_factor=1.0,
        total_jobs = 1000,
        arrival_period = None,
        seed = 2025,
        ratio : float = 0.5,
        quantized: bool=True,
        model_name: str="llama3_1_8B"
    ):
    print(f"mixed_sharegpt_long_data_collections_leval")
    MIXED_SHAREGPT_LONG_DATA_COLLECT = f"translate_mixed_tasks_{total_jobs}_ratio_{ratio}_arrival_period_{arrival_period}_arrival_rate_{arrival_rate}.pkl"
    if os.path.exists(MIXED_SHAREGPT_LONG_DATA_COLLECT):
        with open(MIXED_SHAREGPT_LONG_DATA_COLLECT, 'rb') as f:
            return pickle.load(f)
    sharegpt_calls = load_sharegpt_traces(model_name = model_name)
    long_data_collect = load_long_data_collections()
    leval_dataset_path = "LEval/LEval-data"
    leval_name = "leval"
    leval_calls = load_lonngserve_dataset()
    system_prompt = "please help me translate the following text into Chinese."
    print(f"len(sharegpt_calls): {len(sharegpt_calls)} and len(long_data_collect): {len(long_data_collect)} and len(leval_calls): {len(leval_calls)}", flush=True)
    prefill_threshold1 = 2500
    prefill_threshold2 = 1000
    decode_threshold1 = 2500 
    decode_threshold2 = 75000 
    new_long_data_collect = []
    for llm_call in long_data_collect:
        if llm_call.prefill_tokens > prefill_threshold2 or llm_call.prefill_tokens < prefill_threshold1:
            continue
        #generate the translation task
        llm_call.input = system_prompt + llm_call.input
        llm_call.prefill_tokens = count_tokens(llm_call.input)
        llm_call.decode_tokens = np.random.randint(decode_threshold1, decode_threshold2)
        new_long_data_collect.append(llm_call)
    long_data_collect = new_long_data_collect
    new_leval_calls = []
    for llm_call in leval_calls:
        if llm_call.prefill_tokens > prefill_threshold2 or llm_call.prefill_tokens < prefill_threshold1:
            continue
        #generate the translation task
        llm_call.input = system_prompt + llm_call.input
        llm_call.prefill_tokens = count_tokens(llm_call.input)
        llm_call.decode_tokens = np.random.randint(decode_threshold1, decode_threshold2)
        new_leval_calls.append(llm_call)

    leval_calls = new_leval_calls 

    long_data_collect = new_long_data_collect
    new_sharegpt_size = int(total_jobs * ratio)
    new_long_data_collect_size = int(total_jobs - new_sharegpt_size)
    if new_long_data_collect_size > len(long_data_collect):
        new_long_data_collect_size = len(long_data_collect)
        new_leval_size = total_jobs - new_sharegpt_size - new_long_data_collect_size
    else:
        new_leval_size = 0
    print(f"2 len(sharegpt_calls): {len(sharegpt_calls)} and len(long_data_collect): {len(long_data_collect)} and len(leval_calls): {len(leval_calls)}", flush=True)
    print(f"3 len(new_sharegpt_calls): {new_sharegpt_size} and len(new_long_data_collect): {new_long_data_collect_size} and len(new_leval_calls): {new_leval_size}", flush=True)
    # Set the random seed
    np.random.seed(seed)
    alpha = (1.0 / cv_factor)**2
    if new_leval_size == 0:
        new_sharegpt_calls = random.sample(sharegpt_calls, new_sharegpt_size)
        new_long_data_collect_calls = random.sample(long_data_collect, new_long_data_collect_size)
    else:
        new_sharegpt_calls = random.sample(sharegpt_calls, new_sharegpt_size)
        new_long_data_collect_calls = long_data_collect
        new_leval_calls = random.sample(leval_calls, new_leval_size)
        new_long_data_collect_calls.extend(new_leval_calls)

    # Generate arrival times
    if arrival_period is not None:
        # Generate interarrival times until current_time >= arrival_period
        interarrival_times = []
        arrival_times = []
        current_time = 0
        while current_time < arrival_period:
            interarrival_time = np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
            interarrival_times.append(interarrival_time)
            current_time += interarrival_time
            arrival_times.append(current_time)
        arrival_times = np.array(arrival_times)
        # Exclude the last arrival time if it exceeds arrival_period
        if arrival_times[-1] > arrival_period:
            arrival_times = arrival_times[:-1]
        total_jobs = len(arrival_times)
    else:
        # Generate a fixed number of interarrival times
        interarrival_times = np.array([
            np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
            for _ in range(total_jobs - 1)
        ])
        interarrival_times = np.insert(interarrival_times, 0, 0)
        arrival_times = np.cumsum(interarrival_times)
    

    mixed_calls = new_sharegpt_calls + new_long_data_collect_calls
    random.shuffle(mixed_calls)
    mixed_calls = mixed_calls[:total_jobs]
    
    #Assigin arrival time
    for idx, llm_call in enumerate(mixed_calls):
        if quantized:
            llm_call.arrival_time = math.ceil(arrival_times[idx])
        else:
            llm_call.arrival_time = arrival_times[idx]

    for llm_call in mixed_calls:
        uid = str(uuid.uuid4())
        llm_call.id = uid
    print(f"3 len(mixed_calls): {len(mixed_calls)} and total_jobs:{total_jobs}", flush=True)
    #save the mixed_calls into a file
    with open(MIXED_SHAREGPT_LONG_DATA_COLLECT, 'wb') as f:
        pickle.dump(mixed_calls, f)
    return mixed_calls

def hybridserve_sharegpt(
        arrival_rate = 1.0,
        cv_factor=1.0,
        total_jobs = 1000,
        arrival_period = None,
        seed = 2025,
        ratio : float = 0.5,
        quantized: bool=True,
        model_name: str="llama3_1_8B"
    ):
    print(f"mixed_sharegpt_long_data_collections_leval")
    MIXED_SHAREGPT_LONG_DATA_COLLECT = f"sharegpt_{total_jobs}.0_ratio_{ratio}_arrival_period_{arrival_period}_arrival_rate_{arrival_rate}.pkl"
    if os.path.exists(MIXED_SHAREGPT_LONG_DATA_COLLECT):
        print(f"1 os.path.exists(MIXED_SHAREGPT_LONG_DATA_COLLECT):{MIXED_SHAREGPT_LONG_DATA_COLLECT}", flush=True)
        with open(MIXED_SHAREGPT_LONG_DATA_COLLECT, 'rb') as f:
            return pickle.load(f)
    sharegpt_calls =load_sharegpt_traces(model_name = model_name)
    long_data_collect = []
    leval_dataset_path = "LEval/LEval-data"
    leval_name = "leval"
    leval_calls = [] #load_lonngserve_dataset()

    new_sharegpt_size = int(total_jobs)

    print(f"2 len(sharegpt_calls): {len(sharegpt_calls)} and len(long_data_collect): {len(long_data_collect)} and len(leval_calls): {len(leval_calls)}", flush=True)
    # Set the random seed
    np.random.seed(seed)
    alpha = (1.0 / cv_factor)**2
    new_sharegpt_calls = random.sample(sharegpt_calls, new_sharegpt_size)
    # Generate arrival times
    if arrival_period is not None:
        # Generate interarrival times until current_time >= arrival_period
        interarrival_times = []
        arrival_times = []
        current_time = 0
        while current_time < arrival_period:
            interarrival_time = np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
            interarrival_times.append(interarrival_time)
            current_time += interarrival_time
            arrival_times.append(current_time)
        arrival_times = np.array(arrival_times)
        # Exclude the last arrival time if it exceeds arrival_period
        if arrival_times[-1] > arrival_period:
            arrival_times = arrival_times[:-1]
        total_jobs = len(arrival_times)
    else:
        # Generate a fixed number of interarrival times
        interarrival_times = np.array([
            np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
            for _ in range(total_jobs - 1)
        ])
        interarrival_times = np.insert(interarrival_times, 0, 0)
        arrival_times = np.cumsum(interarrival_times)
    

    mixed_calls = new_sharegpt_calls 
    random.shuffle(mixed_calls)
    mixed_calls = mixed_calls[:total_jobs]
    
    #Assigin arrival time
    for idx, llm_call in enumerate(mixed_calls):
        if quantized:
            llm_call.arrival_time = math.ceil(arrival_times[idx])
        else:
            llm_call.arrival_time = arrival_times[idx]

    for llm_call in mixed_calls:
        uid = str(uuid.uuid4())
        llm_call.id = uid
    print(f"3 len(mixed_calls): {len(mixed_calls)} and total_jobs:{total_jobs}", flush=True)
    #save the mixed_calls into a file
    with open(MIXED_SHAREGPT_LONG_DATA_COLLECT, 'wb') as f:
        pickle.dump(mixed_calls, f)
    return mixed_calls

def skew_hybridserve_arxiv(
        arrival_rate=1.0,
        cv_factor=1.0,
        total_jobs=1000,
        arrival_period=None,
        seed=2025,
        ratio: float = 0.5,
        quantized: bool = True,
        model_name: str = "llama3_1_8B"
):
    print(f"skew_hybridserve_arxiv")
    MIXED_ARXIV = f"skew_arxiv_{total_jobs}.0_ratio_{ratio}_arrival_period_{arrival_period}_arrival_rate_{arrival_rate}.pkl"
    if os.path.exists(MIXED_ARXIV):
        print(f"1 os.path.exists(MIXED_SHAREGPT_LONG_DATA_COLLECT):{MIXED_ARXIV}", flush=True)
        with open(MIXED_ARXIV, 'rb') as f:
            return pickle.load(f)

    sharegpt_calls = []  # load_sharegpt_traces(model_name=model_name)
    long_data_collect = load_arxiv_summary()
    leval_calls = []  # load_leval_dataset()

    decode_threshold1 = 100
    decode_threshold2 = 500
    prefill_threshold1 = 4000
    prefill_threshold2 = 7000
    # short = [] 
    # long = []
    # for llm_call in long_data_collect:
    #     if llm_call.prefill_tokens <= 6000:
    #         short.append(llm_call)
    #     if llm_call.prefill_tokens > 6000:
    #         long.append(llm_call)
    #     print(f"prefill_tokens: {llm_call.prefill_tokens} and decode_tokens: {llm_call.decode_tokens}")
    # print(f"len(short): {len(short)} and len(long): {len(long)}", flush=True)
    # Step 1: Filter long_data_collect into two buckets
    short_prefill = [
        x for x in long_data_collect
        if  x.prefill_tokens <=  prefill_threshold1 and decode_threshold1 <= x.decode_tokens <= decode_threshold2
    ]
    long_prefill = [
        x for x in long_data_collect
        if  prefill_threshold1 < x.prefill_tokens <= prefill_threshold2  and decode_threshold1 <= x.decode_tokens <= decode_threshold2
    ]
    print(f"len(short_prefill): {len(short_prefill)} and len(long_prefill): {len(long_prefill)}", flush=True)
    # Step 2: Decide sample sizes

    remaining_jobs = int(total_jobs)
    long_quota = int(remaining_jobs * 0.25)
    short_quota = int(remaining_jobs - long_quota)
    print(f"short quota: {short_quota} and long_quota: {long_quota} and remaining_jobs :{remaining_jobs }", flush=True)
    if len(long_prefill) >= long_quota:
        selected_long = random.sample(long_prefill, long_quota)
    else:
        selected_long = long_prefill
        short_quota += long_quota - len(long_prefill)

    selected_short = random.sample(short_prefill, min(int(short_quota), len(short_prefill)))
    new_arxiv_calls = selected_short + selected_long
    random.shuffle(new_arxiv_calls)
    print(f"len(new_long_data_collect_calls): {len(new_arxiv_calls)} and total_jobs:{total_jobs}", flush=True)
    # Step 3: Sample sharegpt


    # Step 4: Combine
    mixed_calls = new_arxiv_calls
    random.shuffle(mixed_calls)
    mixed_calls = mixed_calls[:int(total_jobs)]

    # Step 5: Generate arrival times
    np.random.seed(seed)
    alpha = (1.0 / cv_factor) ** 2
    if arrival_period is not None:
        interarrival_times = []
        arrival_times = []
        current_time = 0
        while current_time < arrival_period:
            interarrival_time = np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
            interarrival_times.append(interarrival_time)
            current_time += interarrival_time
            arrival_times.append(current_time)
        arrival_times = np.array(arrival_times)
        if arrival_times[-1] > arrival_period:
            arrival_times = arrival_times[:-1]
        total_jobs = len(arrival_times)
    else:
        interarrival_times = np.array([
            np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
            for _ in range(total_jobs - 1)
        ])
        interarrival_times = np.insert(interarrival_times, 0, 0)
        arrival_times = np.cumsum(interarrival_times)
    current_len = min(len(arrival_times), len(mixed_calls))
    arrival_times = arrival_times[:current_len]
    # Step 6: Assign arrival times
    mixed_calls = mixed_calls[:current_len]
    print(f"current_len:{ current_len} and total_jobs:{total_jobs}", flush=True)
    for idx, llm_call in enumerate(mixed_calls):
        llm_call.arrival_time = math.ceil(arrival_times[idx]) if quantized else arrival_times[idx]
        llm_call.id = str(uuid.uuid4())

    print(f"Final mixed_calls: {len(mixed_calls)} / {total_jobs}")
    with open(MIXED_ARXIV, 'wb') as f:
        pickle.dump(mixed_calls, f)

    return mixed_calls

def skew_hybridserve_mixed_sharegpt_long_data_collections_leval(
        arrival_rate=1.0,
        cv_factor=1.0,
        total_jobs=1000,
        arrival_period=None,
        seed=2025,
        ratio: float = 0.5,
        quantized: bool = True,
        model_name: str = "llama3_1_8B"
):
    print(f"mixed_sharegpt_long_data_collections_leval")
    MIXED_SHAREGPT_LONG_DATA_COLLECT = f"skew_mixed_sharegpt_long_data_collect_leval_total_jobs_{total_jobs}.0_ratio_{ratio}_arrival_period_{arrival_period}_arrival_rate_{arrival_rate}.pkl"
    if os.path.exists(MIXED_SHAREGPT_LONG_DATA_COLLECT):
        print(f"1 os.path.exists(MIXED_SHAREGPT_LONG_DATA_COLLECT):{MIXED_SHAREGPT_LONG_DATA_COLLECT}", flush=True)
        with open(MIXED_SHAREGPT_LONG_DATA_COLLECT, 'rb') as f:
            return pickle.load(f)

    sharegpt_calls = []  # load_sharegpt_traces(model_name=model_name)
    long_data_collect = load_long_data_collections()
    leval_calls = []  # load_leval_dataset()

    decode_threshold1 = 100
    decode_threshold2 = 500
    prefill_threshold1 = 4000
    prefill_threshold2 = 10000
    # short = [] 
    # long = []
    # for llm_call in long_data_collect:
    #     if llm_call.prefill_tokens <= 6000:
    #         short.append(llm_call)
    #     if llm_call.prefill_tokens > 6000:
    #         long.append(llm_call)
    #     print(f"prefill_tokens: {llm_call.prefill_tokens} and decode_tokens: {llm_call.decode_tokens}")
    # print(f"len(short): {len(short)} and len(long): {len(long)}", flush=True)
    # Step 1: Filter long_data_collect into two buckets
    short_prefill = [
        x for x in long_data_collect
        if  x.prefill_tokens <= 6000 
    ]
    temp_short_prefill = []
    for llm_call in short_prefill:
        llm_call_temp = copy.deepcopy(llm_call)
        temp_short_prefill.append(llm_call)
        llm_call_temp.input = str(uuid.uuid4())  + llm_call_temp.input  + str(uuid.uuid4())
        temp_short_prefill.append(llm_call_temp)
    print(f"1 len(temp_short_prefill):{len(temp_short_prefill)} and len(short_prefill): {len(short_prefill)} ", flush=True)
    short_prefill = copy.deepcopy(temp_short_prefill)
    long_prefill = [
        x for x in long_data_collect
        if 6000 < x.prefill_tokens 
    ]
    print(f"2 len(temp_short_prefill):{len(temp_short_prefill)} and len(short_prefill): {len(short_prefill)} and len(long_prefill): {len(long_prefill)}", flush=True)
    # Step 2: Decide sample sizes
    new_sharegpt_size = int(total_jobs * ratio)
    remaining_jobs = total_jobs - new_sharegpt_size
    long_quota = int(remaining_jobs * 0.3)
    short_quota = remaining_jobs - long_quota

    if len(long_prefill) >= long_quota:
        selected_long = random.sample(long_prefill, long_quota)
    else:
        selected_long = long_prefill
        short_quota += long_quota - len(long_prefill)

    selected_short = random.sample(short_prefill, min(int(short_quota), len(short_prefill)))
    new_long_data_collect_calls = selected_short + selected_long
    random.shuffle(new_long_data_collect_calls)
    print(f"len(new_long_data_collect_calls): {len(new_long_data_collect_calls)} and total_jobs:{total_jobs}", flush=True)
    # Step 3: Sample sharegpt
    new_sharegpt_calls = random.sample(sharegpt_calls, new_sharegpt_size)

    # Step 4: Combine
    mixed_calls = new_sharegpt_calls + new_long_data_collect_calls
    random.shuffle(mixed_calls)
    mixed_calls = mixed_calls[:int(total_jobs)]

    # Step 5: Generate arrival times
    np.random.seed(seed)
    alpha = (1.0 / cv_factor) ** 2
    if arrival_period is not None:
        interarrival_times = []
        arrival_times = []
        current_time = 0
        while current_time < arrival_period:
            interarrival_time = np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
            interarrival_times.append(interarrival_time)
            current_time += interarrival_time
            arrival_times.append(current_time)
        arrival_times = np.array(arrival_times)
        if arrival_times[-1] > arrival_period:
            arrival_times = arrival_times[:-1]
        total_jobs = len(arrival_times)
    else:
        interarrival_times = np.array([
            np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
            for _ in range(total_jobs - 1)
        ])
        interarrival_times = np.insert(interarrival_times, 0, 0)
        arrival_times = np.cumsum(interarrival_times)
    current_len = min(len(arrival_times), len(mixed_calls))
    arrival_times = arrival_times[:current_len]
    # Step 6: Assign arrival times
    mixed_calls = mixed_calls[:current_len]
    print(f"current_len:{ current_len} and total_jobs:{total_jobs}", flush=True)
    for idx, llm_call in enumerate(mixed_calls):
        llm_call.arrival_time = math.ceil(arrival_times[idx]) if quantized else arrival_times[idx]
        llm_call.id = str(uuid.uuid4())

    print(f"Final mixed_calls: {len(mixed_calls)} / {total_jobs}")
    with open(MIXED_SHAREGPT_LONG_DATA_COLLECT, 'wb') as f:
        pickle.dump(mixed_calls, f)

    return mixed_calls

def hybridserve_mixed_sharegpt_arixv(
        arrival_rate = 1.0,
        cv_factor=1.0,
        total_jobs = 1000,
        arrival_period = None,
        seed = 2025,
        ratio : float = 0.5,
        quantized: bool=True,
        model_name: str="llama3_1_8B"
    ):
    print(f"mixed_sharegpt_long_data_collections_leval")
    MIXED_SHAREGPT_LONG_DATA_COLLECT = f"mixed_sharegpt_long_data_collect_leval_{total_jobs}.0_ratio_{ratio}_arrival_period_{arrival_period}_arrival_rate_{arrival_rate}.pkl"
    if os.path.exists(MIXED_SHAREGPT_LONG_DATA_COLLECT):
        print(f"1 os.path.exists(MIXED_SHAREGPT_LONG_DATA_COLLECT):{MIXED_SHAREGPT_LONG_DATA_COLLECT}", flush=True)
        with open(MIXED_SHAREGPT_LONG_DATA_COLLECT, 'rb') as f:
            return pickle.load(f)
    sharegpt_calls =load_sharegpt_traces() # load_sharegpt_traces(model_name = model_name)
    long_data_collect = load_arxiv_summary()
    leval_dataset_path = "LEval/LEval-data"
    leval_name = "leval"
    leval_calls = [] #load_lonngserve_dataset()
    print(f"len(sharegpt_calls): {len(sharegpt_calls)} and len(long_data_collect): {len(long_data_collect)} and len(leval_calls): {len(leval_calls)}", flush=True)
    decode_threshold1 = 10
    decode_threshold2 = 500
    prefill_threshold1 = 3000
    prefill_threshold2 = 5000
    new_long_data_collect = []
    for llm_call in long_data_collect:
        if llm_call.prefill_tokens > prefill_threshold2 or llm_call.prefill_tokens < prefill_threshold1:
            continue
        if (decode_threshold1<= llm_call.decode_tokens <= decode_threshold2):
            new_long_data_collect.append(llm_call)
    long_data_collect = new_long_data_collect
    new_leval_calls = []
    for llm_call in leval_calls:
        if llm_call.prefill_tokens > prefill_threshold2 or llm_call.prefill_tokens < prefill_threshold1:
            continue
        if (decode_threshold1<= llm_call.decode_tokens <= decode_threshold2):
            new_leval_calls.append(llm_call)
    leval_calls = new_leval_calls 

    long_data_collect = new_long_data_collect
    new_sharegpt_size = int(total_jobs * ratio)
    new_long_data_collect_size = int(total_jobs - new_sharegpt_size)
    if new_long_data_collect_size > len(long_data_collect):
        new_long_data_collect_size = len(long_data_collect)
        new_leval_size = total_jobs - new_sharegpt_size - new_long_data_collect_size
    else:
        new_leval_size = 0
    print(f"2 len(sharegpt_calls): {len(sharegpt_calls)} and len(long_data_collect): {len(long_data_collect)} and len(leval_calls): {len(leval_calls)}", flush=True)
    print(f"3 len(new_sharegpt_calls): {new_sharegpt_size} and len(new_long_data_collect): {new_long_data_collect_size} and len(new_leval_calls): {new_leval_size}", flush=True)
    # Set the random seed
    np.random.seed(seed)
    alpha = (1.0 / cv_factor)**2
    if new_leval_size == 0:
        new_sharegpt_calls = random.sample(sharegpt_calls, new_sharegpt_size)
        new_long_data_collect_calls = random.sample(long_data_collect, new_long_data_collect_size)
    else:
        new_sharegpt_calls = random.sample(sharegpt_calls, new_sharegpt_size)
        new_long_data_collect_calls = long_data_collect
        new_leval_calls = random.sample(leval_calls, new_leval_size)
        new_long_data_collect_calls.extend(new_leval_calls)

    # Generate arrival times
    if arrival_period is not None:
        # Generate interarrival times until current_time >= arrival_period
        interarrival_times = []
        arrival_times = []
        current_time = 0
        while current_time < arrival_period:
            interarrival_time = np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
            interarrival_times.append(interarrival_time)
            current_time += interarrival_time
            arrival_times.append(current_time)
        arrival_times = np.array(arrival_times)
        # Exclude the last arrival time if it exceeds arrival_period
        if arrival_times[-1] > arrival_period:
            arrival_times = arrival_times[:-1]
        total_jobs = len(arrival_times)
    else:
        # Generate a fixed number of interarrival times
        interarrival_times = np.array([
            np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
            for _ in range(total_jobs - 1)
        ])
        interarrival_times = np.insert(interarrival_times, 0, 0)
        arrival_times = np.cumsum(interarrival_times)
    

    mixed_calls = new_sharegpt_calls + new_long_data_collect_calls
    random.shuffle(mixed_calls)
    mixed_calls = mixed_calls[:total_jobs]
    
    #Assigin arrival time
    for idx, llm_call in enumerate(mixed_calls):
        if quantized:
            llm_call.arrival_time = math.ceil(arrival_times[idx])
        else:
            llm_call.arrival_time = arrival_times[idx]

    for llm_call in mixed_calls:
        uid = str(uuid.uuid4())
        llm_call.id = uid
    print(f"3 len(mixed_calls): {len(mixed_calls)} and total_jobs:{total_jobs}", flush=True)
    #save the mixed_calls into a file
    with open(MIXED_SHAREGPT_LONG_DATA_COLLECT, 'wb') as f:
        pickle.dump(mixed_calls, f)
    return mixed_calls


def hybridserve_mixed_sharegpt_long_data_collections_leval(
        arrival_rate = 1.0,
        cv_factor=1.0,
        total_jobs = 1000,
        arrival_period = None,
        seed = 2025,
        ratio : float = 0.5,
        quantized: bool=True,
        model_name: str="llama3_1_8B"
    ):
    print(f"mixed_sharegpt_long_data_collections_leval")
    MIXED_SHAREGPT_LONG_DATA_COLLECT = f"mixed_sharegpt_long_data_collect_leval_total_jobs_{total_jobs}.0_ratio_{ratio}_arrival_period_{arrival_period}_arrival_rate_{arrival_rate}.pkl"
    if os.path.exists(MIXED_SHAREGPT_LONG_DATA_COLLECT):
        print(f"1 os.path.exists(MIXED_SHAREGPT_LONG_DATA_COLLECT):{MIXED_SHAREGPT_LONG_DATA_COLLECT}", flush=True)
        with open(MIXED_SHAREGPT_LONG_DATA_COLLECT, 'rb') as f:
            return pickle.load(f)
    sharegpt_calls =load_sharegpt_traces() # load_sharegpt_traces(model_name = model_name)
    long_data_collect = load_long_data_collections()
    leval_dataset_path = "LEval/LEval-data"
    leval_name = "leval"
    leval_calls = [] #load_lonngserve_dataset()
    print(f"len(sharegpt_calls): {len(sharegpt_calls)} and len(long_data_collect): {len(long_data_collect)} and len(leval_calls): {len(leval_calls)}", flush=True)
    decode_threshold1 = 10
    decode_threshold2 = 500
    prefill_threshold1 = 4000
    prefill_threshold2 = 10000
    new_long_data_collect = []
    for llm_call in long_data_collect:
        if llm_call.prefill_tokens > prefill_threshold2 or llm_call.prefill_tokens < prefill_threshold1:
            continue
        if (decode_threshold1<= llm_call.decode_tokens <= decode_threshold2):
            new_long_data_collect.append(llm_call)
    long_data_collect = new_long_data_collect
    new_leval_calls = []
    for llm_call in leval_calls:
        if llm_call.prefill_tokens > prefill_threshold2 or llm_call.prefill_tokens < prefill_threshold1:
            continue
        if (decode_threshold1<= llm_call.decode_tokens <= decode_threshold2):
            new_leval_calls.append(llm_call)
    leval_calls = new_leval_calls 

    long_data_collect = new_long_data_collect
    new_sharegpt_size = int(total_jobs * ratio)
    new_long_data_collect_size = int(total_jobs - new_sharegpt_size)
    if new_long_data_collect_size > len(long_data_collect):
        new_long_data_collect_size = len(long_data_collect)
        new_leval_size = total_jobs - new_sharegpt_size - new_long_data_collect_size
    else:
        new_leval_size = 0
    print(f"2 len(sharegpt_calls): {len(sharegpt_calls)} and len(long_data_collect): {len(long_data_collect)} and len(leval_calls): {len(leval_calls)}", flush=True)
    print(f"3 len(new_sharegpt_calls): {new_sharegpt_size} and len(new_long_data_collect): {new_long_data_collect_size} and len(new_leval_calls): {new_leval_size}", flush=True)
    # Set the random seed
    np.random.seed(seed)
    alpha = (1.0 / cv_factor)**2
    if new_leval_size == 0:
        new_sharegpt_calls = random.sample(sharegpt_calls, new_sharegpt_size)
        new_long_data_collect_calls = random.sample(long_data_collect, new_long_data_collect_size)
    else:
        new_sharegpt_calls = random.sample(sharegpt_calls, new_sharegpt_size)
        new_long_data_collect_calls = long_data_collect
        new_leval_calls = random.sample(leval_calls, new_leval_size)
        new_long_data_collect_calls.extend(new_leval_calls)

    # Generate arrival times
    if arrival_period is not None:
        # Generate interarrival times until current_time >= arrival_period
        interarrival_times = []
        arrival_times = []
        current_time = 0
        while current_time < arrival_period:
            interarrival_time = np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
            interarrival_times.append(interarrival_time)
            current_time += interarrival_time
            arrival_times.append(current_time)
        arrival_times = np.array(arrival_times)
        # Exclude the last arrival time if it exceeds arrival_period
        if arrival_times[-1] > arrival_period:
            arrival_times = arrival_times[:-1]
        total_jobs = len(arrival_times)
    else:
        # Generate a fixed number of interarrival times
        interarrival_times = np.array([
            np.random.gamma(shape=alpha, scale=1 / (alpha * arrival_rate))
            for _ in range(total_jobs - 1)
        ])
        interarrival_times = np.insert(interarrival_times, 0, 0)
        arrival_times = np.cumsum(interarrival_times)
    

    mixed_calls = new_sharegpt_calls + new_long_data_collect_calls
    random.shuffle(mixed_calls)
    mixed_calls = mixed_calls[:total_jobs]
    
    #Assigin arrival time
    for idx, llm_call in enumerate(mixed_calls):
        if quantized:
            llm_call.arrival_time = math.ceil(arrival_times[idx])
        else:
            llm_call.arrival_time = arrival_times[idx]

    for llm_call in mixed_calls:
        uid = str(uuid.uuid4())
        llm_call.id = uid
    print(f"3 len(mixed_calls): {len(mixed_calls)} and total_jobs:{total_jobs}", flush=True)
    #save the mixed_calls into a file
    with open(MIXED_SHAREGPT_LONG_DATA_COLLECT, 'wb') as f:
        pickle.dump(mixed_calls, f)
    return mixed_calls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # load_sharegpt_traces()
    # load_lonngserve_dataset()

    #load_long_data_collections()

    # load_long_data_collections()

    load_arxiv_summary()
    # parser.add_argument("--dataset", type=str, default="sharegpt", help="dataset name")
    # parser.add_argument("--dataset_path", type=str, default="./ShareGPT_V3_unfiltered_cleaned_split.json", help="dataset path")
    # args = parser.parse_args()
    # if args.dataset == "sharegpt":
    #     load_sharegpt_traces()
    # elif args.dataset == "leval":
    #     load_lonngserve_dataset()
    # elif args.dataset == "long_data_collections":
    #     load_long_data_collections()
    # else:
    #     print("Invalid dataset")
    #     sys.exit(1)
    print("Done")