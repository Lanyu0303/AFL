import os
import time
import argparse
import json
import jsonlines

from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.lastfmAB_dataset import LastfmABDataset

from utils.regular_function import split_user_ab_response, split_rec_reponse, split_user_response
from utils.rw_process import append_jsonl, read_jsonl
from utils.api_request import api_request
from utils.agent import RecAgent, UserModelAgent
from utils.model import SASRec

finish_num = 0
total = 0
correct = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--init_num', type=int, default=10)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--prior_file', type=str, default='')
    parser.add_argument('--train_cans_num', type=int, default=10)
    parser.add_argument('--eval_cans_num', type=int, default=20)
    parser.add_argument('--a_ratio', type=int, default=1)
    parser.add_argument('--b_ratio', type=int, default=1)
    parser.add_argument('--stage', type=str, default='test')
    parser.add_argument('--sep', type=str, default=', ')
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--max_retry_num', type=int, default=5)
    parser.add_argument('--seed', type=int, default=333)
    parser.add_argument('--mp', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument("--save_info",action="store_true")
    parser.add_argument("--save_rec_dir", type=str, default=None)
    parser.add_argument("--save_user_dir", type=str, default=None)
    return parser.parse_args()

def recommend(data, args):
    start_time = time.time()
    rec_agent = RecAgent(args, 'prior_rec')
    user_agent = UserModelAgent(args, "prior_rec")
    flag = False
    epoch = 1
    rec_item = None
    new_data_list = []
    while flag == False and epoch <= args.max_epoch:
        #rec agent:
        while True:
            rec_agent_response = rec_agent.act(data)
            rec_reason, rec_item = split_rec_reponse(rec_agent_response)
            if rec_item is not None:
                break
        if args.max_epoch == 1:
            new_data = {'id':data['id'],'seq_name':data['seq_name'], 'cans_name':data['cans_name'], 'correct_answer':data['correct_answer'], 'epoch':epoch, 'rec_res':rec_agent_response, 'user_res':None,'prior_answer':data['prior_answer']}
            new_data_list.append(new_data)
            
            memory_info = {"epoch":epoch, "rec_reason":rec_reason, "rec_item":rec_item, "user_reason":None}
            rec_agent.update_memory(memory_info)
            user_agent.update_memory(memory_info)
            break

        #user agent:
        while True:
            user_agent_response = user_agent.act(data,rec_reason, rec_item)
            user_reason, flag  = split_user_response(user_agent_response)
            if flag is not None:
                break

        #save
        new_data = {'id':data['id'],'seq_name':data['seq_name'], 'cans_name':data['cans_name'], 'correct_answer':data['correct_answer'], 'epoch':epoch, 'rec_res':rec_agent_response, 'user_res':user_agent_response,'prior_answer':data['prior_answer']}
        new_data_list.append(new_data)

        memory_info = {"epoch":epoch, "rec_reason":rec_reason, "rec_item":rec_item, "user_reason":user_reason}
        rec_agent.update_memory(memory_info)
        user_agent.update_memory(memory_info)
        
        if flag:
            break
        epoch += 1
    end_time = time.time()
    print("recommend time = ", end_time - start_time)
    # save
    if args.save_info:
        rec_file_path = os.path.join(args.save_rec_dir, f"{data['id']}.jsonl")
        user_file_path = os.path.join(args.save_user_dir, f"{data['id']}.jsonl")
        rec_agent.save_memory(rec_file_path)
        user_agent.save_memory(user_file_path)
    # evaluate
    if rec_item.lower() == data['correct_answer'].lower().strip():
        return new_data_list, 1, args
    else:
        return new_data_list, 0, args


def setcallback(x):
    global finish_num
    global total
    global correct
    data_list, flag, args = x
    for data in data_list:
        append_jsonl(args.output_file, data)
    finish_num += 1
    correct += flag
    print("==============")
    print("correct = ", correct)
    print("finish = ", finish_num)
    print(f"now hit@1 = {correct} / {finish_num} = {correct/finish_num}")
    print(f"total hit@1 = {correct} / {total} = {correct/total}")
    print("==============")
    

def main(args):
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    if 'lastfm' in args.data_dir:
        dataset = LastfmABDataset(args.init_num, args.train_cans_num, args.eval_cans_num, args.a_ratio, args.b_ratio, args.data_dir, args.stage, args.sep)
    else:
        raise ValueError("Invalid dataset name.")
    global total
    data_list = []
    for data in tqdm(dataset):
        data_list.append(data)

    import pandas as pd
    prior_df = pd.read_csv(args.prior_file)
    prior_list = prior_df.to_dict('records')
    prior_dict = {}
    for data in prior_list:
        prior_dict[data['id']] = data
    merge_data_list = []
    for data in data_list:
        generate = prior_dict[data['id']]['generate']
        merge_data = data.copy()
        merge_data['prior_answer'] = generate
        merge_data_list.append(merge_data)

    # make dir
    if args.save_info and args.save_rec_dir is not None and not os.path.exists(args.save_rec_dir):
        os.makedirs(args.save_rec_dir)
    if args.save_info and args.save_user_dir is not None and not os.path.exists(args.save_user_dir):
        os.makedirs(args.save_user_dir)
    # mp
    pool = multiprocessing.Pool(args.mp)
    total = len(merge_data_list)
    for data in tqdm(merge_data_list):
        pool.apply_async(func=recommend, args=(data, args), callback=setcallback)
    pool.close()
    pool.join()


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    main(args)