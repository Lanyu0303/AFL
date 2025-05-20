import os
import time
import argparse
import json
import jsonlines
import torch
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import sys
import pandas as pd
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.regular_function import split_user_response, split_rec_reponse
from utils.rw_process import append_jsonl, write_jsonl, read_jsonl
from utils.api_request import api_request
from utils.model import SASRec

class RecAgent:
    def __init__(self, args, mode='prior_rec'):
        self.memory = []
        self.info_list = []
        self.args = args
        self.mode = mode
        self.load_prompt()

    def load_prompt(self):
        if self.mode =='prior_rec':
            if 'lastfm' in self.args.data_dir:
                from constant.lastfm_prior_model_prompt import rec_system_prompt, rec_user_prompt, rec_memory_system_prompt, rec_memory_user_prompt, rec_build_memory
            else:
                raise ValueError("Invalid mode: {}".format(self.args.data_dir))
            self.rec_system_prompt = rec_system_prompt
            self.rec_user_prompt = rec_user_prompt
            self.rec_memory_system_prompt = rec_memory_system_prompt
            self.rec_memory_user_prompt = rec_memory_user_prompt
            self.rec_build_memory = rec_build_memory
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def act(self, data, reason=None, item=None):
        if self.mode =='prior_rec':
            if len(self.memory) == 0:
                system_prompt = self.rec_system_prompt
                user_prompt = self.rec_user_prompt.format(data['seq_str'], data['len_cans'],data['cans_str'], data['prior_answer'])
            else:
                system_prompt = self.rec_memory_system_prompt
                user_prompt = self.rec_memory_user_prompt.format(data['seq_str'],data['len_cans'], data['cans_str'], '\n'.join(self.memory))
            response = api_request(system_prompt, user_prompt, self.args)
            return response
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))
        
    def build_memory(self, info):
        return self.rec_build_memory.format(info['epoch'], info['rec_item'], info['rec_reason'], info['user_reason'])
    
    def update_memory(self, info):
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def save_memory(self, path):
        write_jsonl(path, self.info_list)
    
    def load_memory(self, path):
        self.info_list = read_jsonl(path)
        self.memory = [self.build_memory(info) for info in self.info_list]


class UserModelAgent:
    def __init__(self, args, mode='prior_rec'):
        self.memory = []
        self.info_list = []
        self.args = args
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_prompt()
        self.load_model()
        self.id2name = dict()
        self.name2id = dict()
        self.build_id2name()

    def build_id2name(self):
        if 'movielens' in self.args.data_dir:
            def get_mv_title(s):
                sub_list=[", The", ", A", ", An"]
                for sub_s in sub_list:
                    if sub_s in s:
                        return sub_s[2:]+" "+s.replace(sub_s,"")
                return s
            item_path = os.path.join(self.args.data_dir, 'u.item')
            with open(item_path, 'r', encoding = "ISO-8859-1") as f:
                for line in f.readlines():
                    features = line.strip('\n').split('|')
                    id = int(features[0]) - 1
                    name = get_mv_title(features[1][:-7])
                    self.id2name[id] = name
                    self.name2id[name] = id
        elif 'lastfm' in self.args.data_dir or 'steam' in self.args.data_dir:
            if '_ab' in self.args.data_dir:
                item_path=os.path.join(self.args.data_dir, 'id2name_AB_llara.txt')
            else:
                item_path=os.path.join(self.args.data_dir, 'id2name.txt')
            with open(item_path, 'r') as f:
                for l in f.readlines():
                    ll = l.strip('\n').split('::')
                    self.id2name[int(ll[0])] = ll[1].strip()
                    self.name2id[ll[1].strip()] = int(ll[0])
        else:
            raise ValueError("Invalid data dir: {}".format(self.args.data_dir))
        
    def load_model(self):
        print("loading model")
        data_directory = self.args.data_dir
        data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
        self.seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
        self.item_num = data_statis['item_num'][0]  # total number of items
        self.model = SASRec(64,self.item_num, self.seq_size,0.1,self.device)
        self.model.to(self.device)
        self.model=torch.load(self.args.model_path)
        print("load model success")
    
    def load_prompt(self):
        if self.mode == 'prior_rec':
            if 'lastfm' in self.args.data_dir:
                from constant.lastfm_prior_model_prompt import user_system_promt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_recommend_system_prompt, user_recommend_user_prompt, user_recommend_memory_system_prompt, user_recommend_memory_user_prompt, user_build_memory, user_build_memory_2
            else:
                raise ValueError("Invalid dataset: {}".format(self.args.data_dir))
            self.user_system_promt = user_system_promt
            self.user_user_prompt = user_user_prompt
            self.user_memory_system_prompt = user_memory_system_prompt
            self.user_memory_user_prompt = user_memory_user_prompt
            self.user_recommend_system_prompt = user_recommend_system_prompt
            self.user_recommend_user_prompt = user_recommend_user_prompt
            self.user_recommend_memory_system_prompt = user_recommend_memory_system_prompt
            self.user_recommend_memory_user_prompt = user_recommend_memory_user_prompt
            self.user_build_memory = user_build_memory
            self.user_build_memory_2 = user_build_memory_2

        elif self.mode == 'pred':
            if 'lastfm' in self.args.data_dir:
                from constant.lastfm_ab_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2, user_memory_system_prompt, user_memory_user_prompt
            else:
                raise ValueError("Invalid dataset: {}".format(self.args.data_dir))
            self.user_system_prompt = user_system_prompt
            self.user_user_prompt = user_user_prompt
            self.user_memory_system_prompt = user_memory_system_prompt
            self.user_memory_user_prompt = user_memory_user_prompt
            self.user_build_memory = user_build_memory
            self.user_build_memory_2 = user_build_memory_2
    def act(self, data, reason=None, item=None):
        if self.mode == 'prior_rec':
            model_output = self.model_generate(data['seq'], data['len_seq'], data['cans'])
            if len(self.memory) == 0:
                system_prompt = self.user_system_promt.format(data['seq_str'], data['prior_answer'])
            else:
                system_prompt = self.user_system_promt.format(data['seq_str'], data['prior_answer'])
            user_prompt = self.user_user_prompt.format(data['cans_str'],model_output, item, reason)
            response = api_request(system_prompt, user_prompt, self.args)
            return response
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def pred_model(self, data, score):
        if len(self.memory) == 0:
            system_prompt = self.user_system_prompt.format(data['seq_str'])
            user_prompt = self.user_user_prompt.format(data['pred_item'], score)
        else:
            system_prompt = self.user_memory_system_prompt.format(data['seq_str'], data['cans_str'],'\n'.join(self.memory))
            user_prompt = self.user_memory_user_prompt.format(data['pred_item'],score)
            
        response = api_request(system_prompt, user_prompt, self.args)
        return response
    
    def build_memory(self, info):
        if info['user_reason'] is not None:
            return self.user_build_memory.format(info['epoch'], info['rec_item'], info['rec_reason'], info['user_reason'])
        else:
            return self.user_build_memory_2.format(info['epoch'], info['rec_item'], info['rec_reason'])
    
    def update_memory(self, info):
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def save_memory(self, path):
        write_jsonl(path, self.info_list)
    
    def load_memory(self, path):
        self.info_list = read_jsonl(path)
        self.memory = [self.build_memory(info) for info in self.info_list]

    def model_generate(self, seq, len_seq, candidates):
        seq_b = [seq]
        len_seq_b = [len_seq]
        states = np.array(seq_b)
        states = torch.LongTensor(states)
        states = states.to(self.device)
        prediction = self.model.forward_eval(states, np.array(len_seq_b))

        sampling_idx=[True]*self.item_num
        cans_num = len(candidates)
        for i in candidates:
            sampling_idx.__setitem__(i,False)
        sampling_idxs = [torch.tensor(sampling_idx)]
        sampling_idxs=torch.stack(sampling_idxs,dim=0)
        prediction = prediction.cpu().detach().masked_fill(sampling_idxs,prediction.min().item()-1)
        values, topK = prediction.topk(cans_num, dim=1, largest=True, sorted=True)
        topK = topK.numpy()[0]
        name_list = [self.id2name[id] for id in topK]
        len_ret = int(len(name_list) /4 )
        return ', '.join(name_list[:len_ret])