import os
import __main__

import numpy as np
import pandas as pd
import torch

from utils.local_model import get_local_model, local_inference
from utils.model import SASRec
from utils.rw_process import write_jsonl, read_jsonl


REC_OUTPUT_RULE = (
    "\n\nOutput format (strict):\n"
    "Reason: <one short sentence>\n"
    "Item: <one artist name from candidates>\n"
    "Do not output markdown, examples, or extra sections."
)

USER_OUTPUT_RULE = (
    "\n\nOutput format (strict):\n"
    "Reason: <one short sentence>\n"
    "Decision: yes/no\n"
    "Do not output markdown, examples, or extra sections."
)


class RecAgent:
    def __init__(self, args, mode='prior_rec'):
        self.memory = []
        self.info_list = []
        self.args = args
        self.mode = mode
        self.load_prompt()
        self.llm_model, self.llm_tokenizer = get_local_model(self.args)

    def load_prompt(self):
        if self.mode == 'prior_rec':
            if 'lastfm' in self.args.data_dir:
                from constant.lastfm_prior_model_prompt import rec_system_prompt, rec_user_prompt, rec_memory_system_prompt, rec_memory_user_prompt, rec_build_memory
            else:
                raise ValueError(f"Invalid mode: {self.args.data_dir}")
            self.rec_system_prompt = rec_system_prompt
            self.rec_user_prompt = rec_user_prompt
            self.rec_memory_system_prompt = rec_memory_system_prompt
            self.rec_memory_user_prompt = rec_memory_user_prompt
            self.rec_build_memory = rec_build_memory
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def act(self, data, reason=None, item=None):
        if self.mode != 'prior_rec':
            raise ValueError(f"Invalid mode: {self.mode}")

        if len(self.memory) == 0:
            system_prompt = self.rec_system_prompt
            user_prompt = self.rec_user_prompt.format(data['seq_str'], data['len_cans'], data['cans_str'], data['prior_answer'])
        else:
            system_prompt = self.rec_memory_system_prompt
            user_prompt = self.rec_memory_user_prompt.format(data['seq_str'], data['len_cans'], data['cans_str'], '\n'.join(self.memory))
        user_prompt = f"{user_prompt}{REC_OUTPUT_RULE}"
        return local_inference(system_prompt, user_prompt, self.llm_model, self.llm_tokenizer, self.args)

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
        self.llm_model, self.llm_tokenizer = get_local_model(self.args)
        self.id2name = {}
        self.name2id = {}
        self.build_id2name()

    def build_id2name(self):
        if 'movielens' in self.args.data_dir:
            item_path = os.path.join(self.args.data_dir, 'u.item')
            with open(item_path, 'r', encoding="ISO-8859-1") as f:
                for line in f.readlines():
                    features = line.strip('\n').split('|')
                    idx = int(features[0]) - 1
                    name = features[1][:-7]
                    self.id2name[idx] = name
                    self.name2id[name] = idx
        elif 'lastfm' in self.args.data_dir or 'steam' in self.args.data_dir:
            item_path = os.path.join(self.args.data_dir, 'id2name_AB_llara.txt' if '_ab' in self.args.data_dir else 'id2name.txt')
            with open(item_path, 'r', encoding="ISO-8859-1") as f:
                for line in f.readlines():
                    key, value = line.strip('\n').split('::')
                    key = int(key)
                    value = value.strip()
                    self.id2name[key] = value
                    self.name2id[value] = key
        else:
            raise ValueError(f"Invalid data dir: {self.args.data_dir}")

    def load_model(self):
        data_statis = pd.read_pickle(os.path.join(self.args.data_dir, 'data_statis.df'))
        self.seq_size = data_statis['seq_size'][0]
        self.item_num = data_statis['item_num'][0]
        self.model = SASRec(64, self.item_num, self.seq_size, 0.1, self.device).to(self.device)

        __main__.SASRec = SASRec

        def load_checkpoint(path, map_location):
            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([SASRec])
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)

        try:
            self.model = load_checkpoint(self.args.model_path, self.device)
        except RuntimeError as exc:
            if "Attempting to deserialize object on a CUDA device" in str(exc):
                self.model = load_checkpoint(self.args.model_path, torch.device("cpu"))
            else:
                raise
        self.model.to(self.device)
        if hasattr(self.model, "device"):
            self.model.device = self.device
        self.model.eval()

    def load_prompt(self):
        if self.mode == 'prior_rec':
            from constant.lastfm_prior_model_prompt import user_system_promt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            self.user_system_promt = user_system_promt
            self.user_user_prompt = user_user_prompt
            self.user_memory_system_prompt = user_memory_system_prompt
            self.user_memory_user_prompt = user_memory_user_prompt
            self.user_build_memory = user_build_memory
            self.user_build_memory_2 = user_build_memory_2
        elif self.mode == 'pred':
            from constant.lastfm_ab_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            self.user_system_prompt = user_system_prompt
            self.user_user_prompt = user_user_prompt
            self.user_memory_system_prompt = user_memory_system_prompt
            self.user_memory_user_prompt = user_memory_user_prompt
            self.user_build_memory = user_build_memory
            self.user_build_memory_2 = user_build_memory_2
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def act(self, data, reason=None, item=None):
        if self.mode != 'prior_rec':
            raise ValueError(f"Invalid mode: {self.mode}")
        model_output = self.model_generate(data['seq'], data['len_seq'], data['cans'])
        system_prompt = self.user_system_promt.format(data['seq_str'], data['prior_answer'])
        user_prompt = self.user_user_prompt.format(data['cans_str'], model_output, item, reason)
        user_prompt = f"{user_prompt}{USER_OUTPUT_RULE}"
        return local_inference(system_prompt, user_prompt, self.llm_model, self.llm_tokenizer, self.args)

    def pred_model(self, data, score):
        if len(self.memory) == 0:
            system_prompt = self.user_system_prompt.format(data['seq_str'])
            user_prompt = self.user_user_prompt.format(data['pred_item'], score)
        else:
            system_prompt = self.user_memory_system_prompt.format(data['seq_str'], data['cans_str'], '\n'.join(self.memory))
            user_prompt = self.user_memory_user_prompt.format(data['pred_item'], score)
        user_prompt = f"{user_prompt}{USER_OUTPUT_RULE}"
        return local_inference(system_prompt, user_prompt, self.llm_model, self.llm_tokenizer, self.args)

    def build_memory(self, info):
        if info['user_reason'] is not None:
            return self.user_build_memory.format(info['epoch'], info['rec_item'], info['rec_reason'], info['user_reason'])
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
        states = torch.LongTensor(np.array([seq])).to(self.device)
        prediction = self.model.forward_eval(states, np.array([len_seq]))

        sampling_idx = [True] * self.item_num
        for item in candidates:
            sampling_idx[item] = False
        sampling_idxs = torch.stack([torch.tensor(sampling_idx)], dim=0)
        prediction = prediction.cpu().detach().masked_fill(sampling_idxs, prediction.min().item() - 1)
        _, topK = prediction.topk(len(candidates), dim=1, largest=True, sorted=True)
        topK = topK.numpy()[0]
        name_list = [self.id2name[idx] for idx in topK]
        return ', '.join(name_list[: int(len(name_list) / 4)])

    def score(self, seq, len_seq, candidates):
        states = torch.LongTensor(np.array([seq])).to(self.device)
        prediction = self.model.forward_eval(states, np.array([len_seq]))

        sampling_idx = [True] * self.item_num
        for item in candidates:
            sampling_idx[item] = False
        sampling_idxs = torch.stack([torch.tensor(sampling_idx)], dim=0)
        prediction = prediction.cpu().detach().masked_fill(sampling_idxs, prediction.min().item() - 1)
        values, topK = prediction.topk(len(candidates), dim=1, largest=True, sorted=True)
        values = values.numpy()[0]
        topK = topK.numpy()[0]
        return {self.id2name[topK[i]]: values[i] for i in range(len(topK))}
