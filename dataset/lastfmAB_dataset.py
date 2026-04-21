import os
import torch.utils.data as data

import random
import pickle

class LastfmABDataset(data.Dataset):
    def __init__(
        self,
        res_num=10,
        train_cans_num=10,
        eval_cans_num=20,
        a_ratio=1,
        b_ratio=1,
        data_dir='./data/lastfm_ab',
        stage='test',
        sep=", ",
    ):
        # initialize
        self.data_dir = data_dir
        self.stage = stage
        self.sep = sep
        self.res_num = res_num
        self.train_cans_num = train_cans_num
        self.eval_cans_num = eval_cans_num
        self.a_ratio = a_ratio
        self.b_ratio = b_ratio
        self.padding_item_id = 4606
        self.max_seq_len = 10
        # calculate
        self.pred_num = int((a_ratio) / (a_ratio + b_ratio) * eval_cans_num)
        self.fake_num = int((b_ratio) / (a_ratio + b_ratio) * eval_cans_num)
        assert self.pred_num + self.fake_num == self.eval_cans_num, "[calculate] sum error"
        self.item_id2name = self.get_music_id2name()
        self.data = self.build_data()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        temp = self.data[i]
        train_cands = self.train_negative_sampling(temp['full_list'], temp['next'])
        train_cands_name = [self.item_id2name[id] for id in train_cands]
        eval_cands = self.eval_negative_sampling(temp['full_list'], temp['pred_list'], train_cands)
        eval_cands_name = [self.item_id2name[id] for id in eval_cands]
        label = []
        for id in eval_cands:
            if id in temp['pred_list']:
                label.append(1)
            else:
                label.append(0)
        sample = {
            'id':i,
            'user_id':temp['user_id'],
            'seq':temp['seq'],
            'seq_name':temp['seq_title'],
            'seq_str':self.sep.join(temp['seq_title']),
            'len_seq':temp['len_seq'],
            'init_id':temp['init_list'],
            'init_title':temp['init_title_list'],
            'init_str':self.sep.join(temp['init_title_list']),
            'init_pad':temp['init_pad'],
            'len_init':temp['len_init'],
            'item_id':temp['next'],
            'item_name':temp['next_item_title'],
            'correct_answer':temp['next_item_title'],
            'pred_id':temp['pred_list'],
            'pred_title':temp['pred_title_list'],
            'cans':train_cands,
            'cans_name':train_cands_name,
            'cans_str':self.sep.join(train_cands_name),
            'len_cans':self.train_cans_num,
            'eval_cans_id':eval_cands,
            'eval_cans_title':eval_cands_name,
            'eval_cans_str':self.sep.join(eval_cands_name),
            'label':label
        }
        return sample
    
    def train_negative_sampling(self, full_list, next_item):
        canset = [i for i in list(self.item_id2name.keys()) if i not in full_list]
        sample_num = self.train_cans_num - 1
        if len(canset) < sample_num:
            raise ValueError(
                f"[train_negative_sampling] not enough candidates: "
                f"need {sample_num}, got {len(canset)}"
            )
        candidates = random.sample(canset, sample_num) + [next_item]
        random.shuffle(candidates)
        return candidates
    
    def eval_negative_sampling(self, full_list, pred_list, train_cans):
        canset = [i for i in list(self.item_id2name.keys()) if i not in full_list and i not in train_cans]
        if len(canset) < self.fake_num:
            raise ValueError(
                f"[eval_negative_sampling] not enough candidates: "
                f"need {self.fake_num}, got {len(canset)}"
            )
        candidates = random.sample(canset, self.fake_num) + pred_list
        random.shuffle(candidates)
        return candidates   
    
    def get_music_id2name(self):
        music_id2name = dict()
        item_path=os.path.join(self.data_dir, 'id2name_AB_llara.txt')
        with open(item_path, 'r') as f:
            for l in f.readlines():
                ll = l.strip('\n').split('::')
                music_id2name[int(ll[0])] = ll[1].strip()
        return music_id2name    
    
    def build_data(self):
        stage_to_filename = {
            'test': 'test_llara.pkl',
        }
        if self.stage not in stage_to_filename:
            raise ValueError(f"Unsupported stage: {self.stage}")
        file_name = stage_to_filename[self.stage]
        data_path = os.path.join(self.data_dir, file_name)
        with open(data_path, 'rb') as file:
            data_list = pickle.load(file)
        # function
        def split_seq_next(x):
            total_num = len(x)
            init_num = total_num - self.res_num
            seq_num = init_num - 1
            seq_list = x[:seq_num]
            next_item = x[seq_num]
            return (seq_list, next_item)
        
        def split_init_pred(x):
            total_num = len(x)
            init_num = total_num - self.res_num
            pred_num = self.pred_num
            assert init_num + pred_num <= total_num, "[split_init_pred]init + pred should be less than total "
            init_list = x[:init_num]
            pred_list = x[init_num:init_num + pred_num]
            return (init_list, pred_list)
        
        def seq_to_title(x): 
            return [self.item_id2name[x_i] for x_i in x]
        
        def id_to_title(x):
            return self.item_id2name[x]
        def get_seq_len(x):
            return len(x)
        
        def pad_seq(x):
            pad_x = x.copy()
            while len(pad_x) < self.max_seq_len:
                pad_x.append(self.padding_item_id)
            return pad_x
        
        processed_data = []
        for row in data_list:
            # keep backward compatibility with older key name 'artist_list'
            item_list = row.get('item_list', row.get('artist_list'))
            if item_list is None:
                raise KeyError("Each row in test_llara.pkl must contain 'item_list' (or legacy 'artist_list').")
            seq_unpad, next_item = split_seq_next(item_list)
            init_list, pred_list = split_init_pred(item_list)
            sample = dict(row)
            sample['seq_unpad'] = seq_unpad
            sample['next'] = next_item
            sample['seq'] = pad_seq(seq_unpad)
            sample['len_seq'] = get_seq_len(seq_unpad)
            sample['seq_title'] = seq_to_title(seq_unpad)
            sample['next_item_title'] = id_to_title(next_item)
            sample['init_list'] = init_list
            sample['init_pad'] = pad_seq(init_list)
            sample['len_init'] = get_seq_len(init_list)
            sample['pred_list'] = pred_list
            sample['init_title_list'] = seq_to_title(init_list)
            sample['pred_title_list'] = seq_to_title(pred_list)
            processed_data.append(sample)
        data = processed_data
        return data
    

if __name__ == '__main__':
    dataset = LastfmABDataset()
    print("data example = \n", dataset[0])
