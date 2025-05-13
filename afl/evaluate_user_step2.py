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
from utils.regular_function import split_user_ab_response
from utils.rw_process import append_jsonl
from utils.agent import UserModelAgent

TP_sum = 0
FN_sum = 0
FP_sum = 0
TN_sum = 0
total = 0
recall_list = []
precision_list = []
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument("--load_info",action="store_true")
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--info_dir', type=str, default='')
    parser.add_argument('--init_num', type=int, default=10)
    parser.add_argument('--train_cans_num', type=int, default=10)
    parser.add_argument('--eval_cans_num', type=int, default=20)
    parser.add_argument('--a_ratio', type=int, default=1)
    parser.add_argument('--b_ratio', type=int, default=1)
    parser.add_argument('--stage', type=str, default='test')
    parser.add_argument('--sep', type=str, default=', ')
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--max_retry_num', type=int, default=5)
    parser.add_argument('--seed', type=int, default=333)
    parser.add_argument('--mp', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    return parser.parse_args()

def recommend(data, args):
    start_time = time.time()
    user_agent = UserModelAgent(args, 'pred')
    pred_label = []
    # load memory
    if args.load_info:
        file_path = os.path.join(args.info_dir, f"{data['id']}.jsonl")
        
        user_agent.load_memory(file_path)
    score_dict =user_agent.score(data['init_pad'], data['len_init'], data['eval_cans_id'])

    for index, next_item in enumerate(data['eval_cans_title']):
        pred_data = data.copy()
        pred_data['pred_item'] = next_item
        if next_item in score_dict:
            score = score_dict[next_item]
        else:
            score = min(score_dict.values())
        idx = 0
        while idx < 5:
            pred_user_response = user_agent.pred_model(pred_data, score)
            pred_reason, pred_res = split_user_ab_response(pred_user_response)
            if pred_res is not None:
                break
            idx += 1

        if pred_res is not None:
            pred_label.append(pred_res)
        else:
            y_t = data['label'][index]
            print("API request error. Adding 'false' label.")
            pred_label.append(1 - y_t)
        print("label list = ", pred_label)
    end_time = time.time()
    print("pred time = ", end_time - start_time)
    # evaluation
    return data, pred_label, args

def setcallback(x):
    global TP_sum, FN_sum, FP_sum, TN_sum
    global accuracy_list, recall_list, precision_list

    data, y_pred, args = x
    #save
    new_data = {'id':data['id'], 'init_title':data['init_title'],'pred_title':data['pred_title'],'label':data['label'],'y_pred':y_pred}
    append_jsonl(args.output_file, new_data)
    y_true = data['label']
    #cal
    TP = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))  
    FN = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))  
    FP = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))  
    TN = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))  

    TP_sum += TP
    FN_sum += FN
    FP_sum += FP
    TN_sum += TN
    try:
        recall = TP / (TP + FN)
        recall_list.append(recall)
    except Exception as e:
        recall = None
    try:
        precision = TP / (TP + FP)
        precision_list.append(precision)
    except Exception as e:
        precision = None
    print("==============")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print("==============")
    

def main(args):
    global TP_sum, FN_sum, FP_sum, TN_sum
    global accuracy_list, recall_list, precision_list
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
    pool = multiprocessing.Pool(args.mp)
    total = len(data_list)
    for data in tqdm(data_list):
        pool.apply_async(func=recommend, args=(data, args), callback=setcallback)
    pool.close()
    pool.join()

    print("================================global================================")
    global_recall = TP_sum / (TP_sum + FN_sum)
    global_precision = TP_sum / (TP_sum + FP_sum)
    print("global recall = ", global_recall)
    print("global precision = ", global_precision)
    print("==============================avg==================================")
    avg_recall = sum(recall_list) / len(recall_list)
    avg_precision = sum(precision_list) / len(precision_list)
    print("avg recall = ", avg_recall)
    print("avg precision = ", avg_precision)

if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    main(args)