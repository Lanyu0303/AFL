import os
import time
import argparse
import multiprocessing
import random
import sys

from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.lastfm_dataset import LastfmDataset
from utils.regular_function import split_user_response, split_rec_reponse
from utils.rw_process import append_jsonl
from utils.agent_llara import RecAgent, UserModelAgent

finish_num = 0
total = 0
correct = 0
_WORKER_REC_AGENT = None
_WORKER_USER_AGENT = None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--prior_file', type=str, default='')
    parser.add_argument('--stage', type=str, default='test')
    parser.add_argument('--cans_num', type=int, default=20)
    parser.add_argument('--sep', type=str, default=', ')
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--max_parse_retry', type=int, default=5)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--seed', type=int, default=333)
    parser.add_argument('--mp', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--llm_path', type=str, default='')
    parser.add_argument('--llara_ckpt_path', type=str, default='')
    parser.add_argument('--prompt_style', type=str, default='llama2_chat')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--torch_dtype', type=str, default='float16')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument("--save_info", action="store_true")
    parser.add_argument("--save_rec_dir", type=str, default=None)
    parser.add_argument("--save_user_dir", type=str, default=None)
    return parser.parse_args()


def recommend(data, args):
    start_time = time.time()
    global _WORKER_REC_AGENT, _WORKER_USER_AGENT
    if _WORKER_REC_AGENT is None or _WORKER_USER_AGENT is None:
        _WORKER_REC_AGENT = RecAgent(args, 'prior_rec')
        _WORKER_USER_AGENT = UserModelAgent(args, 'prior_rec')

    rec_agent = _WORKER_REC_AGENT
    user_agent = _WORKER_USER_AGENT
    rec_agent.memory, rec_agent.info_list = [], []
    user_agent.memory, user_agent.info_list = [], []
    flag = False
    epoch = 1
    rec_item = None
    new_data_list = []
    while flag is False and epoch <= args.max_epoch:
        rec_reason = None
        for _ in range(args.max_parse_retry):
            rec_agent_response = rec_agent.act(data)
            rec_reason, rec_item = split_rec_reponse(rec_agent_response)
            if rec_item is not None:
                break
        if rec_item is None:
            rec_item = data['cans_name'][0]
            rec_reason = ""
        if args.max_epoch == 1:
            new_data = {
                'id': data['id'], 'seq_name': data['seq_name'], 'cans_name': data['cans_name'],
                'correct_answer': data['correct_answer'], 'epoch': epoch, 'rec_res': rec_agent_response,
                'user_res': None, 'prior_answer': data['prior_answer']
            }
            new_data_list.append(new_data)
            memory_info = {"epoch": epoch, "rec_reason": rec_reason, "rec_item": rec_item, "user_reason": None}
            rec_agent.update_memory(memory_info)
            user_agent.update_memory(memory_info)
            break

        user_reason = None
        for _ in range(args.max_parse_retry):
            user_agent_response = user_agent.act(data, rec_reason, rec_item)
            user_reason, flag = split_user_response(user_agent_response)
            if flag is not None:
                break
        if flag is None:
            flag = False
            user_reason = ""

        new_data = {
            'id': data['id'], 'seq_name': data['seq_name'], 'cans_name': data['cans_name'],
            'correct_answer': data['correct_answer'], 'epoch': epoch, 'rec_res': rec_agent_response,
            'user_res': user_agent_response, 'prior_answer': data['prior_answer']
        }
        new_data_list.append(new_data)
        memory_info = {"epoch": epoch, "rec_reason": rec_reason, "rec_item": rec_item, "user_reason": user_reason}
        rec_agent.update_memory(memory_info)
        user_agent.update_memory(memory_info)
        if flag:
            break
        epoch += 1

    print("recommendation time = ", time.time() - start_time)
    if args.save_info:
        rec_file_path = os.path.join(args.save_rec_dir, f"{data['id']}.jsonl")
        user_file_path = os.path.join(args.save_user_dir, f"{data['id']}.jsonl")
        rec_agent.save_memory(rec_file_path)
        user_agent.save_memory(user_file_path)
    return (new_data_list, 1, args) if rec_item.lower() == data['correct_answer'].lower().strip() else (new_data_list, 0, args)


def setcallback(x):
    global finish_num, total, correct
    data_list, flag, args = x
    for data in data_list:
        append_jsonl(args.output_file, data)
    finish_num += 1
    correct += flag
    print("==============")
    print(f"current hit@1 = {correct} / {finish_num} = {correct/finish_num}")
    print(f"global hit@1 = {correct} / {total} = {correct/total}")
    print("==============")


def main(args):
    if args.save_info and args.save_rec_dir is not None and not os.path.exists(args.save_rec_dir):
        os.makedirs(args.save_rec_dir)
    if args.save_info and args.save_user_dir is not None and not os.path.exists(args.save_user_dir):
        os.makedirs(args.save_user_dir)
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    dataset = LastfmDataset(args.data_dir, args.stage, args.cans_num, args.sep, True)
    data_list = list(tqdm(dataset))
    print("data size = ", len(data_list))
    if len(data_list) == 0:
        raise ValueError(
            f"Dataset is empty for data_dir={args.data_dir}, stage={args.stage}. "
            "Check dataset files and stage split."
        )

    import pandas as pd
    if not os.path.exists(args.prior_file):
        raise FileNotFoundError(f"prior_file does not exist: {args.prior_file}")
    prior_df = pd.read_csv(args.prior_file)
    print("prior rows = ", len(prior_df))
    prior_dict = {row['id']: row for row in prior_df.to_dict('records')}

    merge_data_list = []
    for data in data_list:
        merge_data = data.copy()
        merge_data['prior_answer'] = prior_dict[data['id']]['generate']
        merge_data_list.append(merge_data)

    global total
    total = len(merge_data_list)
    if args.mp <= 1:
        for data in tqdm(merge_data_list):
            result = recommend(data, args)
            setcallback(result)
        return

    pool = multiprocessing.Pool(args.mp)
    for data in tqdm(merge_data_list):
        pool.apply_async(func=recommend, args=(data, args), callback=setcallback)
    pool.close()
    pool.join()


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    main(args)
