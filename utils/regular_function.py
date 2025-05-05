import re

def split_rec_reponse(response):
    response += '\n'
    pattern = r'Reason: (.*?)\nItem: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_rec_reponse]can not split, response = ", response)
        return None, None
    match = matches[0]
    return match[0].strip(), match[1].strip()

def split_user_response(response):
    response += '\n'
    pattern = r'Reason: (.*?)\Decision: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_user_reponse]can not split, response = ", response)
        return None, None

    match = matches[0]
    if match[1].lower().startswith('yes'):
        return match[0].strip(), True
    elif match[1].lower().startswith('no'):
        return match[0].strip(), False
    else:
        print("[split_user_reponse]can not find flag, response = ", response)
        return None, None
    

def split_user_rec_reponse(response):
    response += '\n'
    pattern = r'Reason: (.*?)\nItem: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_user_rec_reponse]can not split, response = ", response)
        return None, None
    match = matches[0]
    return match[0].strip(), match[1].strip()

def split_user_ab_response(response):
    response += '\n'
    pattern = r'Reason: (.*?)\nDecision: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_user_ab_reponse]can not split, response = ", response)
        return None, None

    match = matches[0]
    if match[1].lower().startswith('yes'):
        return match[0].strip(), 1
    elif match[1].lower().startswith('no'):
        return match[0].strip(), 0
    else:
        print("[split_user_ab_reponse]can not find flag, response = ", response)
        return None, None
    
def split_prior_rec_response(response):
    response += '\n'
    pattern = r'Item: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_prior_rec_response]can not split, response = ", response)
        return None
    match = matches[0]
    return match.strip()

def split_prior_llama3_response(response):
    pattern = r'Item: (.*?)<\|eot_id\|>'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_prior_llama3_response]can not split,try split2,  response = ", response)
        return split_prior_rec_response(response)
    match = matches[0]
    return match.strip()