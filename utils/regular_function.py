import re


def _brief_response(response, max_len=120):
    one_line = " ".join((response or "").split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[:max_len] + "..."


def split_rec_reponse(response):
    response = (response or "") + '\n'
    pattern = (
        r'(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?\s*Reason\s*(?:\*\*)?\s*:\s*(.*?)\s*'
        r'(?:\n|\r\n)+\s*(?:[-*]\s*)?(?:\*\*)?\s*Item\s*(?:\*\*)?\s*:\s*(.*?)\s*(?:\n|$)'
    )
    matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
    if len(matches) < 1:
        print(f"[split_rec_reponse] can not split. sample='{_brief_response(response)}'")
        return None, None
    for match in reversed(matches):
        reason = match[0].strip()
        item = match[1].strip()
        if reason and item:
            return reason, item
    return None, None

def split_user_response(response):
    response = (response or "") + '\n'
    pattern = (
        r'(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?\s*Reason\s*(?:\*\*)?\s*:\s*(.*?)\s*'
        r'(?:\n|\r\n)+\s*(?:[-*]\s*)?(?:\*\*)?\s*Decision\s*(?:\*\*)?\s*:\s*(.*?)\s*(?:\n|$)'
    )
    matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
    if len(matches) < 1:
        print(f"[split_user_reponse] can not split. sample='{_brief_response(response)}'")
        return None, None

    for match in reversed(matches):
        decision = match[1].strip().lower()
        if decision.startswith('yes'):
            return match[0].strip(), True
        if decision.startswith('no'):
            return match[0].strip(), False
    print(f"[split_user_reponse] can not find flag. sample='{_brief_response(response)}'")
    return None, None
    

def split_user_rec_reponse(response):
    response += '\n'
    pattern = r'Reason: (.*?)\nItem: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print(f"[split_user_rec_reponse] can not split. sample='{_brief_response(response)}'")
        return None, None
    match = matches[0]
    return match[0].strip(), match[1].strip()

def split_user_ab_response(response):
    response += '\n'
    pattern = r'Reason:\s*(.*?)\nDecision:\s*(.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) < 1:
        print(f"[split_user_ab_reponse] can not split. sample='{_brief_response(response)}'")
        return None, None

    for match in reversed(matches):
        decision = match[1].strip().lower()
        if decision.startswith('yes'):
            return match[0].strip(), 1
        if decision.startswith('no'):
            return match[0].strip(), 0
    print(f"[split_user_ab_reponse] can not find flag. sample='{_brief_response(response)}'")
    return None, None
    
def split_prior_rec_response(response):
    response += '\n'
    pattern = r'Item: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print(f"[split_prior_rec_response] can not split. sample='{_brief_response(response)}'")
        return None
    match = matches[0]
    return match.strip()

def split_prior_llama3_response(response):
    pattern = r'Item: (.*?)<\|eot_id\|>'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print(f"[split_prior_llama3_response] can not split, try split2. sample='{_brief_response(response)}'")
        return split_prior_rec_response(response)
    match = matches[0]
    return match.strip()
