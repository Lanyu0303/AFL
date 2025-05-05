import requests
import time

def api_request(system_prompt, user_prompt, args, few_shot=None):
    if "gpt" in args.model:
        return gpt_api(system_prompt, user_prompt, args, few_shot)
    else:
        raise ValueError(f"Unsupported model: {args.model}") 



def gpt_api(system_prompt, user_prompt, args):
    max_retry_num = args.max_retry_num
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}"
    }
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": args.model, 
        "messages": messages,
        "temperature": args.temperature,
    }
    while max_retry_num >= 0:
        request_result = None
        try:
            request_result = requests.post(url, headers=headers, json=payload)
            result_json = request_result.json()
            if 'error' not in result_json: 
                model_output = result_json['choices'][0]['message']['content']
                return model_output.strip()
            else:
                max_retry_num -= 1
        except:
            if request_result is not None:
                print("[warning]request_result = ", request_result.json())
                time.sleep(2)
            else:
                print("[warning]request_result = NULL")
            max_retry_num -= 1
    return None




            
