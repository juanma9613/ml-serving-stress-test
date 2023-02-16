import requests
import json
from transformers import AutoTokenizer, AutoConfig, DistilBertTokenizer, BertTokenizer
import numpy as np
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


tokenizer = DistilBertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

print('tokinzer', type(tokenizer))

text = "I like you, but i'm sad"
MAX_SEQ_LEN = 100

encoded_input = tokenizer(text, pad_to_max_length=MAX_SEQ_LEN, max_length=MAX_SEQ_LEN)
#TF Serve endpoint

# LOCAL URL
#url = "http://localhost:8501/v1/models/bert-base-uncased-SST-2:predict"
# ELASTIC URL EC2
url = "ec2-3-132-201-36.us-east-2.compute.amazonaws.com:8501/v1/models/bert-base-uncased-SST-2:predict"



payload={"instances": [{"input_ids": encoded_input['input_ids'], "attention_mask": encoded_input['attention_mask']}]}
print(payload)
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=json.dumps(payload))


print(softmax(json.loads(response.text)['predictions'][0]))