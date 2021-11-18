import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import trange


parser = argparse.ArgumentParser()
parser.add_argument('--embeddings_list_file', type=str, default='tensor/embeddings_list.pt')
parser.add_argument('--aug_delta', type=int, default=0)
parser.add_argument('--doc_ids_file', type=str, default='tensor/doc_ids.pt')
parser.add_argument('--phrase_list_file', type=str, default='tensor/phrase_list.txt')
parser.add_argument('--data_path', type=str, default='data/ucas_faq.xlsx')
parser.add_argument('--model_name', type=str, default="sentence-transformers/distiluse-base-multilingual-cased-v2")

args = parser.parse_args()

if args.data_path.endswith('xlsx'):
    df = pd.read_excel(args.data_path)
else:
    df = pd.read_csv(args.data_path)

df = df.fillna('')
n = len(df.index)
print('read docs')
print(df)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

model = AutoModel.from_pretrained(args.model_name).cuda()

def encode_text(text, model=model):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    for key in encoded_input.keys():
        encoded_input[key] = encoded_input[key].cuda() 
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)
    return encoded_input['input_ids'][0], model_output.last_hidden_state[0]
    

def encode_phrase(token_ids, key_phrase, full_embeddings):
    len_text = token_ids.size()[0]
    key_phrase_embeddings = []
    matched_phrases = []
    key_phrase_token_ids = [torch.tensor(tokenizer.encode(phrase)[1:-1]).cuda() for phrase in key_phrase]
    i = 0
    while i < len_text and len(matched_phrases) < len(key_phrase):
        for phrase, phrase_token_ids in zip(key_phrase, key_phrase_token_ids):
            len_phrase = int(phrase_token_ids.size(0))
            if phrase not in matched_phrases and torch.equal(token_ids[i: i+len_phrase], phrase_token_ids):
                r = min(len(full_embeddings), i + len_phrase)
                key_phrase_embedding = torch.sum(full_embeddings[i: r, :], 0) / (r - i)
                key_phrase_embeddings.append(key_phrase_embedding)
                matched_phrases.append(phrase)
                i += len_phrase - 1
                break
        i += 1
    return key_phrase_embeddings, matched_phrases

def annotate_similar(doc_embeddings, question_embeddings):
    score = torch.softmax(doc_embeddings[:64, :] @ question_embeddings.T, 1)
    score = score.max(1).values
    return score

embeddings_list, key_phrase_list, doc_ids = [], [], []

for index, row in df.iterrows():
    school, province, key_phrase, question, answer = row['school'], row['province'], row['key_phrase'], row['question'], row['answer']
    key_phrase = key_phrase.split()
    text = ' [SEP] '.join([school, province, question, answer[:64]])
    text = text.replace('[SEP] [SEP]', '[SEP]')
    token_ids, full_embeddings = encode_text(text)
    key_phrase_embeddings, key_phrase = encode_phrase(token_ids, key_phrase, full_embeddings)
    key_phrase_list += key_phrase
    embeddings_list += key_phrase_embeddings
    doc_ids += [index] * len(key_phrase)

embeddings_list = torch.vstack(embeddings_list)
doc_ids = torch.tensor(doc_ids).cuda()
torch.save(embeddings_list, args.embeddings_list_file)
torch.save(doc_ids, args.doc_ids_file)
with open(args.phrase_list_file, 'w') as f:
    print(*key_phrase_list, sep='\n', file=f)
print(embeddings_list.shape)
