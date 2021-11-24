import faiss
import faiss.contrib.torch_utils
import argparse
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
import pandas as pd

class QueryDocumentMatcher():
    def __init__(self, embeddings_list_file='tensor/embeddings_list.pt', doc_ids_file='tensor/doc_ids.pt', \
        phrases_list_file='tensor/phrase_list.txt', data_path='data/ucas_faq.xlsx', \
        component_file='data/components.txt', question_column='点位名称', car_column='车系名称', text_column='内容节选', \
        test_size=-1, gpu=True, index_search_topk=32, res_topk=32, \
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2", \
        cls_model_name="/mnt/bd/zhengxin/transformers/examples/pytorch/text-classification/models/content_thickness_category", \
        vector_dimension=768*1, **args):
        self.embeddings_list_file=embeddings_list_file
        self.doc_ids_file=doc_ids_file
        self.appear_phrases_list_file=phrases_list_file
        self.data_path=data_path
        self.component_file=component_file
        self.question_column=question_column
        self.car_column=car_column
        self.text_column=text_column
        self.test_size=test_size
        self.gpu=gpu
        self.index_search_topk=index_search_topk
        self.res_topk=res_topk
        self.model_name=model_name
        self.cls_model_name=cls_model_name
        self.vector_dimension=vector_dimension


        if self.data_path.endswith('xlsx'):
            df = pd.read_excel(self.data_path)
        else:
            df = pd.read_csv(self.data_path)
        self.df = df.fillna('')
        self.answer_list = df['answer'].tolist()
        self.n = len(df.index)

        with open(phrases_list_file, 'r') as f:
            key_phrase_list = f.readlines()
            self.key_phrase_list = [phrase.strip() for phrase in key_phrase_list]

        self.doc_id = torch.load(self.doc_ids_file)
        self.embeddings_list = torch.load(self.embeddings_list_file)
        self.embeddings_list = F.normalize(self.embeddings_list) 
        self.question_list = self.df['question'].tolist()


        self.index_flat = faiss.IndexFlatL2(self.vector_dimension)
        if self.gpu:
            res = faiss.StandardGpuResources()  # use a single GPU
            self.index_flat = faiss.index_cpu_to_gpu(res, 0, self.index_flat)
        self.index_flat.add(self.embeddings_list)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).cuda()

    def encode_texts(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        for key in encoded_input.keys():
            encoded_input[key] = encoded_input[key].cuda() 
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        return encoded_input['input_ids'], model_output.last_hidden_state

    def encode_question(self, question, normalize=True):
        token_ids, question_embeddings = self.encode_texts(question)
        question_vector = question_embeddings.sum(0) / len(token_ids[0])
        if normalize:
            question_vector = F.normalize(question_vector)
        return question_vector
        
    def query_document_match(self, question, return_id=False):
        question_embedding = self.encode_question(question)
        _, I = self.index_flat.search(question_embedding, self.index_search_topk)
        I = I[0].tolist()
        top_doc_id = self.doc_id[I[0]]
        for phrase_id in I:
            phrase = self.key_phrase_list[phrase_id]
            if phrase in question:
                top_doc_id = self.doc_id[phrase_id]
                break

        if not return_id:
            return self.answer_list[top_doc_id], [self.answer_list[self.doc_id[phrase_id].item()] for phrase_id in I]
        else:
            return top_doc_id, I





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # torch.Size([60648, 768])
    parser.add_argument('--embeddings_list_file', type=str, default='tensor/embeddings_list.pt')
    parser.add_argument('--aug_delta', type=int, default=0)
    parser.add_argument('--doc_ids_file', type=str, default='tensor/doc_ids.pt')
    parser.add_argument('--question_file', type=str, default='')
    parser.add_argument('--phrase_list_file', type=str, default='tensor/phrase_list.txt')
    parser.add_argument('--data_path', type=str, default='data/ucas_faq.xlsx')
    parser.add_argument('--test_size', type=int, default=-1)
    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--index_search_topk', type=int, default=32)
    parser.add_argument('--res_topk', type=int, default=32)
    parser.add_argument('--model_name', type=str, default="sentence-transformers/distiluse-base-multilingual-cased-v2")
    parser.add_argument('--vector_dimension', type=int, default=768*1)

    args = parser.parse_args()

    matcher = QueryDocumentMatcher(**vars(args))

    question_list = matcher.question_list
    questions = question_list
    if args.question_file != '':
        with open(args.question_file, 'r') as f:
            questions = f.readlines()
            questions = [q.strip() for q in questions]
    answer_list = matcher.answer_list
    doc_id = matcher.doc_id
    key_phrases = matcher.key_phrase_list
    topk_acc = 0
    precision = 0
    for i, question in enumerate(questions):
        if args.test_size != -1 and i >= args.test_size:
            break 
        
        top_doc_id, car_related_answer = matcher.query_document_match(question, return_id=True)

        print(i, question, answer_list[i][:32])

        if (top_doc_id == i or answer_list[i][:32] == answer_list[top_doc_id][:32]) or question_list[i][:32] == question_list[top_doc_id][:32]:
            precision += 1
            print(True, answer_list[top_doc_id].strip(), sep='\t')
        else:
            print(False, answer_list[top_doc_id].strip(), sep='\t')

        hit = 0
        for phrase_id in car_related_answer:
            phrase_id_in_doc = doc_id[phrase_id].item()
            if i == phrase_id_in_doc or answer_list[i][:32] == answer_list[phrase_id_in_doc][:32] or question_list[i][:32] == question_list[phrase_id_in_doc][:32]:
                hit = 1
            print(key_phrases[phrase_id], phrase_id_in_doc, answer_list[phrase_id_in_doc][:64], sep='\t')
        topk_acc += hit
        print("right" if hit == 1 else "wrong", topk_acc / (i+1), precision / (i+1), sep='\t')