#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from datetime import datetime
from collections import Counter
import re
import time
import os
import pandas as pd
import numpy as np
import codecs

# 取得 NER input array (陳水扁貪汙 -> CKIP(陳水扁) -> 1 2 2 0 0)
def transfer_NER(data, tokenizer, maxlen):
    org_list = []
    label = np.zeros([len(data), maxlen])
    
    # 用CKIP抓出每個新聞的組織
    for index, (_, row) in enumerate(data.iterrows()):
        
        if index % 100 == 0:
            print(index)
        
        token = tokenizer.tokenize(row['content'][0:maxlen])

        if len(token) > maxlen:
            token = token[0:maxlen-1]
            token.append('[SEP]')

        y = np.zeros([maxlen])
        content = ''.join(token)
        
        #CKIP
        word_sentence_list = ws([content],
                    sentence_segmentation=True,
                    segment_delimiter_set={'?', '？', '!', '！', '。', ',', '，', ';', ':', '、'})
        pos_sentence_list = pos(word_sentence_list)
        entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
        
        org = [org for org in list(entity_sentence_list[0]) if (org[2] == 'ORG') & (org[1] < maxlen)]
        org = [org for org in org if ('#' not in org[3])]
        org.sort()
        org.append(org)
        
        #轉換成input array
        j = 0
        for p in org: 
            for i, _ in enumerate(token):        
                if token[i:i+len(p[3])] == list(p[3]):
                    if len(p) == 1:
                        y[i+j] = 1
                        token = token[i+1:]
                        j = i+j+1
                        break

                    y[i+j] = 1
                    y[i+j+1:i+j+len(p[3])] = 2
                    token = token[i+len(p[3]):]
                    j = i+j+len(p[3])
                    break
        label[index, :] = y
    
    #用不到 org_list，只是檢查用
    return org_list, label.reshape([label.shape[0], label.shape[1], 1])


def create_tokenizer(dict_path):
    
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
            
    return token_dict


# 把大於512的新聞以句點分段 (bert最多只能吃512)
def split_contentTokenizer(data):
    data_more_split = pd.DataFrame()
    for i, row in data.iterrows():
        if (len(row['content']) > 512) & (len(row['content']) <= 1024):

            s = row['content']
            s_split = [(i, abs(len(s)//2 - s.find(x)), x) for i, x in enumerate(s.split('。'))]
            idx_left = min(s_split, key=lambda x: x[1])[0]
            first = "。".join([s_split[i][2] for i in range(idx_left)])
            second = "。".join([s_split[i][2] for i in range(idx_left, len(s_split))])    
            contents = [first, second]

            for content in contents:
                data_more_split = data_more_split.append(pd.DataFrame({'news_id':row['news_id'], 'content':content}, index=[66]), ignore_index=True)

        elif len(row['content']) > 1024:

            s = row['content']
            s_split1 = [(i, abs(len(s)//3 - s.find(x)), x) for i, x in enumerate(s.split('。'))]
            s_split2 = [(i, abs(len(s)*2//3 - s.find(x)), x) for i, x in enumerate(s.split('。'))]
            idx_left1 = min(s_split1, key=lambda x: x[1])[0]
            idx_left2 = min(s_split2, key=lambda x: x[1])[0]
            first = "。".join([s_split1[i][2] for i in range(idx_left1)])
            second = "。".join([s_split1[i][2] for i in range(idx_left1, idx_left2)])
            third = "。".join([s_split1[i][2] for i in range(idx_left2, len(s_split1))])
            contents = [first, second, third]

            for content in contents:
                data_more_split = data_more_split.append(pd.DataFrame({'news_id':row['news_id'], 'content':content}, index=[66]), ignore_index=True)
    
    return data_more_split



# 取得 NER input array (陳水扁貪汙 -> CKIP(陳水扁) -> 1 2 2 0 0)
def transfer_NER_array(data, org_list, tokenizer, maxlen):
    
    label = np.zeros([len(data), maxlen])
    
    # 用CKIP抓出每個新聞的名字
    for (index, (_, row)), org in zip(enumerate(data.iterrows()), org_list):
        
        #if index % 100 == 0:
        #    print(index)
        
        token = tokenizer.tokenize(row['content'][0:maxlen])

        if len(token) > maxlen:
            token = token[0:maxlen-1]
            token.append('[SEP]')

        y = np.zeros([maxlen])
        content = ''.join(token)
        
        #轉換成input array
        j = 0
        for p in org: 
            for i, _ in enumerate(token):        
                if token[i:i+len(p[3])] == list(p[3]):
                    if len(p) == 1:
                        y[i+j] = 1
                        token = token[i+1:]
                        j = i+j+1
                        break

                    y[i+j] = 1
                    y[i+j+1:i+j+len(p[3])] = 2
                    token = token[i+len(p[3]):]
                    j = i+j+len(p[3])
                    break
        label[index, :] = y
    
    #用不到 org_list，只是檢查用
    return label.reshape([label.shape[0], label.shape[1], 1])

def transfer(i):
    
    if i != 0:
        return 1
    else:
        return 0



def encoded(tokenizer, data, maxlen):
    
    x, y, z = [], [], []
    if 'content' in data.columns:
        for content in data['content']:
            x1, x2 = tokenizer.encode(content, max_len=maxlen)
            x3 = [transfer(i) for i in x1]
            x.append(x1)
            y.append(x2)
            z.append(x3)
    elif 'Sentence' in data.columns:
        for content in data['Sentence']:
            x1, x2 = tokenizer.encode(content, max_len=maxlen)
            x3 = [transfer(i) for i in x1]
            x.append(x1)
            y.append(x2)
            z.append(x3)
            
    return x, y, z


# 取得名字 (預測結果為onehot的狀態)
def get_name(token_dict, input_id, y_pred):
    
    label_list = []
    word_dict = {v: k for k, v in token_dict.items()}
    
    for input_data, y in zip(input_id, y_pred):
        org_index = ''.join([str(a) for a in list(y)])
        j = 0
        name_list = []
        split_index = re.findall('[12]2*', org_index)
        name = ''.join([word_dict.get(input_data[index]) for index, value in enumerate(y) if value != 0])
        
        # [UNK], [PAD]會被算成 5 個字元，避免轉換成文字的index因長度不同對不上，故用 1 個字元的其他符號替代
        # 王春甡 -> 王春[UNK] -> 王春?
        name = name.replace('[UNK]','?')
        name = name.replace('[PAD]','!')
        
        for i in split_index:
            name_list.append(name[0+j:len(i)+j])
            j = len(i) + j
            
        name_list = [name for name in name_list]
        label_list.append(list(set(name_list)))
    
    return label_list

if __name__ == "__main__":    
    print('ok')

