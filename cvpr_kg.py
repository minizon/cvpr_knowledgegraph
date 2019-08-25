# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 23:39:48 2019

@author: RockyZhou
"""

#import requests
#
#r=requests.get('http://openaccess.thecvf.com/CVPR2019.py')
#with open('cvpr.txt','w') as f:
#    f.write(r.text)

with open('cvpr.txt','r') as f:
    txt = f.read()
    
import re

regexp = r'<dt class="ptitle">(.*?)</dt>.*?<dd>(.*?)</dd>'
paperinfos = re.findall(regexp, txt,flags=re.DOTALL)

title_regexp='<a href=".*?">(.*?)</a>'

paper_titles=[]
for pinfo in paperinfos:
    result = re.findall(title_regexp,pinfo[0])
    paper_titles.append(result[0])
    
#分词
bad_words = ['a', 'an', 'and', 'for', 'by', 'with', 'to', 'of', 'in', 'from', 'the', 'via', 'using', 'on', 'towards', 'all', '&']
#分词的原则
#1. 按空格断开
#2. 取1-4个连续词组
#3. 不计bad words，连续词组中不含bad words

from collections import defaultdict

topic_counts = defaultdict(int)

maxn_gram = 4
minn_gram = 2

for title in paper_titles:
    
    title = title.lower()
    words = title.split()
    
    words_len = len(words)
    
    for i in range(words_len):
        
        flag_valid_gram = True
        for k in range(minn_gram, maxn_gram+1):
            
            if i+k > words_len:
                break
            
#            if not flag_valid_gram:
#                break
            
            ngram = tuple(words[i:i+k])
            
            for bad_word in bad_words:
                
                if bad_word in ngram:
                    flag_valid_gram = False
                    break
                
            if flag_valid_gram:
                
                topic_counts[ngram] +=1
                
            else:
                break
    
print(len(topic_counts))

cw = sorted([(count,topic) for topic,count in topic_counts.items()], reverse=True)

lowest_freq = 4

frequent_cws = []
for item in cw:
    if item[0] >= lowest_freq:
        frequent_cws.append( (item[0], " ".join(item[1])) )
    else:
        break
    
print("There are %d phrases appearing more than %d times"%(len(frequent_cws), lowest_freq))
    
#filtering the overlapped phrases
grouped_cws = set()

for item in frequent_cws:
    
    item = (item,)  #turn to a tuple
    
    topflag = True
    
    list_g_cws = list(grouped_cws) #make a copy
    
    for query_items in list_g_cws:
        
        if item[0][1] in query_items[0][1]: #eg. "neural network" in "neural networks"
            
            grouped_cws.remove(query_items)
            
            query_items = query_items + item
            
            grouped_cws.add(query_items)
            
            topflag=False
            
        elif query_items[0][1] in item[0][1]:
            
            grouped_cws.remove(query_items)
            
            item = item + query_items
    
    if topflag:        
        grouped_cws.add(item)
            

cw = sorted([(count,topic) for topic,count in topic_counts.items()], reverse=True)
#sorted(iterable, cmp=None, key=None, reverse=False)
sorted_cws = sorted(grouped_cws, key=lambda s: s[0][0], reverse=True)
print("There are %d phrases after grouping"%(len(sorted_cws)))


##record the names
from collections import Counter

author_freq_counter = Counter()
author_regexp = '<a .*?>\s?(.*?)</a>'
authors_per_paper = []
for pinfo in paperinfos:
    authors = re.findall(author_regexp,pinfo[1])
    author_freq_counter.update(authors)
    authors_per_paper.append(authors)


ca = sorted([(count, author) for author,count in author_freq_counter.items()], reverse=True)

author_num = len(ca)
import numpy as np

author2idx = {author:i for i, author in enumerate(author_freq_counter.keys())}
idx2author = {i:author for author, i in author2idx.items()}

author_relation = np.zeros((author_num,author_num))

for authors in authors_per_paper:
    
    author_num_in_onepaper = len(authors)
    
    for i in range(author_num_in_onepaper):
        
        for j in range(i+1, author_num_in_onepaper):
    
            author_relation[author2idx[authors[i]], author2idx[authors[j]]] += 1
            
            author_relation[author2idx[authors[j]], author2idx[authors[i]]] += 1
            

relation_thresh = 2

authors_related = author_relation > relation_thresh

h, c = np.nonzero(authors_related)

author_related_decoded = {}
for i in range(len(h)):
    one_a = idx2author[h[i]]
    another_a = idx2author[c[i]]
    
    author_pair = tuple(sorted([one_a, another_a]))
    
    if author_pair not in author_related_decoded:
        author_related_decoded[author_pair] = author_relation[h[i],c[i]]
        
author_related_decoded_list = sorted([(count, author_pair) for author_pair,count in author_related_decoded.items()], reverse=True)

#for item in frequent_cws:
#    
#    matched = False
#    
#    for query_items in grouped_cws:
#        
#        if item[1] in query_items[0][1]:
#            
#            query_items.append(item)
#            matched=True
#            break
#        else:
#            for i, q_item in enumerate(query_items):
#                if q_item[1] in item[1]:
#                    query_items.insert(i, item)
#                    matched = True
#                    break
#                
#            if matched:
#                break
#        
#    if not matched:
#        
#        grouped_cws.append([item])
#            
#    
#i = 0
#while len(frequent_cws)>0:
#    
#    item = frequent_cws[0]
#    
#    item = frequent_cws.pop(index=0)
#    
#    j=0
#    while len(frequent_cws)>0:
#        
#        
#    
#    j=1
#    while j<len(frequent_cws):
#        
#        if item[1] in frequent_cws[j][1]:
#            #if the separate words are the same, do not increase the count
#            #else accumulate the counts
#            words1 = set(item[1].split())
#            words2 = set(frequent_cws[j][1].split())
#            
#            if len(words1-words2)>0:
#                item[0] += frequent_cws[j][0]
#            
#            filtered_cws.append[item]
#            
#        elif frequent_cws[j][1] in item[1]:
#            
#        
#        j+=1
#for i, item in enumerate(frequent_cws):
#    for j in range(i+1,len(frequent_cws)):
#        
#        if item[1] in frequent_cws[j][1]:
#            #if the separate words are the same, do not increase the count
#            #else accumulate the counts
#            words1 = set(item[1].split())
#            words2 = set(frequent_cws[j][1].split())
#            
#            if len(words1-words2)>0:
#                item[0] += frequent_cws[j][0]
#            
#            filtered_cws.append[item]
#                    
#            
#        elif frequent_cws[j][1] in item[1]:
#            frequent_cws[j][0] += item[0]
#
#
#def precook(s, n=4, out=False):
#  """
#  Takes a string as input and returns an object that can be given to
#  either cook_refs or cook_test. This is optional: cook_refs and cook_test
#  can take string arguments as well.
#  :param s: string : sentence to be converted into ngrams
#  :param n: int    : number of ngrams for which representation is calculated
#  :return: term frequency vector for occuring ngrams
#  """
#  words = s.split()
#  counts = defaultdict(int)
#  for k in range(1,n+1):
#    for i in range(len(words)-k+1):
#      ngram = tuple(words[i:i+k])
#      counts[ngram] += 1
#  return counts
#
#def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
#    '''Takes a list of reference sentences for a single segment
#    and returns an object that encapsulates everything that BLEU
#    needs to know about them.
#    :param refs: list of string : reference sentences for some image
#    :param n: int : number of ngrams for which (ngram) representation is calculated
#    :return: result (list of dict)
#    '''
#    return [precook(ref, n) for ref in refs]
#
#def create_crefs(refs):
#  crefs = []
#  for ref in refs:
#    # ref is a list of 5 captions
#    crefs.append(cook_refs(ref))
#  return crefs