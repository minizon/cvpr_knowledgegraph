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


class IntegralOccupancyMap(object):
    def __init__(self, height, width, mask):
        self.height = height
        self.width = width
        if mask is not None:
            # the order of the cumsum's is important for speed ?!
            self.integral = np.cumsum(np.cumsum(255 * mask, axis=1),
                                      axis=0).astype(np.uint32)
        else:
            self.integral = np.zeros((height, width), dtype=np.uint32)

    def sample_position(self, size_x, size_y, random_state):
        return query_integral_image(self.integral, size_x, size_y,
                                    random_state)

    def update(self, img_array, pos_x, pos_y):
        partial_integral = np.cumsum(np.cumsum(img_array[pos_x:, pos_y:],
                                               axis=1), axis=0)
        # paste recomputed part into old image
        # if x or y is zero it is a bit annoying
        if pos_x > 0:
            if pos_y > 0:
                partial_integral += (self.integral[pos_x - 1, pos_y:]
                                     - self.integral[pos_x - 1, pos_y - 1])
            else:
                partial_integral += self.integral[pos_x - 1, pos_y:]
        if pos_y > 0:
            partial_integral += self.integral[pos_x:, pos_y - 1][:, np.newaxis]

        self.integral[pos_x:, pos_y:] = partial_integral



from random import Random
def generate_from_frequencies(frequencies, max_font_size=None):  # noqa: C901
    """Create a word_cloud from words and frequencies.
    Parameters
    ----------
    frequencies : dict from string to float
        A contains words and associated frequency.
    max_font_size : int
        Use this font-size instead of self.max_font_size
    Returns
    -------
    self
    """
    # make sure frequencies are sorted and normalized
    frequencies = sorted(frequencies.items(), key=itemgetter(1), reverse=True)
    if len(frequencies) <= 0:
        raise ValueError("We need at least 1 word to plot a word cloud, "
                         "got %d." % len(frequencies))
    frequencies = frequencies[:self.max_words]

    # largest entry will be 1
    max_frequency = float(frequencies[0][1])

    frequencies = [(word, freq / max_frequency)
                   for word, freq in frequencies]

    if self.random_state is not None:
        random_state = self.random_state
    else:
        random_state = Random()


    boolean_mask = None
    height, width = 200, 400
    occupancy = IntegralOccupancyMap(height, width, boolean_mask)

    # create image
    img_grey = Image.new("L", (width, height))
    draw = ImageDraw.Draw(img_grey)
    img_array = np.asarray(img_grey)
    font_sizes, positions, orientations, colors = [], [], [], []

    last_freq = 1.

    if max_font_size is None:
        # if not provided use default font_size
        max_font_size = self.max_font_size

    if max_font_size is None:
        # figure out a good font size by trying to draw with
        # just the first two words
        if len(frequencies) == 1:
            # we only have one word. We make it big!
            font_size = self.height
        else:
            self.generate_from_frequencies(dict(frequencies[:2]),
                                           max_font_size=self.height)
            # find font sizes
            sizes = [x[1] for x in self.layout_]
            try:
                font_size = int(2 * sizes[0] * sizes[1]
                                / (sizes[0] + sizes[1]))
            # quick fix for if self.layout_ contains less than 2 values
            # on very small images it can be empty
            except IndexError:
                try:
                    font_size = sizes[0]
                except IndexError:
                    raise ValueError(
                        "Couldn't find space to draw. Either the Canvas size"
                        " is too small or too much of the image is masked "
                        "out.")
    else:
        font_size = max_font_size

    # we set self.words_ here because we called generate_from_frequencies
    # above... hurray for good design?
    self.words_ = dict(frequencies)

    if self.repeat and len(frequencies) < self.max_words:
        # pad frequencies with repeating words.
        times_extend = int(np.ceil(self.max_words / len(frequencies))) - 1
        # get smallest frequency
        frequencies_org = list(frequencies)
        downweight = frequencies[-1][1]
        for i in range(times_extend):
            frequencies.extend([(word, freq * downweight ** (i + 1))
                                for word, freq in frequencies_org])

    # start drawing grey image
    for word, freq in frequencies:
        if freq == 0:
            continue
        # select the font size
        rs = self.relative_scaling
        if rs != 0:
            font_size = int(round((rs * (freq / float(last_freq))
                                   + (1 - rs)) * font_size))
        if random_state.random() < self.prefer_horizontal:
            orientation = None
        else:
            orientation = Image.ROTATE_90
        tried_other_orientation = False
        while True:
            # try to find a position
            font = ImageFont.truetype(self.font_path, font_size)
            # transpose font optionally
            transposed_font = ImageFont.TransposedFont(
                font, orientation=orientation)
            # get size of resulting text
            box_size = draw.textsize(word, font=transposed_font)
            # find possible places using integral image:
            result = occupancy.sample_position(box_size[1] + self.margin,
                                               box_size[0] + self.margin,
                                               random_state)
            if result is not None or font_size < self.min_font_size:
                # either we found a place or font-size went too small
                break
            # if we didn't find a place, make font smaller
            # but first try to rotate!
            if not tried_other_orientation and self.prefer_horizontal < 1:
                orientation = (Image.ROTATE_90 if orientation is None else
                               Image.ROTATE_90)
                tried_other_orientation = True
            else:
                font_size -= self.font_step
                orientation = None

        if font_size < self.min_font_size:
            # we were unable to draw any more
            break

        x, y = np.array(result) + self.margin // 2
        # actually draw the text
        draw.text((y, x), word, fill="white", font=transposed_font)
        positions.append((x, y))
        orientations.append(orientation)
        font_sizes.append(font_size)
        colors.append(self.color_func(word, font_size=font_size,
                                      position=(x, y),
                                      orientation=orientation,
                                      random_state=random_state,
                                      font_path=self.font_path))
        # recompute integral image
        if self.mask is None:
            img_array = np.asarray(img_grey)
        else:
            img_array = np.asarray(img_grey) + boolean_mask
        # recompute bottom right
        # the order of the cumsum's is important for speed ?!
        occupancy.update(img_array, x, y)
        last_freq = freq

    self.layout_ = list(zip(frequencies, font_sizes, positions,
                            orientations, colors))
    return self

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