# coding: utf-8
import csv

# Stores question IDs
question_ids = []

# Limit on how many question IDs to retrieve
num_ids = 15000

"""
Our dataset is based on the StackLite dataset released by Stackoverflow on Kaggle.
https://www.kaggle.com/stackoverflow/stacklite/data
But the original dataset contains deleted items as well. 
We are only going to consider undeleted posts here.
"""
# NOTE: Make sure you have StackLite folder installed in the same directory as this file.
"""
curr_cnt = 0
with open('stacklite/questions.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        question_ids.append(row[0])
        curr_cnt += 1
        if curr_cnt == num_ids:
            break
"""

"""
Fetch questions tags and body using the obtained question IDs.
"""

from stackapi import StackAPI
import math

datafile = open('dataset.txt', 'w')
datafile_InitialTag = open('dataset_ITag.txt', 'w')
"""
SITE = StackAPI('stackoverflow')

# Number of requests required, each request can process upto 100 IDs
num_rounds = math.ceil(len(question_ids)/100)
"""
question_delimiter = "\n|||||\n"
category_delimiter = "\n;;;;;\n"
tags_delimiter = " "
"""
for round_i in range(num_rounds):
    start = (round_i*100)
    end = min(len(question_ids), (round_i+1)*100)
    questions = SITE.fetch('questions', ids=question_ids[start:end], filter='withbody')
    
    for item in questions['items']:
        datafile.write(str(item['question_id']) + category_delimiter)
        datafile.write(" ".join(item['tags']) + category_delimiter)
        datafile.write(str(item['body']) + category_delimiter)
        datafile.write(question_delimiter)
        datafile_InitialTag.write(str(item['question_id']) + category_delimiter)
        datafile_InitialTag.write(" ".join(item['tags']) + category_delimiter)
        datafile_InitialTag.write(question_delimiter)
"""
#csvの数を設定
csv_file = open("./dataset.csv", "r", encoding ="Latin-1")
f = csv.DictReader(csv_file)
j=0
for row in f:
    tag = []
    a = row['Tags']
    a = a.split('>')
    for i in range(len(a)-1):
        b = a[i].split('<')
        tag.append(b[1])
    I_tag = []
    aa = row['I_Tags']
    aa = aa.split('>')
    for i in range(len(aa)-1):
        bb = aa[i].split('<')
        I_tag.append(bb[1])
    if tag != I_tag :
        datafile.write(str(row['Id']) + category_delimiter)
        datafile.write(" ".join(tag) + category_delimiter)
        datafile.write(str(row['Title']) + str(row['Bodys']) + category_delimiter)
        datafile.write(question_delimiter)
        datafile_InitialTag.write(str(row['Id']) + category_delimiter)
        datafile_InitialTag.write(" ".join(I_tag) + category_delimiter)
        datafile_InitialTag.write(question_delimiter)
    j = j+1
