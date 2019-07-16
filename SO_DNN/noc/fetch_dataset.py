# coding: utf-8
import csv

# Stores question IDs
question_ids = []

# Limit on how many question IDs to retrieve
num_ids = 15000


"""
Fetch questions tags and body using the obtained question IDs.
"""
import chardet
import math

datafile = open('dataset.txt', 'w')
datafile_InitialTag = open('dataset_ITag.txt', 'w')

question_delimiter = "\n|||||\n"
category_delimiter = "\n;;;;;\n"
tags_delimiter = " "
"""
with open("./noc_DATASET_1.csv", 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
"""
#csvの数を設定

csv_file = open("./dataset.csv", "r", encoding ="Latin-1")
f = csv.DictReader(csv_file)
for row in f:
    tag = []
    a = row['Tags']
    a = a.split('>')
    for i in range(len(a)-1):
        b = a[i].split('<')
        tag.append(b[1])
    datafile.write(str(row['Id']) + category_delimiter)
    datafile.write(" ".join(tag) + category_delimiter)
    datafile.write(str(row['Title']) + str(row['Bodys']) + category_delimiter)
    datafile.write(question_delimiter)
