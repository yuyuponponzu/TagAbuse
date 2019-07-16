# coding: utf-8
import csv

"""
Fetch questions tags and body using the obtained question IDs.
"""

from stackapi import StackAPI
import math

datafile = open('dataset_test.txt', 'w')
datafile_InitialTag = open('dataset_ITag_test.txt', 'w')
"""
SITE = StackAPI('stackoverflow')

# Number of requests required, each request can process upto 100 IDs
num_rounds = math.ceil(len(question_ids)/100)
"""
question_delimiter = "\n|||||\n"
category_delimiter = "\n;;;;;\n"
tags_delimiter = " "

f_list = ["./testdata.csv", "testdata_noc.csv"]

#csvの数を設定
for k in f_list:
    csv_file = open(k, "r", encoding ="Latin-1")
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
            datafile.write(str(row['Title']) + str(row['Body']) + category_delimiter)
            datafile.write(question_delimiter)
            datafile_InitialTag.write(str(row['Id']) + category_delimiter)
            datafile_InitialTag.write(" ".join(I_tag) + category_delimiter)
            datafile_InitialTag.write(question_delimiter)
        j = j+1
