# coding=utf-8

import warc
import os
import sys

tag = int(sys.argv[1])
split_num = 6
print tag #tag:0-5
docid_list = {}
file = open('./docid13_eng_9.txt', 'r')
for line in file:
    attr = line.strip().split('-')
    if attr[1] not in docid_list:
        docid_list[attr[1]] = {}
    if attr[2] not in docid_list[attr[1]]:
        docid_list[attr[1]][attr[2]] = []
    docid_list[attr[1]][attr[2]].append(line.strip())

file.close()
print len(docid_list)

path_list = {'00': '/mnt/d1/ClueWeb12/Disk1/ClueWeb12_00/',
             '01': '/mnt/d1/ClueWeb12/Disk1/ClueWeb12_01/',
             '02': '/mnt/d1/ClueWeb12/Disk1/ClueWeb12_02/',
             '03': '/mnt/d1/ClueWeb12/Disk1/ClueWeb12_03/',
             '04': '/mnt/d1/ClueWeb12/Disk1/ClueWeb12_04/',
             '05': '/mnt/d1/ClueWeb12/Disk2/ClueWeb12_05/',
             '06': '/mnt/d1/ClueWeb12/Disk2/ClueWeb12_06/',
             '07': '/mnt/d1/ClueWeb12/Disk2/ClueWeb12_07/',
             '08': '/mnt/d1/ClueWeb12/Disk2/ClueWeb12_08/',
             '09': '/mnt/d1/ClueWeb12/Disk2/ClueWeb12_09/',
             '10': '/mnt/d1/ClueWeb12/Disk3/ClueWeb12_10/',
             '11': '/mnt/d1/ClueWeb12/Disk3/ClueWeb12_11/',
             '12': '/mnt/d1/ClueWeb12/Disk3/ClueWeb12_12/',
             '13': '/mnt/d1/ClueWeb12/Disk3/ClueWeb12_13/',
             '14': '/mnt/d1/ClueWeb12/Disk3/ClueWeb12_14/',
             '15': '/mnt/d1/ClueWeb12/Disk4/ClueWeb12_15/',
             '16': '/mnt/d1/ClueWeb12/Disk4/ClueWeb12_16/',
             '17': '/mnt/d1/ClueWeb12/Disk4/ClueWeb12_17/',
             '18': '/mnt/d1/ClueWeb12/Disk4/ClueWeb12_18/',
             '19': '/mnt/d1/ClueWeb12/Disk4/ClueWeb12_19/'}

output = open('id_url/id_url_'+str(tag)+'.txt', 'w')
print len(docid_list)
count_i = 0
for i in docid_list:
    count_i += 1
    if count_i % split_num != tag:
        continue
    for j in docid_list[i]:
        path = path_list[i[:2]] + i + '/' + i + '-' + j + '.warc.7z'
        tmp_list = [int(t.split('-')[3]) for t in docid_list[i][j]]
        max_line_num = max(tmp_list)
        print '#', path, len(tmp_list)
        os.system('7za x %s -o/tmp/' % path)
        f = warc.open('/tmp/' + i + '-' + j + '.warc', 'rb')
        for record in f:
            try:
                docid = record.header._d['warc-trec-id'].lower()
                print(docid)
            except:
                continue
            if int(docid.split('-')[3]) > max_line_num:
                break
            if docid in docid_list[i][j]:
                try:
                    output.write(docid + '\t' + path + '\t' + record.url + '\n')
                except:
                    pass
        f.close()
        os.remove('/tmp/' + i + '-' + j + '.warc')
output.close()
