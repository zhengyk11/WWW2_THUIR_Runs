# encoding=utf8
import sys
import time
reload(sys)
sys.setdefaultencoding('utf8')
import re
import chardet
from bs4 import BeautifulSoup
import os
import warc
import sys

def main(tag, piece):
    output = open('html_urls/ntcir_back_url_'+piece+'_'+tag+'.txt', 'w')
    reading_path = '/mnt/d1/ClueWeb12/Disk1/ClueWeb12_00/'+piece+'/'+piece+'-'+tag+'.warc.7z'
    os.system('7za x %s -o/tmp/' % reading_path)
    f = warc.open('/tmp/'+piece+'-'+tag+'.warc', 'rb')
    for record in f:
        try:
            docid = record.header._d['warc-trec-id'].lower()
            print(docid)
        except:
            continue
        content = str(record.payload.read())
        #print content
        #time.sleep(8)

        content = content.lower()

        try:
            soup = BeautifulSoup(content, 'html.parser')
            anchors = soup.select('a')
            anchors_texts = [anchor.text.lower().strip() for anchor in anchors]
            anchors_text = ' '.join(anchors_texts)
            anchors_text = anchors_text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').strip()
            print anchors_text
            number_anchors = len(anchors)

        except:
            print 'fail'
            continue

        uid = record.header._d['warc-trec-id'].lower()
        output.write(uid + '\t' + record.url + '\t' + anchors_text.encode('utf-8',
                                                                                                           'ignore').strip()
                     + '\t' + str(number_anchors) + '\n')
        print 'success'
    f.close()
    os.remove('/tmp/'+piece+'-'+tag+'.warc')
    output.close()


if __name__ == '__main__':
    tags_1 = ['0', '1', '2', '3', '4']
    tags_0 = ['0', '1', '2', '3', '4']
    for tag_0 in tags_0:
        for tag_1 in tags_1:
            main(tag_0+tag_1, '0000wb')