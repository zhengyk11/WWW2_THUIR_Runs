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

def main(tag):
    output = open('html_result/ntcir_back_content_20180917_'+tag+'.txt', 'w')
    reading_path = '/mnt/d1/ClueWeb12/Disk1/ClueWeb12_00/0000tw/0000tw-'+tag+'.warc.7z'
    os.system('7za x %s -o/tmp/' % reading_path)
    f = warc.open('/tmp/0000tw-'+tag+'.warc', 'rb')
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
        content = content.split('<html', 1)
        if (len(content) == 2):
            content = '<html' + content[1]
        else:
            content = content[0].split('<rss', 1)
            if (len(content) == 2):
                content = '<rss' + content[1]
            else:
                content = content[0]
        try:
            soup = BeautifulSoup(content, 'html.parser')
            # content = soup.get_text()
            [script.extract() for script in soup.findAll('script')]
            [style.extract() for style in soup.findAll('style')]
            content = soup.get_text()
            title = soup.title.string

        except:
            print 'fail'
            continue
        if title is None:
            print 'fail'
            continue
        content = content.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').strip()
        content = ' '.join(re.split('[ ]+', content))
        title = title.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').strip()
        title = ' '.join(re.split('[ ]+', title))
        output.write(docid + '\t' + title.encode('utf-8', 'ignore').strip() + '\t' + content.encode('utf-8',
                                                                                                            'ignore').strip() + '\n')
        print 'success'
    f.close()
    os.remove('/tmp/0000tw-'+tag+'.warc')
    output.close()


if __name__ == '__main__':
    tags_1 = ['0', '1', '2', '3', '4']
    tags_0 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for tag_0 in tags_0:
        for tag_1 in tags_1:
            main(tag_0+tag_1)