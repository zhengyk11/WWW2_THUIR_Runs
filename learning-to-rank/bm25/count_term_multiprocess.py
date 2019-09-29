import os
import sys
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer

tag = int(sys.argv[1])
split_num = 16


def getStemResult(text):
    words = word_tokenize(text.lower())

    stopwordset = stopwords.words('english')
    puncts = [',', '.', '!', '?', '&']
    clean_list = [token for token in words if token not in stopwordset]
    clean_list = [token for token in clean_list if token not in puncts]

    porter = PorterStemmer()
    result = [porter.stem(w) for w in clean_list]
    return result

if __name__ == '__main__':
    file_term = open('term_list.txt', 'r')
    terms = {}
    while True:
        term = file_term.readline().decode('utf-8').strip()
        if term:
            terms[term] = 0
        else:
            break

    path = 'html_result'
    for dirpath, dirnames, filenames in os.walk(path):
        count_file = -1
        for file in filenames:
            count_file += 1
            if count_file % 16 == tag:
                print 'process@',file
                file_path = path + '/' + file
                f = open(file_path, 'r')
                count = 0
                count_null = 0
                while True:
                    line = f.readline().decode('utf-8')
                    if line:
                        count += 1
                        count_null = 0
                        docid, title, content = line.split('\t', 2)
                        contents = getStemResult(content)
                        # contents = getStemResult(title)
                        for term in terms:
                            if term in contents:
                                terms[term] += 1
                        if count % 10 == 0:
                            print 'go on #', count
                    else:
                        count_null += 1
                        if count_null >= 100:
                            break
                print count
                f.close()

    file_out = open('count_content/term_idf_count2_'+str(tag)+'.txt', 'w')
    for term in terms:
        print term, ': ', terms[term]
        file_out.write(term+'\t'+str(terms[term])+'\n')