"""
train.txt

M      K    Accuracy
1     0.05.   99.23
1     0.1     99.17
5.    0.5.    98.408

dev.txt

M     K    Accuracy
5    0.5   97.127
1    0.5   97.307
1.   0.05. 97.307

test.txt

M   K     Accuracy
1   1      98.204
1   0.1    98.0251
2.  0.1.   97.84

After removing stopwards

train.txt

M   K     Accuracy
1.  0.05.  99.304
2   1      98.789
1   0.1    98.0251

dev.txt

M   K     Accuracy
1  0.01   97.127
1  0.05   96.947
2. 0.001. 97.307

test.txt

M   K     Accuracy
1  0.01   97.127
2  0.001  98.025
1  0.05   98.204

"""

import sys
import string
import math
from collections import Counter
class NbClassifier(object):

    """
    A Naive Bayes classifier object has three parameters, all of which are populated during initialization:
    - a set of all possible attribute types
    - a dictionary of the probabilities P(Y), labels as keys and probabilities as values
    - a dictionary of the probabilities P(F|Y), with (feature, label) pairs as keys and probabilities as values
    """
    def __init__(self, training_filename, stopword_file):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}   

        self.collect_attribute_types(training_filename) 
        if stopword_file is not None:
            self.remove_stopwords(stopword_file)
        self.train(training_filename)


    """
    A helper function to transform a string into a list of word strings.
    You should not need to modify this unless you want to improve your classifier in the extra credit portion.
    """
    def extract_words(self, text):
        no_punct_text = "".join([x for x in text.lower() if not x in string.punctuation])
        return [word for word in no_punct_text.split()]


    """
    Given a stopword_file, read in all stop words and remove them from self.attribute_types
    Implement this for extra credit.
    """
    def remove_stopwords(self, stopword_file):

        stop_file=open(stopword_file)
        #print(self.attribute_types)
        my_word=stop_file.read()
        my_list=my_word.split('\n')
        self.attribute_types=self.attribute_types.difference(set(my_list))
        #print(len(self.attribute_types))

    """
    Given a training datafile, add all features that appear at least m times to self.attribute_types
    """
    def collect_attribute_types(self, training_filename, m=1
        ):
        self.attribute_types = set()
        my_file=open(training_filename,'r')
        
        i=my_file.read()
        sorted=self.replace_punctuation(i)
        final=self.extract_words(sorted)
    
        counts=Counter(final)
        d=dict((k,v) for k,v in counts.items() if v>=m)
        self.attribute_types=set(list(d.keys()))


        my_file.close()

        


    #Below are my functions
    
     
    def replace_punctuation(self,training_filename):

    
        translator=str.maketrans(string.punctuation,' '*len(string.punctuation))
        #print(training_filename.translate(translator))
        return training_filename.translate(translator)

    def mail_sort(self,training_filename):
        hamlist=[]
        spamlist=[]
        hamcount=0
        spamcount=0


        my_file=open(training_filename,'r')
        for line in my_file:
            array=line.split('\t')
            if (array[0])=='ham':
                new_line=self.replace_punctuation(array[1])
                line_list=self.extract_words(new_line)
                hamcount+=len(line_list)
                #hamcount+=len(array[1].split())
                line=new_line.replace('\t',' ') 
                line=new_line.replace('\n',' ')
                #line=line
                hamlist.append(new_line.lower())
            elif (array[0])=='spam':
                new_line=self.replace_punctuation(array[1])
                line_list=self.extract_words(new_line)
                spamcount+=len(line_list)
                #spamcount+=len(line.split())
                line=new_line.replace('\t',' ')
                line=new_line.replace('\n',' ')
                spamlist.append(new_line.lower())
            
        hamcount=hamcount-len(hamlist)
        spamcount=spamcount-len(spamlist)

        my_file.close()
        return hamcount,spamcount,hamlist,spamlist


    """
    Given a training datafile, estimate the model probability parameters P(Y) and P(F|Y).
    Estimates should be smoothed using the smoothing parameter k.
    """
    def train(self, training_filename, k=0.05):

        self.label_prior = {}
        self.word_given_label = {}
        word_count=0


        hamcount,spamcount,hamlist,spamlist=self.mail_sort(training_filename)

        ham_prob=len(hamlist)/(len(hamlist)+len(spamlist))
        spam_prob=len(spamlist)/(len(hamlist)+len(spamlist))
        self.label_prior["ham"]=ham_prob
        self.label_prior["spam"]=spam_prob
     

        for attr in self.attribute_types:
            for line in hamlist:
                word_count+=line.count(attr)
            self.word_given_label[(attr,'ham')]=(word_count+k)/(hamcount+k*len(self.attribute_types))
            word_count=0
            for line in spamlist:
                word_count+=line.count(attr)  
            self.word_given_label[(attr,'spam')]=(word_count+k)/(spamcount+k*len(self.attribute_types))
            word_count=0


    """
    Given a piece of text, return a relative belief distribution over all possible labels.
    The return value should be a dictionary with labels as keys and relative beliefs as values.
    The probabilities need not be normalized and may be expressed as log probabilities. 
    """
    def predict(self, text):

        word_list=self.extract_words(text)
        #print(word_list)
        ham_sum=0
        spam_sum=0
        log_ham_prob=0
        log_spam_prob=0

        ham_log=math.log(self.label_prior["ham"])
        spam_log=math.log(self.label_prior["spam"])

        ham_dictlist={k:v for (k,v) in self.word_given_label.items() if k[1]=="ham"}
        spam_dictlist={k:v for (k,v) in self.word_given_label.items() if k[1]=="spam"}

        for word in word_list:
            if (word,"ham") in ham_dictlist:
                ham_sum+=math.log(ham_dictlist[(word,"ham")])
            if (word,"spam") in spam_dictlist:
                spam_sum+=math.log(spam_dictlist[(word,"spam")])

        log_ham_prob = ham_log+ham_sum
        log_spam_prob=spam_log+spam_sum

        predict_dict = {"ham": log_ham_prob, "spam":log_spam_prob}


        return predict_dict


    """
    Given a datafile, classify all lines using predict() and return the accuracy as the fraction classified correctly.
    """
    def evaluate(self, test_filename):

        accurate_counter=0
        line_num=0

        my_file=open(test_filename,'r')
        my_line=my_file.readline()
        #print(my_line)
        while my_line:
            line_num+=1
            my_line=self.replace_punctuation(my_line)
            second_line=my_line.split('\t')[1]
            mod_line=self.replace_punctuation(second_line)
            #print(second_line)
            predict_dict=self.predict(mod_line)
            
        
 
            if my_line.split('\t')[0]=="ham" and predict_dict["ham"]>predict_dict["spam"]:
                accurate_counter+=1
            elif my_line.split('\t')[0]=="spam" and predict_dict["spam"]>predict_dict["ham"]:
                accurate_counter+=1


            my_line=my_file.readline()

        final_accuracy=accurate_counter/line_num


        my_file.close()

        return final_accuracy


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("\nusage: ./hmm.py [training data file] [test or dev data file] [(optional) stopword file]")
        exit(0)
    elif len(sys.argv) == 3:
        classifier = NbClassifier(sys.argv[1], None)
    else:
        classifier = NbClassifier(sys.argv[1], sys.argv[3])
    print(classifier.evaluate(sys.argv[2]))