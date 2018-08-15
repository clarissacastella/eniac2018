# coding=UTF-8
import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, SklearnClassifier
import csv
from sklearn import cross_validation
from sklearn.svm import LinearSVC, SVC
import random
from nltk.corpus import stopwords
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import *

from scipy.sparse import coo_matrix
from sklearn.utils import resample
import numpy as np

from nltk.metrics import scores



posdata = []
#with open('./data/positive-data.csv', 'rb') as myfile:    
with open('./data/gold/train_EPTC_POA_v3nbal_1.data', 'rb') as myfile:    
    reader = csv.reader(myfile, delimiter=',')
    for val in reader:
        posdata.append(val[0])        
 
negdata = []
#with open('./data/negative-data.csv', 'rb') as myfile:    
with open('./data/gold/train_EPTC_POA_v3nbal_0.data', 'rb') as myfile:    
    reader = csv.reader(myfile, delimiter=',')
    for val in reader:
        negdata.append(val[0])            

neudata = []
#with open('./data/negative-data.csv', 'rb') as myfile:    
with open('./data/gold/train_EPTC_POA_v3nbal_2.data', 'rb') as myfile:    
    reader = csv.reader(myfile, delimiter=',')
    for val in reader:
        neudata.append(val[0])

def word_split(data):    
    data_new = []
    for word in data:
        word_filter = [i.lower() for i in word.split()]
        data_new.append(word_filter)
    return data_new

def word_feats(words):    
    return dict([(word, words.count(word)) for word in words])
    
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out    
    
def gettrainfeat(l, n):
	out = []
	for c in range(0,len(l)-1):
		if (c != n): 
			out = out + l[c]     
 	return out
 	
# Calculating Precision, Recall & F-measure
def evaluate_classifier(featx, balance=False):
    global negdata
    global neudata
    global posdata
	
    if balance:

        neudata = resample(neudata,n_samples=len(negdata))
        posdata = resample(posdata,n_samples=len(negdata))


    # using 3 classifiers
    classifier_list = ['svm','nb', 'maxent' ] 
          
    negfeats = [(featx(f), 'neg') for f in word_split(negdata)]
    posfeats = [(featx(f), 'pos') for f in word_split(posdata)]
    neufeats = [(featx(f), 'neu') for f in word_split(neudata)]
    alldata = negdata + posdata + neudata
    allfeats = negfeats + posfeats + neufeats           
           
             
    #10-fold cross-validation  
    correct = []
    incorrect = []
    for n in [10]: #range(2,6):          
        negfeatssplit = chunkIt(negfeats, n)
        negdatasplit = chunkIt(negdata, n)             
        posfeatssplit = chunkIt(posfeats, n)
        posdatasplit = chunkIt(posdata, n)             
        neufeatssplit = chunkIt(neufeats, n)
        neudatasplit = chunkIt(neudata, n)             
        for cl in classifier_list:           
            accuracy = []
            pos_precision = []
            pos_recall = []
            neg_precision = []
            neg_recall = []
            neu_precision = []
            neu_recall = []
            pos_fmeasure = []
            neg_fmeasure = []
            neu_fmeasure = []
            cv_count = 1
            res = {}
            res["neg"] = 0
            res["pos"] = 0
            res["neu"] = 0

            for i in range(n):        
                testing_this_round = negfeatssplit[i-1] + posfeatssplit[i-1] + neufeatssplit[i-1] 
                training_this_round = gettrainfeat(negfeatssplit, i) + gettrainfeat(posfeatssplit, i) + gettrainfeat(neufeatssplit, i) 
                
                if cl == 'maxent':
                    classifierName = 'Maximum Entropy'
                    classifier = MaxentClassifier.train(training_this_round, 'GIS', trace=0, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 1)
                elif cl == 'svm':
                    classifierName = 'SVM'
                    classifier = SklearnClassifier(LinearSVC(), sparse=False)
                    classifier.train(training_this_round)
                else:
                    classifierName = 'Naive Bayes'
                    classifier = NaiveBayesClassifier.train(training_this_round)
					                        
                refsets = collections.defaultdict(set)
                testsets = collections.defaultdict(set)
                aux_test = {}
                auxFP_test = {}
                aux_test['pos'] = 0
                aux_test['neu'] = 0
                aux_test['neg'] = 0
                auxFP_test['pos'] = 0
                auxFP_test['neu'] = 0
                auxFP_test['neg'] = 0
                for ii, (feats, label) in enumerate(testing_this_round):
                    refsets[label].add(ii)
                    observed = classifier.classify(feats)
                    testsets[observed].add(ii)
                    res[observed] = res[observed] + 1
                    auxFP_test[observed] = auxFP_test[observed] + 1
                    if (observed == label) :
                    	correct.append((feats,label))
                    	aux_test[label] = aux_test[label] + 1 
                    else:
                    	incorrect.append((feats,label))   
                    	
                    	
                cv_accuracy = nltk.classify.util.accuracy(classifier, testing_this_round)  
                cv_neg_precision = float(aux_test['neg'])/float(len(negfeatssplit[i-1]))
                print cv_neg_precision
               
                cv_neg_recall = float(aux_test['neg'])/float(auxFP_test['neg'])
                cv_neg_fmeasure = 2*((cv_neg_precision*cv_neg_recall)/(cv_neg_precision+cv_neg_recall))
                cv_pos_precision = float(aux_test['pos'])/float(len(posfeatssplit[i-1]))
                cv_pos_recall = float(aux_test['pos'])/float(auxFP_test['pos'])
                cv_pos_fmeasure = 2*((cv_pos_precision*cv_pos_recall)/(cv_pos_precision+cv_pos_recall))
                cv_neu_precision = float(aux_test['neu'])/float(len(neufeatssplit[i-1]))
                cv_neu_recall = float(aux_test['neu'])/float(auxFP_test['neu'])
                cv_neu_fmeasure = 2*((cv_neu_precision*cv_neu_recall)/(cv_neu_precision+cv_neu_recall))
                #cv_accuracy = float(aux_test['neg'] + aux_test['pos']+ aux_test['neu'])/float(len(testing_this_round))

                accuracy.append(cv_accuracy)
                pos_precision.append(cv_pos_precision)
                neg_precision.append(cv_neg_precision)
                neu_precision.append(cv_neu_precision)
                pos_recall.append(cv_pos_recall)
                neg_recall.append(cv_neg_recall)
                neu_recall.append(cv_neu_recall)
                pos_fmeasure.append(cv_pos_fmeasure)
                neg_fmeasure.append(cv_neg_fmeasure)
                neu_fmeasure.append(cv_neu_fmeasure)
                
                cv_count += 1


            print "Balance = ", balance 
            print '---------------------------------------'
            print str(n)+'-FOLD CROSS VALIDATION RESULT ' + '(' + classifierName + ')' 
            print "Nbr = ", res
            print 'accuracy:', sum(accuracy) / n
            print 'precision', ((sum(pos_precision)/n) + (sum(neg_precision)/n) + (sum(neu_precision)/n)) / 3.0
            print sum(pos_precision)/n , sum(neg_precision)/n, sum(neu_precision)/n
            print 'recall', (sum(pos_recall)/n + sum(neg_recall)/n + sum(neu_recall)/n) / 3.0
            print sum(pos_recall)/n , sum(neg_recall)/n, sum(neu_recall)/n
            print 'f-measure', (sum(pos_fmeasure)/n + sum(neg_fmeasure)/n + sum(neu_fmeasure)/n) / 3.0
            print sum(pos_fmeasure)/n , sum(neg_fmeasure)/n, sum(neu_fmeasure)/n        


            print "*********CORRECT****"	
            print (len(correct),len(incorrect))
            #print (correct,incorrect)
            
            for tt in correct:
            	print (tt[1],alldata[allfeats.index(tt)])
            print "***INCORRECT**********"	
            for tt in incorrect:
            	print (tt[1],alldata[allfeats.index(tt)]) #.index(correct[0]))
            print "..."
            
evaluate_classifier(word_feats, False)
evaluate_classifier(word_feats, True)
