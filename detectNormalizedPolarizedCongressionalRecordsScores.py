
# coding: utf-8

# #!/home/ubuntu/anaconda3/bin//python
# '''
# MIT License
# 
# Copyright (c) 2018 Riya Dulepet <riyadulepet123@gmail.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# '''

# ##### import elasticsearch libraries to search/query and also initialize

# In[50]:


from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl import Q

es = Elasticsearch()
es_congressional_records = Search(using=es, index="congressional_records")


# ##### import support utilities

# In[75]:


from functools import reduce, partial
import random
from datetime import datetime
import pandas as pd
import json
import numpy as np


# ##### import spacy text processing and initialize

# In[52]:


import spacy
import textacy
nlp = spacy.load('en_core_web_sm')


# ##### define GLOBAL CONSTANTS that can be tweaked

# In[53]:


ACCEPTABLE_EMOTION_THRESHOLD = 0.5
ACCEPTABLE_SENTIMENT_THRESHOLD = -0.3
ACCEPTABLE_EMOTION_SENTENCES_COUNT_THRESHOLD = 3
ACCEPTABLE_MIN_NUM_WORDS_PER_SENTENCE = 8
NUMBER_OF_SEARCH_RECORDS = 100
REPUBLICAN_LABEL = "republican"
DEMOCRATIC_LABEL = "democrat"
NUM_TRIALS = 5
ES_RESULTS_LIMIT = 2000


# ##### import watson API libraries that provide emotional tones and initialize

# In[54]:


from watson_developer_cloud import ToneAnalyzerV3
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1   import Features, EntitiesOptions, KeywordsOptions, EmotionOptions, SentimentOptions

natural_language_understanding = NaturalLanguageUnderstandingV1(
  username='XXXXXXXXXX',
  password='YYYYYYYYYY',
  version='2018-03-16')

tone_analyzer = ToneAnalyzerV3(
    version ='2017-09-21',
    username='XXXXXXXXXX',
    password='YYYYYYYYYY',
    url='https://gateway.watsonplatform.net/tone-analyzer/api'
)
content_type = 'application/json'


# ##### key word/phrase patterns to identify mentions of political affiliations
#     we want literally specific variations of these words to avoid false positives)

# In[55]:


republican = ["rnc", "gop", "republican", "republicans", "conservative",               "conservatives", "right wing", "alt right", "far right"]
democrat = ["dnc", "democrat", "democrats", "democratic", "liberal",             "liberals", "progressive", "progressives", "moderates",             "nonconservative", "nonconservatives", "alt left", "far left", "left wing"]
any_party = republican + democrat


# ##### create elasticsearch search terms for political affiliations

# In[56]:



# --- MAP-REDUCE
map_republican_query = map(lambda v :  Q("term", speech=v), republican)
reduced_republican_query = reduce(lambda x, y: x | y, map_republican_query)

map_democratic_query = map(lambda v :  Q("term", speech=v), democrat)
reduced_democratic_query = reduce(lambda x, y: x | y, map_democratic_query)


map_any_party_query = map(lambda v :  Q("term", speech=v), any_party)
reduced_any_party_query = reduce(lambda x, y: x | y, map_any_party_query)

# q = Q("match", speech='rnc') | Q("match", speech='gop') | Q("match", speech='republican')


# ##### test the output to ensure proper query formation

# In[57]:


# print(reduced_republican_query)
# print(reduced_democratic_query)
print(reduced_any_party_query)


# ##### initialize decades array so we can extract relevant information and draw insight by those

# In[58]:


# decades = [{'start': '2006-01-01', 'end': str(datetime.now().date())}]

decades = [{'start': '1990-01-01', 'end': '1994-12-31'},            {'start': '1995-01-01', 'end': '1999-12-31'},            {'start': '2000-01-01', 'end': '2004-12-31'},            {'start': '2005-01-01', 'end': '2009-12-31'},            {'start': '2010-01-01', 'end': str(datetime.now().date())}]

'''
sample_2_decades = [{'start': '1910-01-01', 'end': '1919-12-31'}, \
           {'start': '1930-01-01', 'end': '1949-12-31'}, \
           {'start': '1960-01-01', 'end': '1979-12-31'}, \
           {'start': '1990-01-01', 'end': '2009-12-31'}, \
           {'start': '2010-01-01', 'end': str(datetime.now().date())}
          ]

old_decades = [{'start': '1900-01-01', 'end': '1909-12-31'}, \
          {'start': '1910-01-01', 'end': '1919-12-31'}, \
          {'start': '1920-01-01', 'end': '1929-12-31'}, \
          {'start': '1930-01-01', 'end': '1939-12-31'}, \
          {'start': '1940-01-01', 'end': '1949-12-31'}, \
          {'start': '1950-01-01', 'end': '1959-12-31'}, \
          {'start': '1960-01-01', 'end': '1969-12-31'}, \
          {'start': '1970-01-01', 'end': '1979-12-31'}, \
          {'start': '1980-01-01', 'end': '1989-12-31'}, \
          {'start': '1990-01-01', 'end': '1999-12-31'}, \
          {'start': '2000-01-01', 'end': '2009-12-31'}, \
          {'start': '2010-01-01', 'end': str(datetime.now().date())}
          ]
'''


# ### def checkIfPartySentence identify mention of any party

# In[59]:


def checkIfPartySentence(sent):
    #if any(word in sent.lower() for word in republican):
        #return True
    #elif any(word in sent.lower() for word in democrat):
        #return True
    # print("checkIfPartySentence=", sent)
    if any(word in sent.lower() for word in any_party):
        return True
    return False


# ### iterate by the decade and extract emotion statistics based by the decade

# In[185]:


def extractEmotionalSentencesAndTabulate(query_date_range, es_search_object, nlp_handle, nlu_handle, tone_analyzer_handle, query_terms, trial_num):
    # refer to global vars
    global ACCEPTABLE_EMOTION_THRESHOLD
    global ACCEPTABLE_EMOTION_SENTENCES_COUNT_THRESHOLD
    global NUMBER_OF_SEARCH_RECORDS
    global ES_RESULTS_LIMIT
    global ACCEPTABLE_SENTIMENT_THRESHOLD
    global ACCEPTABLE_MIN_NUM_WORDS_PER_SENTENCE

    # fetch results that match query filtered by the date range specified
    filteredQuery = es_search_object.query(query_terms).filter('range', speech_date={'gte': query_date_range["start"], 'lte': query_date_range["end"]})
    
    # ES limits by default the number of results to 10, increasing it to 200 (reasons will be clear later)
    responseFilteredQuery = filteredQuery[0:ES_RESULTS_LIMIT].execute()
    total_actual_records = responseFilteredQuery.hits.total
    total_fetched_records = len(responseFilteredQuery)
    

    # track the sentences within congressional records that the machine annotated as emotional
    # for human verification, quality improvement, and most importantly as evidence to
    # build trust
    all_sample_records_emotion_party_sentences = []
    all_sample_records_emotion_no_party_sentences = []
    all_sample_records = []

    # look for random 10 records out of the max 200 records fetched to build estimates of emotion statistics
    for x in range(0, NUMBER_OF_SEARCH_RECORDS):
        
        # generate a random number
        random_index = random.randint(0, total_fetched_records - 1)
        # process NLP on the text, primarily to extract sentences most reliabily
        doc = nlp(responseFilteredQuery[random_index].speech)
        
        # transform data for Watson customer engagement tone analyzer
        utterances = []
        num_party_sentences = 0
        num_no_party_sentences = 0
        for sent in doc.sents:
            # ignore any sentence less than threshold number of words to reduce false positives
            ts = textacy.TextStats(nlp(sent.text))
            # print(ts.n_words)
            # check if party sentence, likely indicator of polarization sentence
            if ts.n_words >= ACCEPTABLE_MIN_NUM_WORDS_PER_SENTENCE:
                utterances.append({"text":sent.text, "user": "customer"})
                if checkIfPartySentence(sent.text):
                    num_party_sentences += 1
                else:
                    num_no_party_sentences += 1

        # extract emotions from Watson Customer Engagement Tone Analyzer API
        # for higher accuracy process on each sentence
        map_congressional_record_party_sentiment_and_emotion = {'sad':[], 'frustrated':[], 'impolite':[],                                                           'negative_sentiment':[]                                                                 # 'sadness':[], 'fear':[], 'disgust':[], 'anger':[] \
                                                               }
        map_congressional_record_no_party_sentiment_and_emotion = {'sad':[], 'frustrated':[], 'impolite':[],                                                           'negative_sentiment':[]                                                                    #'sadness':[], 'fear':[], 'disgust':[], 'anger':[] \
                                                                  }
        # print ("UTTERANCES_LENGTH=",len(utterances))
        if len(utterances) > 0:
            try:
                responseToneChat = tone_analyzer.tone_chat(utterances)
                # print(responseToneChat["utterances_tone"])
                for i in responseToneChat["utterances_tone"]:
                    for j in i["tones"]:
                        # print(i['utterance_text'], "-->", j['tone_id'], "=", j['score'])
                        if (j['tone_id'] in ['sad', 'frustrated', 'impolite']):
                            if j['score'] != 0: 
                                if checkIfPartySentence(i['utterance_text']):
                                    map_congressional_record_party_sentiment_and_emotion[j['tone_id']].append(j['score'])
                                    if j['score'] > ACCEPTABLE_EMOTION_THRESHOLD:
                                        all_sample_records_emotion_party_sentences.append(i['utterance_text'].lower())
                                else:
                                    map_congressional_record_no_party_sentiment_and_emotion[j['tone_id']].append(j['score'])
                                    if j['score'] > ACCEPTABLE_EMOTION_THRESHOLD:
                                        all_sample_records_emotion_no_party_sentences.append(i['utterance_text'].lower())
            except:
                print("JUNK UTTERANCES ?? = ", utterances)
            
        # extract additional emotional characteristics from Watson NLU API
        # for higher accuracy process on each sentence
        
        for sent in doc.sents:
            try:
                # ignore any sentence less than threshold number of words to reduce false positives
                ts = textacy.TextStats(nlp(sent.text))
                # print(ts.n_words)
                # check if party sentence, likely indicator of polarization sentence
                if ts.n_words >= ACCEPTABLE_MIN_NUM_WORDS_PER_SENTENCE:

                    responseNLU = natural_language_understanding.analyze(text=sent.text,                                                                          features=Features(#emotion=EmotionOptions(), \
                                                                                           sentiment=SentimentOptions()))

                    if responseNLU["sentiment"]["document"]["score"] < 0: 
                        if checkIfPartySentence(sent.text):
                            map_congressional_record_party_sentiment_and_emotion['negative_sentiment'].append(responseNLU["sentiment"]["document"]["score"])
                            if responseNLU["sentiment"]["document"]["score"] < ACCEPTABLE_SENTIMENT_THRESHOLD: 
                                all_sample_records_emotion_party_sentences.append(sent.text.lower())
                        else:
                            map_congressional_record_no_party_sentiment_and_emotion['negative_sentiment'].append(responseNLU["sentiment"]["document"]["score"])
                            if responseNLU["sentiment"]["document"]["score"] < ACCEPTABLE_SENTIMENT_THRESHOLD: 
                                all_sample_records_emotion_no_party_sentences.append(sent.text.lower())
                            
                    '''
                    if (responseNLU["emotion"]["document"]["emotion"]["sadness"] != 0):
                        if checkIfPartySentence(sent.text): 
                            map_congressional_record_party_sentiment_and_emotion["sadness"].append(responseNLU["emotion"]["document"]["emotion"]["sadness"])
                            if (responseNLU["emotion"]["document"]["emotion"]["sadness"] > ACCEPTABLE_EMOTION_THRESHOLD):
                                all_sample_records_emotion_party_sentences.append(sent.text.lower())
                        else:
                            map_congressional_record_no_party_sentiment_and_emotion["sadness"].append(responseNLU["emotion"]["document"]["emotion"]["sadness"])
                            if (responseNLU["emotion"]["document"]["emotion"]["sadness"] > ACCEPTABLE_EMOTION_THRESHOLD):
                                all_sample_records_emotion_no_party_sentences.append(sent.text.lower())
                    
                    if (responseNLU["emotion"]["document"]["emotion"]["fear"] != 0):
                        if checkIfPartySentence(sent.text): 
                            map_congressional_record_party_sentiment_and_emotion["fear"].append(responseNLU["emotion"]["document"]["emotion"]["fear"])
                            if (responseNLU["emotion"]["document"]["emotion"]["fear"] > ACCEPTABLE_EMOTION_THRESHOLD):
                                all_sample_records_emotion_party_sentences.append(sent.text.lower())
                        else:
                            map_congressional_record_no_party_sentiment_and_emotion["fear"].append(responseNLU["emotion"]["document"]["emotion"]["fear"])
                            if (responseNLU["emotion"]["document"]["emotion"]["fear"] > ACCEPTABLE_EMOTION_THRESHOLD):
                                all_sample_records_emotion_no_party_sentences.append(sent.text.lower())
                    
                    if (responseNLU["emotion"]["document"]["emotion"]["disgust"] != 0):
                        if checkIfPartySentence(sent.text): 
                            map_congressional_record_party_sentiment_and_emotion["disgust"].append(responseNLU["emotion"]["document"]["emotion"]["disgust"])
                            if (responseNLU["emotion"]["document"]["emotion"]["disgust"] > ACCEPTABLE_EMOTION_THRESHOLD):
                                all_sample_records_emotion_party_sentences.append(sent.text.lower())
                        else:
                            map_congressional_record_no_party_sentiment_and_emotion["disgust"].append(responseNLU["emotion"]["document"]["emotion"]["disgust"])
                            if (responseNLU["emotion"]["document"]["emotion"]["disgust"] > ACCEPTABLE_EMOTION_THRESHOLD):
                                all_sample_records_emotion_no_party_sentences.append(sent.text.lower())
                            
                    if (responseNLU["emotion"]["document"]["emotion"]["anger"] != 0):
                        if checkIfPartySentence(sent.text): 
                            map_congressional_record_party_sentiment_and_emotion["anger"].append(responseNLU["emotion"]["document"]["emotion"]["anger"])
                            if (responseNLU["emotion"]["document"]["emotion"]["anger"] > ACCEPTABLE_EMOTION_THRESHOLD):
                                all_sample_records_emotion_party_sentences.append(sent.text.lower())
                        else:
                            map_congressional_record_no_party_sentiment_and_emotion["anger"].append(responseNLU["emotion"]["document"]["emotion"]["anger"])
                            if (responseNLU["emotion"]["document"]["emotion"]["anger"] > ACCEPTABLE_EMOTION_THRESHOLD):
                                all_sample_records_emotion_no_party_sentences.append(sent.text.lower())
                    '''
                # print(json.dumps(responseNLU, indent=2))
            except:
                # not every sentence yields results, its ok to ignore it
                print("DID NOT EXPECT TO GET HERE")
                next

        print("map_congressional_record_party_sentiment_and_emotion = ", map_congressional_record_party_sentiment_and_emotion)
        print("num_party_sentences = ", num_party_sentences)
        print("map_congressional_record_no_party_sentiment_and_emotion = ", map_congressional_record_no_party_sentiment_and_emotion)
        print("num_no_party_sentences = ", num_no_party_sentences)

        # calculate average general emotion tone and sentiment score across entire congressional record
        average_congressional_record_party_sad = 0
        average_congressional_record_party_frustrated = 0
        average_congressional_record_party_impolite = 0
        average_congressional_record_party_negative_sentiment = 0

        # average_congressional_record_party_sadness = 0
        # average_congressional_record_party_fear = 0
        # average_congressional_record_party_disgust = 0
        # average_congressional_record_party_anger = 0
        
        average_congressional_record_no_party_sad = 0
        average_congressional_record_no_party_frustrated = 0
        average_congressional_record_no_party_impolite = 0
        average_congressional_record_no_party_negative_sentiment = 0

        # average_congressional_record_no_party_sadness = 0
        # average_congressional_record_no_party_fear = 0
        # average_congressional_record_no_party_disgust = 0
        # average_congressional_record_no_party_anger = 0

        if num_party_sentences > 0:
            #party_array_len_max stores the length of the array with the maximum amount of scores for party mentioned sentences
            party_array_len_max = np.amax([len(map_congressional_record_party_sentiment_and_emotion['sad']),                                          len(map_congressional_record_party_sentiment_and_emotion['frustrated']),                                          len(map_congressional_record_party_sentiment_and_emotion['impolite']),                                          len(map_congressional_record_party_sentiment_and_emotion['negative_sentiment'])]                                          # len(map_congressional_record_party_sentiment_and_emotion['sadness']), \
                                         # len(map_congressional_record_party_sentiment_and_emotion['fear']), \
                                         # len(map_congressional_record_party_sentiment_and_emotion['disgust']), \
                                         # len(map_congressional_record_party_sentiment_and_emotion['anger'])] \
    
            '''
            the purpose of assigning party_array_len_max to num_party_sentences is to ensure that if
            there is enough statistically significant evidence of a particular emotion within the
            congressional record, there is no need to value the emotion by dividing by the entire
            length of the paragraph 
            '''
            if (party_array_len_max / num_party_sentences) >= 0.5 or party_array_len_max >= ACCEPTABLE_EMOTION_SENTENCES_COUNT_THRESHOLD:
                num_party_sentences = party_array_len_max 
            print("real num_party_sentences = ", num_party_sentences)
            average_congressional_record_party_sad = np.sum(map_congressional_record_party_sentiment_and_emotion['sad'])/num_party_sentences
            average_congressional_record_party_frustrated = np.sum(map_congressional_record_party_sentiment_and_emotion['frustrated'])/num_party_sentences
            average_congressional_record_party_impolite = np.sum(map_congressional_record_party_sentiment_and_emotion['impolite'])/num_party_sentences
            average_congressional_record_party_negative_sentiment = np.sum(map_congressional_record_party_sentiment_and_emotion['negative_sentiment'])/num_party_sentences
            # average_congressional_record_party_sadness = np.sum(map_congressional_record_party_sentiment_and_emotion['sadness'])/num_party_sentences
            # average_congressional_record_party_fear = np.sum(map_congressional_record_party_sentiment_and_emotion['fear'])/num_party_sentences
            # average_congressional_record_party_disgust = np.sum(map_congressional_record_party_sentiment_and_emotion['disgust'])/num_party_sentences
            # average_congressional_record_party_anger = np.sum(map_congressional_record_party_sentiment_and_emotion['anger'])/num_party_sentences
        
        if num_no_party_sentences > 0:
            no_party_array_len_max = np.amax([len(map_congressional_record_no_party_sentiment_and_emotion['sad']), \
                                         len(map_congressional_record_no_party_sentiment_and_emotion['frustrated']), \
                                         len(map_congressional_record_no_party_sentiment_and_emotion['impolite']), \
                                         len(map_congressional_record_no_party_sentiment_and_emotion['negative_sentiment'])] \
                                         # len(map_congressional_record_no_party_sentiment_and_emotion['sadness']), \
                                         # len(map_congressional_record_no_party_sentiment_and_emotion['fear']), \
                                         # len(map_congressional_record_no_party_sentiment_and_emotion['disgust']), \
                                         # len(map_congressional_record_no_party_sentiment_and_emotion['anger']) \
                                          )
            if (no_party_array_len_max / num_no_party_sentences) >= 0.5 or no_party_array_len_max >= ACCEPTABLE_EMOTION_SENTENCES_COUNT_THRESHOLD:
                num_no_party_sentences = no_party_array_len_max
            print("real num_no_party_sentences = ", num_no_party_sentences)
            average_congressional_record_no_party_sad = np.sum(map_congressional_record_no_party_sentiment_and_emotion['sad'])/num_no_party_sentences
            average_congressional_record_no_party_frustrated = np.sum(map_congressional_record_no_party_sentiment_and_emotion['frustrated'])/num_no_party_sentences
            average_congressional_record_no_party_impolite = np.sum(map_congressional_record_no_party_sentiment_and_emotion['impolite'])/num_no_party_sentences
            average_congressional_record_no_party_negative_sentiment = np.sum(map_congressional_record_no_party_sentiment_and_emotion['negative_sentiment'])/num_no_party_sentences
            # average_congressional_record_no_party_sadness = np.sum(map_congressional_record_no_party_sentiment_and_emotion['sadness'])/num_no_party_sentences
            # average_congressional_record_no_party_fear = np.sum(map_congressional_record_no_party_sentiment_and_emotion['fear'])/num_no_party_sentences
            # average_congressional_record_no_party_disgust = np.sum(map_congressional_record_no_party_sentiment_and_emotion['disgust'])/num_no_party_sentences
            # average_congressional_record_no_party_anger = np.sum(map_congressional_record_no_party_sentiment_and_emotion['anger'])/num_no_party_sentences
            
        # keep only unique sentences
        all_sample_records_emotion_party_sentences = list(set(all_sample_records_emotion_party_sentences))
        all_sample_records_emotion_no_party_sentences = list(set(all_sample_records_emotion_no_party_sentences))
        
        all_sample_records.append({"start_date": query_date_range["start"], \
                                   "end_date": query_date_range["end"], \
                                   "average_congressional_record_party_sad": average_congressional_record_party_sad, \
                                   "average_congressional_record_party_frustrated": average_congressional_record_party_frustrated, \
                                   "average_congressional_record_party_impolite": average_congressional_record_party_impolite, \
                                   "average_congressional_record_party_negative_sentiment": average_congressional_record_party_negative_sentiment, \
                                   # "average_congressional_record_party_sadness": average_congressional_record_party_sadness, \
                                   # "average_congressional_record_party_fear": average_congressional_record_party_fear, \
                                   # "average_congressional_record_party_disgust": average_congressional_record_party_disgust, \
                                   # "average_congressional_record_party_anger": average_congressional_record_party_anger, \
                                   "average_congressional_record_no_party_sad": average_congressional_record_no_party_sad, \
                                   "average_congressional_record_no_party_frustrated": average_congressional_record_no_party_frustrated, \
                                   "average_congressional_record_no_party_impolite": average_congressional_record_no_party_impolite, \
                                   "average_congressional_record_no_party_negative_sentiment": average_congressional_record_no_party_negative_sentiment, \
                                   # "average_congressional_record_no_party_sadness": average_congressional_record_no_party_sadness, \
                                   # "average_congressional_record_no_party_fear": average_congressional_record_no_party_fear, \
                                   # "average_congressional_record_no_party_disgust": average_congressional_record_no_party_disgust, \
                                   # "average_congressional_record_no_party_anger": average_congressional_record_no_party_anger, \
                                   "speech": responseFilteredQuery[random_index].speech
                                  })    
    return {"all_sample_records": all_sample_records, \
            "all_sample_records_emotion_party_sentences": all_sample_records_emotion_party_sentences, \
            "all_sample_records_emotion_no_party_sentences": all_sample_records_emotion_no_party_sentences}    


# In[193]:


df_results = pd.DataFrame()
overall_records_emotion_party_sentences = []
overall_records_emotion_no_party_sentences = []


# In[194]:


for x in range(1, NUM_TRIALS+1):
    for result in map(lambda a_decade: extractEmotionalSentencesAndTabulate(a_decade,                                                    es_congressional_records,                                                    nlp,                                                    natural_language_understanding,                                                    tone_analyzer,                                                    reduced_any_party_query, x),                                                    decades):
        df_results = df_results.append(result["all_sample_records"], ignore_index=True)
        overall_records_emotion_party_sentences.extend(result["all_sample_records_emotion_party_sentences"])
        overall_records_emotion_no_party_sentences.extend(result["all_sample_records_emotion_no_party_sentences"])

overall_records_emotion_party_sentences = list(set(overall_records_emotion_party_sentences))
overall_records_emotion_no_party_sentences = list(set(overall_records_emotion_no_party_sentences))


# In[192]:


df_results


# In[45]:


df_results.to_csv("congressional_records_emotion_computed_results.csv")


# In[46]:


with open('congressional_records_computed_results_emotion_party_evidence.txt', 'w') as filehandle:  
    for listitem in overall_records_emotion_party_sentences:
        filehandle.write('%s\n' % listitem)
with open('congressional_records_computed_results_emotion_no_party_evidence.txt', 'w') as filehandle:  
    for listitem in overall_records_emotion_no_party_sentences:
        filehandle.write('%s\n' % listitem)


# In[41]:


any_party


# In[76]:


import numpy as np
np.sum([])


# In[113]:


np.amax([2,4])

