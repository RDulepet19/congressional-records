#!/home/ubuntu/anaconda3/bin//python
'''
MIT License

Copyright (c) 2018 Riya Dulepet <riyadulepet123@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The code is inspired by https://github.com/erikor/medline project, but the logic to
parse medline XML was substantially modified.
'''
# pre-requisites: pip install elasticsearch
# pip install --upgrade pip

# to execute this code:
# STEP 0: ensure elastic search and kibana are running on port 9200
# and 5601 correspondingly
# STEP 1: make sure you have all the medline XML files downloaded from 
# STEP 2: then you run nohup ls *.xml | xargs -n 1 -P 4 python ./parseMedline.py &
# the above step assume quad-core processor, and runs it as daemon process so when
# you exit SSH session, it runs in background.
# this should load the data into elastic search

import pandas as pd
import glob
import sys
import sys, os


descr_filenames = glob.glob("." + "/descr*.txt")
speech_filenames = glob.glob("." + "/speech*.txt")
speakermap_filenames = glob.glob("." + "/*SpeakerMap.txt")

NO_PARTY_SENTENCE = "N"
REPUBLICAN_SENTENCE = "R"
DEMOCRAT_SENTENCE = "D"
BOTH_PARTY_SENTENCE = "B"

republican = ["rnc", "gop", "republican", "republicans", "conservative", "conservatives", "right wing", "alt right", "far right"]
democrat = ["dnc", "democrat", "democrats", "democratic", "liberal", "liberals", "progressive", "progressives", "moderates",    "nonconservative", "nonconservatives", "alt left", "far left", "left wing"]

from datetime import datetime
import json
import logging
from collections import deque
from pathlib import Path
import os.path

logging.basicConfig(filename='parse.log',level=logging.INFO)

DESTINATION_FILE = "congress_party_affiliation_sentences.csv"

import spacy
import textacy
nlp = spacy.load('en_core_web_sm')

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

def partyTypeSentence(sent):
    global NO_PARTY_SENTENCE, REPUBLICAN_SENTENCE, DEMOCRAT_SENTENCE, BOTH_PARTY_SENTENCE
    global republican, democrat
    from sklearn.feature_extraction.text import CountVectorizer 
    
    # extract unigrams and bigrams
    vectorizer = CountVectorizer(ngram_range=(1,2))
    analyzer = vectorizer.build_analyzer()
    sent_analyzer = analyzer(sent)
    
    if any(word in sent_analyzer for word in republican) and any(word in sent_analyzer for word in democrat):
        return BOTH_PARTY_SENTENCE
    elif any(word in sent_analyzer for word in republican):
        return REPUBLICAN_SENTENCE
    elif any(word in sent_analyzer for word in democrat):
        return DEMOCRAT_SENTENCE
    return NO_PARTY_SENTENCE

for speakermap_filename in speakermap_filenames:
    try:
        prefix = speakermap_filename[2:5]
        print("prefix=", prefix)
        descr_filename = "./descr_" + str(prefix) + ".txt"
        speech_filename = "./speeches_" + str(prefix) + ".txt"
        list_descr = []
        list_speech = []
        list_speakermap = []
        
        list_descr.append(pd.read_csv(descr_filename, sep="|", error_bad_lines=False, header = 0, encoding='ISO-8859-1'))
        list_speech.append(pd.read_csv(speech_filename, sep="|", error_bad_lines=False, header = 0, encoding='ISO-8859-1'))
        list_speakermap.append(pd.read_csv(speakermap_filename, sep="|", error_bad_lines=False, header = 0, encoding='ISO-8859-1'))
        df_descr = pd.concat(list_descr)
        df_speech = pd.concat(list_speech)
        df_speakermap = pd.concat(list_speakermap)
        print("len df_descr=", len(df_descr))
        print("len df_speech=", len(df_speech))
        print("len df_speakerma=", len(df_speakermap))
        list_descr = None
        list_speech = None
        list_speakermap = None
        df_descr_speech_speakermap = pd.merge(pd.merge(df_descr, df_speech, on='speech_id'), df_speakermap, on='speech_id')
        df_descr = None
        df_speech = None
        df_speakermap = None
        # convert date
        df_descr_speech_speakermap['speech'] = df_descr_speech_speakermap['speech'].fillna('')
        df_descr_speech_speakermap['party'] = df_descr_speech_speakermap['party'].fillna('')
        
        df_congressPartySentences = pd.DataFrame(columns=('congress', 'speech_id', 'speaker_party', 'spoken_party', 'sentence'))
        for index, row in df_descr_speech_speakermap.iterrows():
            # process NLP on the text, primarily to extract sentences most reliabily
            # doc = nlp(row["speech"])
            doc = sent_tokenize(row["speech"])
            # for sent in doc.sents:
            for sent in doc:
                party_affiliation = partyTypeSentence(str(sent))
                if party_affiliation in [REPUBLICAN_SENTENCE, DEMOCRAT_SENTENCE]:
                    last_index = len(df_congressPartySentences)
                    df_congressPartySentences.loc[last_index] = "ignore"
                    df_congressPartySentences.loc[last_index]["congress"] = prefix
                    df_congressPartySentences.loc[last_index]["speech_id"] = row["speech_id"]
                    df_congressPartySentences.loc[last_index]["speaker_party"] = row["party"]
                    df_congressPartySentences.loc[last_index]["spoken_party"] = party_affiliation
                    df_congressPartySentences.loc[last_index]["sentence"] = sent
        print ("CONGRESS={},LENGTH={}", prefix, len(df_congressPartySentences))
        if os.path.exists(DESTINATION_FILE):
            # file exists
            df_congressPartySentences.to_csv(DESTINATION_FILE, mode='a', header=False)
        else:
            # brand new file
            df_congressPartySentences.to_csv(DESTINATION_FILE, mode='w', header=True)
    except Exception as e:
        print("Error reading description file = ", descr_filename)
        print("Error reading speech file = ", speech_filename)
        print("Error reading speakermap file = ", speakermap_filename)
        print(e) # for the repr
        print(str(e)) # for just the message
        print(e.args) # the arguments that the exception has been called with. 
                      # the first one is usually the message.        
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
# logging.info(datetime.now().isoformat() + " imported " + str(res[0]) + " records from " + sys.argv[1])