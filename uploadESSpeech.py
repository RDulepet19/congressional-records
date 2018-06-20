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

list_descr = []
list_speech = []

descr_filenames = glob.glob("./hein-bound" + "/descr*.txt")
speech_filenames = glob.glob("./hein-bound" + "/speech*.txt")

for filename in descr_filenames:
	try:
		list_descr.append(pd.read_csv(filename, sep="|", error_bad_lines=False, header = 0, encoding='ISO-8859-1'))
	except:
		print("Error reading description file = ", filename)

for filename in speech_filenames:
	try:
		list_speech.append(pd.read_csv(filename, sep="|", error_bad_lines=False, header = 0, encoding='ISO-8859-1'))
	except:
		print("Error reading speech file = ", filename)


df_descr = pd.concat(list_descr)
df_speech = pd.concat(list_speech)
list_descr = None
list_speech = None

df_descr_speech = pd.merge(df_descr, df_speech, on='speech_id')
df_descr = None
df_speech = None

# convert date
df_descr_speech['date'] = pd.to_datetime(df_descr_speech['date'], format='%Y%m%d').dt.date
df_descr_speech['chamber'] = df_descr_speech['chamber'].fillna('')
df_descr_speech['speaker'] = df_descr_speech['speaker'].fillna('')
df_descr_speech['first_name'] = df_descr_speech['first_name'].fillna('')
df_descr_speech['last_name'] = df_descr_speech['last_name'].fillna('')
df_descr_speech['state'] = df_descr_speech['state'].fillna('')
df_descr_speech['gender'] = df_descr_speech['gender'].fillna('')
df_descr_speech['speech'] = df_descr_speech['speech'].fillna('')


# load into elastic search
from datetime import datetime
import json
from elasticsearch import Elasticsearch, RequestsHttpConnection, serializer, compat, exceptions, helpers
import logging

logging.basicConfig(filename='parse.log',level=logging.INFO)

# rollback recent changes to serializer that choke on unicode
class JSONSerializerPython2(serializer.JSONSerializer):
        def dumps(self, data):
                # don't serialize strings
                if isinstance(data, compat.string_types):
                        return data
                try:
                        return json.dumps(data, default=self.default, ensure_ascii=True)
                except (ValueError, TypeError) as e:
                        raise exceptions.SerializationError(data, e)

es = Elasticsearch(serializer=JSONSerializerPython2())  # use default of localhost, port 9200

elastic_articles = []

for index, row in df_descr_speech.iterrows():
	elastic_articles.append({'_index': 'congressional_records', '_type': 'article', "_op_type": 'index', '_source':{"speech_id": row["speech_id"], "chamber": row["chamber"], "speech_date": str(row["date"]), "speaker": row["speaker"], "first_name": row["first_name"], "last_name": row["last_name"], "state": row["state"], "gender": row["gender"], "word_count": row["word_count"], "speech": row["speech"]}})

print ("LENGTH=",len(elastic_articles))
res = helpers.bulk(es, elastic_articles, raise_on_exception=False)

# logging.info(datetime.now().isoformat() + " imported " + str(res[0]) + " records from " + sys.argv[1])

