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
list_speakermap = []

descr_filenames = glob.glob("." + "/descr*.txt")
speech_filenames = glob.glob("." + "/speech*.txt")
speakermap_filenames = glob.glob("." + "/*SpeakerMap.txt")

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


for filename in speakermap_filenames:
	try:
		list_speakermap.append(pd.read_csv(filename, sep="|", error_bad_lines=False, header = 0, encoding='ISO-8859-1'))
	except:
		print("Error reading speakermap file = ", filename)

df_descr = pd.concat(list_descr)
df_speech = pd.concat(list_speech)
df_speakermap = pd.concat(list_speakermap)
list_descr = None
list_speech = None
list_speakermap = None

df_descr_speech_speakermap = pd.merge(df_descr, df_speech, on='speech_id')
df_descr_speech_speakermap = pd.merge(df_descr_speech_speakermap, df_speakermap, on=['speech_id'])
df_descr = None
df_speech = None
df_speakermap = None

# convert date
df_descr_speech_speakermap['date'] = pd.to_datetime(df_descr_speech_speakermap['date'], format='%Y%m%d').dt.date
# there are duplicate columns across descr file and speakermap table (hence _x and _y fields for the same one)
df_descr_speech_speakermap['chamber_x'] = df_descr_speech_speakermap['chamber_x'].fillna('')
df_descr_speech_speakermap['chamber_y'] = df_descr_speech_speakermap['chamber_y'].fillna('')
df_descr_speech_speakermap['speaker'] = df_descr_speech_speakermap['speaker'].fillna('')
df_descr_speech_speakermap['first_name'] = df_descr_speech_speakermap['first_name'].fillna('')
df_descr_speech_speakermap['last_name'] = df_descr_speech_speakermap['last_name'].fillna('')
df_descr_speech_speakermap['firstname'] = df_descr_speech_speakermap['firstname'].fillna('')
df_descr_speech_speakermap['lastname'] = df_descr_speech_speakermap['lastname'].fillna('')
df_descr_speech_speakermap['state_x'] = df_descr_speech_speakermap['state_x'].fillna('')
df_descr_speech_speakermap['state_y'] = df_descr_speech_speakermap['state_y'].fillna('')
df_descr_speech_speakermap['gender_x'] = df_descr_speech_speakermap['gender_x'].fillna('')
df_descr_speech_speakermap['gender_y'] = df_descr_speech_speakermap['gender_y'].fillna('')
df_descr_speech_speakermap['speech'] = df_descr_speech_speakermap['speech'].fillna('')
df_descr_speech_speakermap['party'] = df_descr_speech_speakermap['party'].fillna('')
df_descr_speech_speakermap['district'] = df_descr_speech_speakermap['district'].fillna('')
df_descr_speech_speakermap['nonvoting'] = df_descr_speech_speakermap['nonvoting'].fillna('')


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

for index, row in df_descr_speech_speakermap.iterrows():
	elastic_articles.append({'_index': 'congressional_records', '_type': 'article', "_op_type": 'index', '_source':{"speech_id": row["speech_id"], "chamber_x": row["chamber_x"], "chamber_y": row["chamber_y"], "speech_date": str(row["date"]), "speaker": row["speaker"], "first_name": row["first_name"], "last_name": row["last_name"], "firstname": row["firstname"], "lastname": row["lastname"], "state_x": row["state_x"], "state_y": row["state_y"], "gender_x": row["gender_x"], "gender_y": row["gender_y"], "word_count": row["word_count"], "speakerid": row["speakerid"], "party": row["party"], "district": row["district"], "nonvoting": row["nonvoting"], "speech": row["speech"]}})

print ("LENGTH=",len(elastic_articles))
res = helpers.bulk(es, elastic_articles, raise_on_exception=False)

# logging.info(datetime.now().isoformat() + " imported " + str(res[0]) + " records from " + sys.argv[1])

