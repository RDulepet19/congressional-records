# congressional-records
This repository contains code to parse congressional records and extract insight, identify polarization, emotions, and relevant context. The code requires Python 3.5 and relevant libraries to execute. In addition requires Watson API credentials. The code is compressed and should be extracted using Zip or TAR/GZIP utility.

The file final_congressional_code.tgz contains following files:
1) detectByPartyNormalizedPolarizedCongressionalRecordsScores.py - Python script to detect, extract polarization/emotion/sentiment score trends from congressional speech records and contextual sentences by party affiliation. The code uses elastic search to retrieve the congressional records, and then picks sample records across decades to process and generate a statistical view of the trend. It also leverages Watson API to calculate the tone/sentiment score, and hence requires watson credentials to run.

2) uploadESSpeech.py - Python script to parse the congressional records downloaded from (https://data.stanford.edu/congress_text#download-data), and upload it to locally installed Elastic Search instance. This enables lucene-based/google style search of congressional records.

3) extractCongressPartyAffiliationSentences.py - Python script to extract contextual sentences from congressional speech records, that highlight party affiliation (speaker and spoken) and pivoted by CONGRESSIONAL SESSION.

final_congressional_results.tgz and congress_party_affiliation_sentences.tgz contains processed results from the script.

To run the code:

Upload speech records to Elastic Search
python uploadESSpeech.py

Extract party affiliation polarization (speaker vs spoken)
1) python detectByPartyNormalizedPolarizedCongressionalRecordsScores.py democrat democrat
2) python detectByPartyNormalizedPolarizedCongressionalRecordsScores.py democrat republican
3) python detectByPartyNormalizedPolarizedCongressionalRecordsScores.py republican republican
4) python detectByPartyNormalizedPolarizedCongressionalRecordsScores.py republican democrat

Extract contextual sentences from congressional records, by party affiliation, pivoted by congressional session
python extractCongressPartyAffiliationSentences.py
