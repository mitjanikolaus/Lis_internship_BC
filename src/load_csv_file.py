data_csv = "../Annotations-adults/AD/AA-AN-DL-annotation.csv"

import pandas as pd
import datetime
import time
from csv import reader

l_feedback = []
l_resp = []

def return_sec(time_string):    
    x = time.strptime(time_string.split('.')[0],'%H:%M:%S')
    s = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    ms = float('0.'+time_string[-3:])
    return s+ms

# open file in read mode
def extract_df_times_csv_file(data_csv,samplerate,A):
    l_feedback = []
    l_resp = []
    with open(data_csv, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            #if x in row:
            if 'SpeechFunction' in row:
                if row[8][:2] == A:
                    if row[8][-8:] == "Feedback":
                        l_feedback.append([return_sec(row[2])*samplerate,return_sec(row[4])*samplerate,return_sec(row[6])*samplerate,row[8]])
                    if row[8][-8:] == "Response":
                        l_resp.append([return_sec(row[2])*samplerate,return_sec(row[4])*samplerate,return_sec(row[6])*samplerate,row[8]])
    df_r = pd.DataFrame(l_resp) 
    df_r.to_csv('resp.csv') 
    df_f = pd.DataFrame(l_feedback) 
    df_f.to_csv('feedback.csv') 
    if l_feedback != []:     
        df_f.columns =['onset', 'end', 'duration', 'label']
    if l_resp != []:
        df_r.columns =['onset', 'end', 'duration', 'label']
    return df_f,df_r
        
        


