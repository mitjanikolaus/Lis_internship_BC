

import pandas as pd
import datetime
import time
from csv import reader
import matplotlib.pyplot as plt
from pydub import AudioSegment
import wave
from scipy.io import wavfile
from scipy.io.wavfile import write
import numpy as np
import os
import contextlib
import torchaudio
import csv


def convert_to_wav(path_m4a):
    l_direct = os.listdir(path_m4a)
    for direct in l_direct:
        #os.mkdir(path_wav+direct)
        l_files_m4a = os.listdir(path_m4a+direct)
        for file in l_files_m4a:
            if 'm4a' in file:
                if len(file) == 15:
                    track = AudioSegment.from_file(path_m4a+'/'+direct+'/'+file,  format='m4a')
                    file_handle = track.export(path_m4a+direct+'/'+file[:-3]+'wav', format='wav')

def create_empty_csv(csv_path,csv_name):
    column_names = ["filename","label","nbframes","folder"] 

    with open(csv_path+csv_name, 'w') as f:
    # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(column_names)

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
        
def cut_one_wav(df,wav,backchannel_type,direct,folder,direct_cut,nb_frames_before_onset):
    samplerate, data = wavfile.read(direct+folder+wav)
    s = len(df)
    for i in range (s):
        onset = int(get_onset(df,i))-nb_frames_before_onset
        offset = int(get_end(df,i))
        if onset+100<offset:
            newAudio = data[onset:offset]
            write(direct_cut+'/'+folder+'/'+wav[:-4]+'_'+str(i)+'_'+backchannel_type+".wav", samplerate, newAudio.astype(np.int16))
    
def cut_wav_file(data_csv,wav1,wav2,direct,folder,direct_cut,nb_frames_before_onset):
    samplerate, data = wavfile.read(direct+folder+wav1)
    df1_f,df1_r = extract_df_times_csv_file(data_csv,samplerate,'A1')
    df2_f,df2_r = extract_df_times_csv_file(data_csv,samplerate,'A2')
    
    cut_one_wav(df1_f,wav1,"feedback",direct,folder,direct_cut,nb_frames_before_onset)
    cut_one_wav(df2_f,wav2,"feedback",direct,folder,direct_cut,nb_frames_before_onset)
    cut_one_wav(df1_r,wav1,"response",direct,folder,direct_cut,nb_frames_before_onset)
    cut_one_wav(df2_r,wav2,"response",direct,folder,direct_cut,nb_frames_before_onset)
    
       
def get_onset(df,i):
    return df.at[i, 'onset']
    
def get_end(df,i):
    return df.at[i, 'end']


def search_extention_in_given_folder (folder,path,extention):
    l_files = os.listdir(path+folder)
    l = []
    for file in l_files:
        if file.endswith(extention):
            l.append(file)
    return l

def cut_all_wav(path_rec,path_annot,path_cut,nb_frames_before_onset):
    l_folder = os.listdir(path_rec) 
    for folder in l_folder:
        os.mkdir(path_cut+folder)
        folder = folder+'/'
        [wav1,wav2] = search_extention_in_given_folder (folder,path_rec,".wav")
        adult_1 = wav1[3:5]
        wav1, wav2 = [wav1, wav2] if wav1[-6:-4] == adult_1 else [wav2, wav1]
        [csv_file] = search_extention_in_given_folder (folder,path_annot,".csv")
        data_csv = path_annot+folder+csv_file
        cut_wav_file(data_csv,wav1,wav2,path_rec,folder,path_cut,nb_frames_before_onset)

def create_csv_from_cut_wav(path_cut,nb_frames_before_onset):
    #print("TO DO: verify if max supposed to be is max frames or max duration")
    l_folders = os.listdir(path_cut)
    l_csv = []
    for folder in l_folders:
        l_wav = os.listdir(path_cut+folder)
        
        for elem in l_wav:
            if "response" in elem:
                label = 0
            else :
                label = 1
                
            with contextlib.closing(wave.open(path_cut+'/'+folder+'/'+elem,'r')) as f:
                frames = f.getnframes()+nb_frames_before_onset
            if f.getnframes() > 1000:
                l_csv.append([elem,label,frames,folder])
        
    df = pd.DataFrame(l_csv) 
    df.columns =['filename', 'label', 'nbframes','folder']
    df.to_csv(path_cut+'filenames_labels_nbframes.csv') 
    return df
        




path_rec = "../data/Adult-rec/"
path_annot = "../data/Annotations-adults/"
path_cut = "../data/cut_wav_with_data_before_onset/"
nb_frames_before_onset = 100000


convert_to_wav("../data/Adult-rec/")
cut_all_wav(path_rec,path_annot,path_cut,nb_frames_before_onset)
df = create_csv_from_cut_wav(path_cut,nb_frames_before_onset)

