

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
        
def cut_one_wav(df,wav,backchannel_type,direct,folder,direct_cut):
    samplerate, data = wavfile.read(direct+folder+wav)
    s = len(df)
    for i in range (s):
        onset = int(get_onset(df,i))
        offset = int(get_end(df,i))
        newAudio = data[onset:offset]
        write(direct_cut+wav[:-4]+'_'+str(i)+'_'+backchannel_type+".wav", samplerate, newAudio.astype(np.int16))
    
def cut_wav_file(data_csv,wav1,wav2,direct,folder,direct_cut):
    samplerate, data = wavfile.read(direct+folder+wav1)
    df1_f,df1_r = extract_df_times_csv_file(data_csv,samplerate,'A1')
    df2_f,df2_r = extract_df_times_csv_file(data_csv,samplerate,'A2')
    
    cut_one_wav(df1_f,wav1,"feedback",direct,folder,direct_cut)
    cut_one_wav(df2_f,wav2,"feedback",direct,folder,direct_cut)
    cut_one_wav(df1_r,wav1,"response",direct,folder,direct_cut)
    cut_one_wav(df2_r,wav2,"response",direct,folder,direct_cut)
    
       
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

def cut_all_wav(path_rec,path_annot,path_cut):
    l_folder = os.listdir(path_rec) 
    for folder in l_folder:
        folder = folder+'/'
        [wav1,wav2] = search_extention_in_given_folder (folder,path_rec,".wav")
        [csv_file] = search_extention_in_given_folder (folder,path_annot,".csv")
        data_csv = path_annot+folder+csv_file
        cut_wav_file(data_csv,wav1,wav2,path_rec,folder,path_cut)

def create_csv_from_cut_wav(path_cut):
    #print("TO DO: verify if max supposed to be is max frames or max duration")
    l_wav = os.listdir(path_cut)
    l_csv = []
    for elem in l_wav:
        
        if "response" in elem:
            label = 0
        else :
            label = 1
            
        with contextlib.closing(wave.open(path_cut+elem,'r')) as f:
            frames = f.getnframes()
            
        l_csv.append([elem,label,frames])
        
    df = pd.DataFrame(l_csv) 
    df.columns =['filename', 'label', 'nbframes']
    df.to_csv('filenames_labels_nbframes.csv') 
    return df
        

path_rec = "../data/Adult-rec/"
path_annot = "../data/Annotations-adults/"
folder = 'AD/'
path_cut = "../data/cut_wav/"
wav1 = "AA-AN-DL-AN.wav"
wav2 = "AA-AN-DL-DL.wav"
data_csv = "../data/Annotations-adults/AD/AA-AN-DL-annotation.csv"


mini_wav = "AA-AN-DL-AN_0_feedback.wav"
#cut_all_wav(path_rec,path_annot,path_cut)
#create_csv_from_cut_wav(path_cut)

#data = pd.read_csv(path_cut+"filenames_labels_nbframes.csv")
#print(data['nbframes'].max())

sound, sample_rate = torchaudio.load(path_cut+mini_wav)
#sound, sample_rate = torchaudio.load(path, out=None, normalization=True)
#soundData = torch.mean(sound, dim=0, keepdim=True)
tempData = torch.zeros([1, 61000])  # tempData accounts for audio clips that are too short


if soundData.numel() < 61000:
    tempData[:, :soundData.numel()] = soundData
else:
    tempData = soundData[:, :61000]

soundData = tempData

mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(soundData)  # (channel, n_mels, time)
mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate)(soundData)  # (channel, n_mfcc, time)
mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
# spectogram = torchaudio.transforms.Spectrogram(sample_rate=sample_rate)(soundData)
feature = torch.cat([mel_specgram, mfcc], axis=1)
#aaa = feature[0].permute(1, 0), self.labels[index]




