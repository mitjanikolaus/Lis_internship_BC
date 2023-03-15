import matplotlib.pyplot as plt
import numpy as np 
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from csv import reader
import load_csv_file
from scipy.io.wavfile import write
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score


'''convert m4a into wav'''
m4a_file = 'AD/AA-AN-DL-AN.m4a'
#track = AudioSegment.from_file(m4a_file,  format= 'm4a')
#file_handle = track.export(wav_filename, format='wav')

'''
convert wav into array
We have a one audio chanel (no stereo)
data len is 31046656
'''
wav_filename_A1 = "Adult-rec/AD/AA-AN-DL-AN.wav"
wav_filename_A2 = "Adult-rec/AD/AA-AN-DL-DL.wav"

data_csv = "Annotations-adults/AD/AA-AN-DL-annotation.csv"
samplerate_A1, data_A1 = wavfile.read(wav_filename_A1) 
samplerate_A2, data_A2 = wavfile.read(wav_filename_A2) 


#w = wave.open(wav_filename, 'r')
#print(w.getnframes())
#print(w.getnchannels())
#plt.plot(data)
#plt.show

'''
first classifier
'''
def foo(i):
    if i>0:
        return 1
    else: 
        return 0

def first_classifier_mean(method,wav_filename,frame_nb,recouvrement, data_set_size, test_set_ratio=0.25):
    samplerate, data = wavfile.read(wav_filename) 
    
    l_data_set = []
    for i in range(data_set_size):
        l_data_set.append(data[i*frame_nb-recouvrement*i:(i+1)*frame_nb-recouvrement*i])

    X = np.array(l_data_set)
    
    y = np.array([foo(np.mean(X[i,:])) for i in range(data_set_size)])
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_set_ratio, random_state=42)
    
    if method == 'log_reg':
        classifier = LogisticRegression()
    if method == 'Neigh':
        classifier = KNeighborsClassifier()
    if method == 'Random_forest':
        classifier = RandomForestClassifier()
    

    classifier.fit(X_train, y_train)

    for i in range(5):
        y_predict = classifier.predict([X_test[i]])
        print(np.mean(X_test[i]))
        print(y_predict)
    return X,y
#def extract_labels(file):
    


        
def max_time_distribution(df):
    duration_list = df['duration'].tolist()
    return max(duration_list)

def hist_time_distribution(df):
    duration_list = df['duration'].tolist()
    plt.hist(duration_list, density=True)
    #plt.show()
    
def generate_wav_files_BC(wav_filename,df,A):
    samplerate, data = wavfile.read(wav_filename) 
    onset_list = df['onset'].tolist()
    m = max_time_distribution(df)
    i = 0
    for elem in onset_list:
        #print(elem,m)
        splitted_data = data[int(elem):int(elem+m)]
        write(A+"_BC"+str(i)+".wav", samplerate, splitted_data)
        i+=1
        
def cut_data(data,df,duration):
    onset_list = df['onset'].tolist()
    feature_list = df['label'].tolist()
    l_data = []
    for elem in onset_list:
        l_data.append(data[int(elem):int(elem+duration)])
    return l_data
        
def label_and_fearture_set(wav_file_A1,wav_file_A2,csv_file):
    
    samplerate_A1, data_A1 = wavfile.read(wav_file_A1) 
    samplerate_A2, data_A2 = wavfile.read(wav_file_A2) 
    
    df_f_A1,df_r_A1=load_csv_file.extract_df_times_csv_file(data_csv,samplerate_A1,'A1')
    df_f_A2,df_r_A2=load_csv_file.extract_df_times_csv_file(data_csv,samplerate_A2,'A2')
    
    #max duration of feedback and resp
    m_f_A1 = max_time_distribution(df_f_A1)
    m_r_A1 = max_time_distribution(df_r_A1)
    m_f_A2 = max_time_distribution(df_f_A2)
    m_r_A2 = max_time_distribution(df_r_A2)
    m = max(m_r_A1,m_r_A2,m_f_A1,m_f_A2)

    
    
    l_data_A1_f = cut_data(data_A1,df_f_A1,m)
    l_data_A2_f = cut_data(data_A2,df_f_A2,m)
    
    l_data_A1_r = cut_data(data_A1,df_r_A1,m)
    l_data_A2_r = cut_data(data_A2,df_r_A2,m)
    
    return l_data_A1_f,l_data_A2_f,l_data_A1_r,l_data_A2_r
    
def generate_X_Y_feedback(wav_file_A1,wav_file_A2,csv_file):
    l_data_A1_f,l_data_A2_f,l_data_A1_r,l_data_A2_r = label_and_fearture_set(wav_file_A1,wav_file_A2,csv_file)
    
    l_data_f = l_data_A1_f + l_data_A2_f
    l_label_f = [1]*len(l_data_f)
    
    return l_data_f,l_label_f

def generate_X_Y_resp(wav_file_A1,wav_file_A2,csv_file):
    l_data_A1_f,l_data_A2_f,l_data_A1_r,l_data_A2_r = label_and_fearture_set(wav_file_A1,wav_file_A2,csv_file)

    l_data_r = l_data_A1_r + l_data_A2_r
    l_label_r = [2]*len(l_data_r)
    
    return l_data_r,l_label_r

#def generate_X_Y_random(wav_file_A1,wav_file_A2,csv_file):
    
        
def first_classifier_BC(method,wav_file_A1,wav_file_A2,csv_file,test_set_ratio=0.25):

    l_data_f,l_label_f = generate_X_Y_feedback(wav_file_A1,wav_file_A2,csv_file)
    l_data_r,l_label_r = generate_X_Y_resp(wav_file_A1,wav_file_A2,csv_file)
    
    l_data_set = l_data_f + l_data_r
    l_label_set = l_label_f + l_label_r
    
    X = np.array(l_data_set)
    y = np.array(l_label_set)
   
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_set_ratio, random_state=42)
    
    if method == 'log_reg':
        classifier = LogisticRegression()
    if method == 'Neigh':
        classifier = KNeighborsClassifier()
    if method == 'Random_forest':
        classifier = RandomForestClassifier()
    

    classifier.fit(X_train, y_train)
    y_predict = []
    for i in range(len(y_test)):
        y_predict.append( classifier.predict([X_test[i]]))
    print(balanced_accuracy_score(y_test, y_predict))
    print(cohen_kappa_score(y_test, y_predict))
    return X,y


    
    
    

frame_nb = 100
recouvrement = 3
data_set_size = 4000
test_set_ratio = 0.25
method = 'Random_forest' # 'Neigh', "log_reg" 

#df_f_A1,df_r_A1=load_csv_file.extract_df_times_csv_file(data_csv,samplerate_A1,'A1')
#df_f_A2,df_r_A2=load_csv_file.extract_df_times_csv_file(data_csv,samplerate_A2,'A2')



#hist_time_distribution(df_f)
#hist_time_distribution(df_r)

#X=first_classifier_mean('log_reg',wav_filename,frame_nb,recouvrement, data_set_size, test_set_ratio)

#onset_list = df_f['onset'].tolist()
#print(max(onset_list))


#generate_wav_files_BC(wav_filename_A1,df_f_A1,'A1')
#generate_wav_files_BC(wav_filename_A2,df_f_A2,'A2')

X,y = first_classifier_BC(method,wav_filename_A1,wav_filename_A2,data_csv)
#X,y = first_classifier_mean(method,wav_filename_A1,frame_nb,recouvrement, data_set_size, test_set_ratio=0.25)

