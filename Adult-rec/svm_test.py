
import matplotlib.pyplot as plt
import numpy as np 


from scipy.io import wavfile

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from csv import reader


'''convert m4a into wav'''
# m4a_file = 'AD/AA-AN-DL-AN.m4a'
# track = AudioSegment.from_file(m4a_file,  format= 'm4a')
# file_handle = track.export(wav_filename, format='wav')

'''
convert wav into array
We have a one audio chanel (no stereo)
data len is 31046656
'''
wav_filename = "Adult-rec/AD/AA-AN-DL-AN.wav"



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


def first_classifier_BC(method,wav_filename,frame_nb,recouvrement, data_set_size, test_set_ratio=0.25):
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

frame_nb = 100
recouvrement = 3
data_set_size = 4000
test_set_ratio = 0.25
first_classifier('log_reg',wav_filename,frame_nb,recouvrement, data_set_size, test_set_ratio)

