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
from pydub import AudioSegment
import convert_mp4_to_wav as conve




def foo(i):
    if i > 0:
        return 1
    else:
        return 0


def first_classifier_mean(method, wav_filename, frame_nb, recouvrement, data_set_size, test_set_ratio=0.25):
    samplerate, data = wavfile.read(wav_filename)

    l_data_set = []
    for i in range(data_set_size):
        l_data_set.append(data[i*frame_nb-recouvrement *
                          i:(i+1)*frame_nb-recouvrement*i])

    X = np.array(l_data_set)

    y = np.array([foo(np.mean(X[i, :])) for i in range(data_set_size)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_set_ratio, random_state=42)

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
    return X, y
# def extract_labels(file):


def max_time_distribution(df):
    duration_list = df['duration'].tolist()
    return max(duration_list)


def hist_time_distribution(df):
    duration_list = df['duration'].tolist()
    plt.hist(duration_list, density=True)
    # plt.show()


def generate_wav_files_BC(wav_filename, df, A):
    samplerate, data = wavfile.read(wav_filename)
    onset_list = df['onset'].tolist()
    m = max_time_distribution(df)
    i = 0
    for elem in onset_list:
        # print(elem,m)
        splitted_data = data[int(elem):int(elem+m)]
        write(A+"_BC"+str(i)+".wav", samplerate, splitted_data)
        i += 1


def cut_data(data, df, duration):
    if not df.empty:
        onset_list = df['onset'].tolist()
        feature_list = df['label'].tolist()
        l_data = []
        for elem in onset_list:
            l_data.append(data[int(elem):int(elem+duration)])
        return l_data
    else:
        return []


def label_and_fearture_set(wav_file_A1, wav_file_A2, data_csv):

    samplerate_A1, data_A1 = wavfile.read(wav_file_A1)
    samplerate_A2, data_A2 = wavfile.read(wav_file_A2)

    df_f_A1, df_r_A1 = load_csv_file.extract_df_times_csv_file(
        data_csv, samplerate_A1, 'A1')
    df_f_A2, df_r_A2 = load_csv_file.extract_df_times_csv_file(
        data_csv, samplerate_A2, 'A2')

    # max duration of feedback and resp
    if not df_f_A1.empty:
        m_f_A1 = max_time_distribution(df_f_A1)
    else:
        m_f_A1 = -1
    if not df_r_A1.empty:
        m_r_A1 = max_time_distribution(df_r_A1)
    else:
        m_r_A1 = -1
    if not df_f_A2.empty:
        m_f_A2 = max_time_distribution(df_f_A2)
    else:
        m_f_A2 = -1
    if not df_r_A2.empty:
        m_r_A2 = max_time_distribution(df_r_A2)
    else:
        m_r_A2 = -1
    
    m = max(m_r_A1, m_r_A2, m_f_A1, m_f_A2)

    l_data_A1_f = cut_data(data_A1, df_f_A1, m)
    l_data_A2_f = cut_data(data_A2, df_f_A2, m)

    l_data_A1_r = cut_data(data_A1, df_r_A1, m)
    l_data_A2_r = cut_data(data_A2, df_r_A2, m)

    return l_data_A1_f, l_data_A2_f, l_data_A1_r, l_data_A2_r


def generate_X_Y_feedback(wav_file_A1, wav_file_A2, csv_file):
    l_data_A1_f, l_data_A2_f, l_data_A1_r, l_data_A2_r = label_and_fearture_set(
        wav_file_A1, wav_file_A2, csv_file)

    l_data_f = l_data_A1_f + l_data_A2_f
    l_label_f = [1]*len(l_data_f)

    return l_data_f, l_label_f


def generate_X_Y_resp(wav_file_A1, wav_file_A2, csv_file):
    l_data_A1_f, l_data_A2_f, l_data_A1_r, l_data_A2_r = label_and_fearture_set(
        wav_file_A1, wav_file_A2, csv_file)

    l_data_r = l_data_A1_r + l_data_A2_r
    l_label_r = [2]*len(l_data_r)

    return l_data_r, l_label_r

# def generate_X_Y_random(wav_file_A1,wav_file_A2,csv_file):


def first_classifier_BC(method, l_X, l_y, test_set_ratio=0.25):


    X = np.array(l_data_set)
    y = np.array(l_label_set)  # 1 if feedbaxk, 2 if responses

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_set_ratio, random_state=42)

    if method == 'log_reg':
        classifier = LogisticRegression()
    if method == 'Neigh':
        classifier = KNeighborsClassifier()
    if method == 'Random_forest':
        classifier = RandomForestClassifier()

    classifier.fit(X_train, y_train)
    y_predict = []
    for i in range(len(y_test)):
        y_predict.append(classifier.predict([X_test[i]]))
    print(balanced_accuracy_score(y_test, y_predict))
    #print(cohen_kappa_score(y_test, y_predict))
    #return X, y

def X_Y_dataset( wav_file_A1, wav_file_A2, csv_file):

    l_data_f, l_label_f = generate_X_Y_feedback(
        wav_file_A1, wav_file_A2, csv_file)
    l_data_r, l_label_r = generate_X_Y_resp(wav_file_A1, wav_file_A2, csv_file)
    # r for reponses, f for feedback

    l_data_set = l_data_f + l_data_r
    l_label_set = l_label_f + l_label_r

    #X = np.array(l_data_set)
    #y = np.array(l_label_set)  # 1 if feedbaxk, 2 if responses

    return l_data_set, l_label_set

def get_csv_files_names(l_names_participants):
    d_csv = {}
    for elem in l_names_participants:
        csv_filename = '../data/Annotations-adults/'+elem[0]+'/AA-'+elem[1]+'-'+elem[2]+'-annotation.csv'
        d_csv[elem[0]]= csv_filename
    return d_csv



'''
get file wav names (after converstion by the file convert_mp4_to_wav.py)
'''


l_names_participants = [["AD", "AN", "DL"], ["BC", "BO", "CM"], ["GD", "GD", "DD"], ["JA", "JL", "AZ"], [
    "LA", "LA", "AN"], ["LB", "LD", "BF"], ["MC", "MJ", "CJ"], ["MHC", "MG", "CH"], ["MM", "ML", "MP"], ["XE", "XA", "EH"]]


d_partic_names = conve.file_names(l_names_participants,False) 
d_csv_filenames = get_csv_files_names(l_names_participants)




#d_data = {} #dict with as keys the name of folders. As elements dict . each of those dicts contain samplerate_A1, data_A1, samplerate_A2, data_A2


#for elem in d_partic_names:
#    samplerate_A1, data_A1 = wavfile.read(d_partic_names[elem][0])
#    samplerate_A2, data_A2 = wavfile.read(d_partic_names[elem][1])
#    d_conv = {"samplerate_A1":samplerate_A1,"samplerate_A2":samplerate_A2, "data_A1":data_A1, "data_A2":data_A2  }
#    d_data[elem] = d_conv


#w = wave.open(wav_filename, 'r')
# print(w.getnframes())
# print(w.getnchannels())
# plt.plot(data)
# plt.show


frame_nb = 100
recouvrement = 3
data_set_size = 4000
test_set_ratio = 0.25
method =  "log_reg" #'Random_forest'   'Neigh',

'''
dataframe data extration of BC info
hist of durations of BC
'''
# df_f_A1,df_r_A1=load_csv_file.extract_df_times_csv_file(data_csv,samplerate_A1,'A1')
# df_f_A2,df_r_A2=load_csv_file.extract_df_times_csv_file(data_csv,samplerate_A2,'A2')


# hist_time_distribution(df_f)
# hist_time_distribution(df_r)

'''
test classifier
'''

#X=first_classifier_mean('log_reg',wav_filename,frame_nb,recouvrement, data_set_size, test_set_ratio)

#onset_list = df_f['onset'].tolist()
# print(max(onset_list))


# generate_wav_files_BC(wav_filename_A1,df_f_A1,'A1')
# generate_wav_files_BC(wav_filename_A2,df_f_A2,'A2')

#X, y = first_classifier_BC(method, wav_filename_A1, wav_filename_A2, data_csv)
#X,y = first_classifier_mean(method,wav_filename_A1,frame_nb,recouvrement, data_set_size, test_set_ratio=0.25)


l_X = []
l_y = []
for elem in d_partic_names:
    wav_file_A1 = d_partic_names[elem][0]
    wav_file_A2 = d_partic_names[elem][1]
    csv_file = d_csv_filenames[elem]
    l_data_set,l_label_set=X_Y_dataset( wav_file_A1, wav_file_A2, csv_file)
    l_X = l_X + l_data_set
    l_y = l_y + l_label_set

#first_classifier_BC(method, l_X, l_y)
