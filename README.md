# Lis_internship_BC
In order to run the code, you have to have the following directory structure:
- Adult-rec and Annotations-children folrders from amubox (LIS laboratory) in a “data” folder
- “load_csv_file.py” and “lstm.py” in “src” folder
- Create an empty folder called “cut_wav_with_data_before_onset” in data.

run load_csv_file.py to:
- to convert m4a into wav (in Adukt-rec folder)
-  cut the audio into wav audios containing responses and backchannels (in  cut_wav_with_data_before_onset folder)
-  a generation a csv contraining file names and labels (in cut_wav_with_data_before_onset folder)

run lstm.py 
