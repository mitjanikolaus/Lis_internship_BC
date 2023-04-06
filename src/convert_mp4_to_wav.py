# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:38:38 2023

@author: Lenovo
"""
from pydub import AudioSegment



l_names_participants = [["AD", "AN", "DL"], ["BC", "BO", "CM"], ["GD", "GD", "DD"], ["JA", "JL", "AZ"], [
    "LA", "LA", "AN"], ["LB", "LD", "BF"], ["MC", "MJ", "CJ"], ["MHC", "MG", "CH"], ["MM", "ML", "MP"], ["XE", "XA", "EH"]]


def gererate_file_names(l,mp4 = True):
    """
    Parameters
    ----------
    l : list of size 3
        l[0] file name, l[1] name participant 1, l[]2 name participant 2

    Returns
    -------
    str: wav file and mp4 file names
    warning: can either rerurn a list of 4 elem, or a couple of onmly wav_file_names

    """
    m4a_filename_A1 = '../data/Adult-rec/'+l[0]+'/AA-'+l[1]+'-'+l[2]+'-'+l[1]+'.m4a'
    wav_filename_A1 = '../data/Adult-rec/' +l[0]+'/AA-'+l[1]+'-'+l[2]+'-'+l[1]+'.wav'
    m4a_filename_A2 = '../data/Adult-rec/'+l[0]+'/AA-'+l[1]+'-'+l[2]+'-'+l[2]+'.m4a'
    wav_filename_A2 = '../data/Adult-rec/'+l[0]+'/AA-'+l[1]+'-'+l[2]+'-'+l[2]+'.wav'
    if mp4:
        return  [m4a_filename_A1, wav_filename_A1, m4a_filename_A2, wav_filename_A2]
    else :
        return  (wav_filename_A1,wav_filename_A2)




def file_names(l_names_participants, mp4 = True):
    d_partic_names = {}
    for elem in l_names_participants:
        d_partic_names[elem[0]] = gererate_file_names(elem,mp4)
    return d_partic_names
        


def convert_files(d_partic_names):
    for elem in d_partic_names:
        l_files__names = d_partic_names[elem]
        m4a_filename_A1,wav_filename_A1,m4a_filename_A2,wav_filename_A2 = l_files__names
        track = AudioSegment.from_file(m4a_filename_A1,  format='m4a')
        file_handle = track.export(wav_filename_A1, format='wav')
        track = AudioSegment.from_file(m4a_filename_A2,  format='m4a')
        file_handle = track.export(wav_filename_A2, format='wav')
    
d_partic_names = file_names(l_names_participants)
convert_files(d_partic_names)


