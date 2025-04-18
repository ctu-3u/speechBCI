import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.manifold import TSNE

from localtool.operator import z_score, subt_mean

"""Data structure and parameters configuration"""
class DataStructConfig:
    # Configuration parameters
    num_channels = 128
    length_timebins = 127457 
    num_trials = 640

    length_timebins_s2 = 159136
    num_trials2_s2 = 800

    ## Phonemes chart
    phonemes_chart = ['B','CH','NOTHING','D','F','G','HH','JH','K','L','ER','M','N','NG','P','R','S','SH','DH','T','TH', 'V','W',\
        'Y','Z','ZH','OY','EH','EY','UH','IY','OW','UW','IH','AA','AW','AY','AH','AO','AE']

    consonant_clusteredchart = ['M', 'B', 'P', 'F', 'V', 'W', 'T', 'TH', 'DH', 'D', 'N', 'L', 'S', 'Z', 'CH', 'SH', 'ZH', 'JH', 'NG', 'K', 'G', 'Y', 'R']
    vowel_clusteredchart = ['OW', 'AO', 'AA', 'AW', 'AY', 'EH', 'AE', 'EY', 'IY', 'IH', 'AH', 'ER', 'OY', 'UW', 'UH', 'HH', 'NOTHING']
    phoneme_clusteredchart = np.append(consonant_clusteredchart[:], vowel_clusteredchart[:])

    # Load file
    def __init__(self, path, name1, name2):
        filepath = path
        filename = name1
        matfile_path = os.path.join(filepath, filename)
        self.data_sess1 = scipy.io.loadmat(matfile_path)

        filename = name2
        matfile_path = os.path.join(filepath, filename)
        self.data_sess2= scipy.io.loadmat(matfile_path)


"""Extract tuning data according to audio envelope peaks"""
def generate_phonemesdict_audio(tuning_data, length_timebins, num_channels, phonemes_chart, phoneme_clusteredchart):
    gotrial_start = np.array(tuning_data['goTrialEpochs'][:, 0])
    gotrial_end = np.array(tuning_data['goTrialEpochs'][:, 1])

    phonemes_indices = {}
    for i in range(len(phonemes_chart)):
        key = phonemes_chart[i]
        value = np.where(tuning_data['trialCues'] == (i + 1))[0]
        phonemes_indices[key] = value

    ### list of maximum audio time for each trial
    classifier_labels = []
    classifier_features = []

    ### Slice phonemese with aligenment by audio envelope, z-scored globally
    slicecut_left = 40
    slicecut_right = 0
    sp_temp = z_score(tuning_data['spikePow'][:, : num_channels].copy())

    phonemes_neuralact = {}
    for i in range(len(phonemes_chart)):
        trial_begins = [max(0, gotrial_start[phonemes_indices[phonemes_chart[i]][j]] - 10) for j in range(len(phonemes_indices[phonemes_chart[i]]))]
        trial_ends = [min(length_timebins, gotrial_end[phonemes_indices[phonemes_chart[i]][j]] - 10) for j in range(len(phonemes_indices[phonemes_chart[i]]))]

        trialslice_sp = np.zeros(num_channels)

        for j in range(len(phonemes_indices[phonemes_chart[i]])): # iterate by trials for each phoneme/condition
            audio_loudest = max(tuning_data['audioEnvelope'][trial_begins[j] - 1 : trial_ends[j]])
            audio_loudest_index = np.where(tuning_data['audioEnvelope'][trial_begins[j] - 1 : trial_ends[j]] == audio_loudest)[0][0] + trial_begins[j] - 1

            slice_left = audio_loudest_index - slicecut_left
            slice_right = audio_loudest_index + slicecut_right

            ### read a trial's neural activity data
            if slice_left >= 0 and slice_right <= length_timebins:
                trialslice_sp_temp = sp_temp[slice_left : slice_right, :] # Spike Power

            elif slice_left < 0:
                tempp_zeros = np.zeros((0 - slice_left, num_channels)) # contemporarily supplementary array
                trialslice_sp_temp = np.concatenate((tempp_zeros, sp_temp[slice_left : slice_right, :] )) # Spike Power

            elif slice_right > length_timebins:
                tempp_zeros = np.zeros((slice_right - length_timebins, num_channels)) # contemporarily supplementary array
                trialslice_sp_temp = np.concatenate((sp_temp[slice_left : slice_right, :] , tempp_zeros)) # Spike Power

            trialslice_sp_temp = np.mean(trialslice_sp_temp, axis = 0)
            trialslice_sp = trialslice_sp + trialslice_sp_temp

            classifier_labels.append(np.where(phoneme_clusteredchart == phonemes_chart[i])[0][0])
            classifier_features.append(trialslice_sp_temp)

        trialslice_sp /= len(phonemes_indices[phonemes_chart[i]])
        key = phonemes_chart[i]
        phonemes_neuralact[key] = trialslice_sp

    classifier_features = subt_mean(classifier_features)   
    return phonemes_neuralact, classifier_labels, classifier_features


"""Extract tuning data according to Go-cues"""
def generate_phonemesdict_gocue(tuning_data, length_timebins, num_channels, delay_times, phonemes_chart, phoneme_clusteredchart):
    gotrial_start = np.array(tuning_data['goTrialEpochs'][:, 0])

    phonemes_indices = {}
    for i in range(len(phonemes_chart)):
        key = phonemes_chart[i]
        value = np.where(tuning_data['trialCues'] == (i + 1))[0]
        phonemes_indices[key] = value

    ### list of maximum audio time for each trial
    classifier_labels = []
    classifier_features = []

    ### Slice phonemese with aligenment by audio envelope, z-scored globally
    delay = np.mean(delay_times[phonemes_chart[i]])
    slicecut_left = int(- delay + 40)
    slicecut_right = int(delay)

    sp_temp = tuning_data['spikePow'][:, : num_channels].copy()

    phonemes_neuralact = {}
    for i in range(len(phonemes_chart)):
        trial_begins = [gotrial_start[phonemes_indices[phonemes_chart[i]][j]] for j in range(len(phonemes_indices[phonemes_chart[i]]))]

        trialslice_sp = np.zeros(num_channels)

        for j in range(len(phonemes_indices[phonemes_chart[i]])): # iterate by trials for each phoneme/condition
            slice_left = trial_begins[j] - slicecut_left
            slice_right = trial_begins[j] + slicecut_right

            ### read a trial's neural activity data
            if slice_left >= 0 and slice_right <= length_timebins:
                trialslice_sp_temp = sp_temp[slice_left : slice_right, :] # Spike Power

            elif slice_left < 0:
                tempp_zeros = np.zeros((0 - slice_left, num_channels)) # contemporarily supplementary array
                trialslice_sp_temp = np.concatenate((tempp_zeros, sp_temp[slice_left : slice_right, :] )) # Spike Power

            elif slice_right > length_timebins:
                tempp_zeros = np.zeros((slice_right - length_timebins, num_channels)) # contemporarily supplementary array
                trialslice_sp_temp = np.concatenate((sp_temp[slice_left : slice_right, :] , tempp_zeros)) # Spike Power

            trialslice_sp_temp = np.mean(trialslice_sp_temp, axis = 0)
            trialslice_sp = trialslice_sp + trialslice_sp_temp

            classifier_labels.append(np.where(phoneme_clusteredchart == phonemes_chart[i])[0][0])
            classifier_features.append(trialslice_sp_temp)

        trialslice_sp /= len(phonemes_indices[phonemes_chart[i]])

        key = phonemes_chart[i]
        phonemes_neuralact[key] = trialslice_sp # an averaged activity for each phoneme
        
    mean = np.mean(classifier_features)
    std = np.std(classifier_features)
    for k, _ in phonemes_neuralact.items():
        phonemes_neuralact[k] -= mean
        phonemes_neuralact[k] /= std
    classifier_features = z_score(classifier_features)   
    return phonemes_neuralact, np.array(classifier_labels), np.array(classifier_features)

