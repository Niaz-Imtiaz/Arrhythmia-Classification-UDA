import sys
import os
import glob
import time
import wfdb
import numpy as np
import scipy
from scipy import signal

from Data_Processor import Data_Processor

database_path_s='./Data/MIT-BIH/'       #Source
database_path_t='./Data/St-Petersburg/' #Target (INCARTDB)
#database_path_t='./Data/ST-T/'         #Target (ESTDB)

class Prepare_Data():
  def preprocess_data(self):
    #Load and process data (source and target)

    data_obj=Data_Processor()
    lead_pos=0            #Lead II (Source)
    fs=360                #Sampling frequency of Source data

    #Load Source data
    all_records_s, all_peaks_s, all_ann_symbols_s= data_obj.load_data(database_path_s, lead_pos)


    #Signal denoising (Source) with bandpass filter (3-20 Hz)
    for i in range(len(all_records_s)):
        f1 = 3                                          
        f2 = 20                                         
        Wn = [f1*2/fs, f2*2/fs]
        N = 3
        a, b = signal.butter(N=N, Wn=Wn, btype='bandpass')   
        all_records_s[i] = signal.filtfilt(a, b, all_records_s[i], padlen=3*(max(len(a), len(b)) - 1))

    lead_pos=1     #Lead II (Target) (Cross Domain- INCARTDB)
    #lead_pos=10   #Lead V5 (Target) (Cross Domain and Cross channel- INCARTDB)
    #lead_pos=0    #Lead V5 (Target) (ESTDB)
    fs=257  #Sampling frequency of Target data (INCARTDB)
    #fs=250  #Sampling frequency of Target data (ESTDB)

    #Load Target data
    all_records_t, all_peaks_t, all_ann_symbols_t= data_obj.load_data(database_path_t, lead_pos)

    #Signal denoising (Target) with bandpass filter (3-20 Hz)
    for i in range(len(all_records_t)):
        f1 = 3
        f2 = 20
        Wn = [f1*2/fs, f2*2/fs]
        N = 3
        a, b = signal.butter(N=N, Wn=Wn, btype='bandpass')   
        all_records_t[i] = signal.filtfilt(a, b, all_records_t[i], padlen=3*(max(len(a), len(b)) - 1))

    n_ecg_s=len(all_peaks_s)
    n_ecg_t=len(all_peaks_t)

    curr_sam_freq_s=360   #Current ECG sampling frequency
    new_sam_freq_s=256    #Resampled (Upsampled or Downsampled) ECG sampling frequency

    #Resample (Downsample or upsample) Source ECG with new sampling frequency (256 Hz)
    all_records_s, all_peaks_s= data_obj.resample_ecg(all_records_s, all_peaks_s, curr_sam_freq_s, new_sam_freq_s, n_ecg_s)

    curr_sam_freq_t=257    
    new_sam_freq_t=256
    #Resample (Downsample or upsample) Target ECG with new sampling frequency (256 Hz)
    all_records_t, all_peaks_t= data_obj.resample_ecg(all_records_t, all_peaks_t, curr_sam_freq_t, new_sam_freq_t, n_ecg_t)

    #Calculate RR-intervals

    all_RR_s, avg_all_RR_s=data_obj.get_RR_intervals(all_peaks_s,n_ecg_s)
    all_RR_t, avg_all_RR_t=data_obj.get_RR_intervals(all_peaks_t,n_ecg_t)
    avg_all_RR_c=204


    #Create ECG segments
    all_ecg_seg_s, ecg_seg_l_s, n_seg_each_ecg_s, pos_other_beat_s=data_obj.segment_ecg(all_records_s, all_peaks_s, all_ann_symbols_s, all_RR_s, avg_all_RR_c, n_ecg_s)
    all_ecg_seg_t, ecg_seg_l_t, n_seg_each_ecg_t, pos_other_beat_t=data_obj.segment_ecg(all_records_t, all_peaks_t, all_ann_symbols_t, all_RR_t, avg_all_RR_c, n_ecg_t)

    #Calculate all pre-RR and last eight pre-RR intervals
    tmp_avg_all_prev_s, tmp_avg_eight_prev_s= data_obj.get_RR_avg(all_RR_s, n_ecg_s)
    tmp_avg_all_prev_t, tmp_avg_eight_prev_t= data_obj.get_RR_avg(all_RR_t, n_ecg_t)

    #Flatten all R-peaks and RR intervals
    all_Rpeaks_s, all_RRintervals_s, all_avg_all_prev_s, all_avg_eight_prev_s, all_target_s =data_obj.flatten_all(all_peaks_s, all_RR_s, all_ann_symbols_s, tmp_avg_all_prev_s, tmp_avg_eight_prev_s, n_ecg_s)
    all_Rpeaks_t, all_RRintervals_t, all_avg_all_prev_t, all_avg_eight_prev_t, all_target_t =data_obj.flatten_all(all_peaks_t, all_RR_t, all_ann_symbols_t, tmp_avg_all_prev_t, tmp_avg_eight_prev_t, n_ecg_t)

    #Delete data with Other Beat annotations

    for index in sorted(pos_other_beat_s, reverse=True):
        del all_Rpeaks_s[index]
        del all_RRintervals_s[index]
        del all_avg_all_prev_s[index]
        del all_avg_eight_prev_s[index]
        del all_target_s[index]
        del all_ecg_seg_s[index]

    for index in sorted(pos_other_beat_t, reverse=True):
        del all_Rpeaks_t[index]
        del all_RRintervals_t[index]
        del all_avg_all_prev_t[index]
        del all_avg_eight_prev_t[index]
        del all_target_t[index]
        del all_ecg_seg_t[index]


    #Reshape data and labels
    n_ecg_seg_s=len(all_ecg_seg_s)
    n_ecg_seg_t=len(all_ecg_seg_t)
    l_ecg_seg_s=all_ecg_seg_s[0].shape[0]
    l_ecg_seg_t=all_ecg_seg_t[0].shape[0]

    temp_s=np.array(all_ecg_seg_s)
    temp_t=np.array(all_ecg_seg_t)

    temp_data_s=np.reshape(temp_s,-1)
    temp_data_t=np.reshape(temp_t,-1)

    data_s=np.reshape(temp_data_s,(n_ecg_seg_s,l_ecg_seg_s,1))
    data_t=np.reshape(temp_data_t,(n_ecg_seg_t,l_ecg_seg_t,1))
    target_s=np.array(all_target_s)
    target_t=np.array(all_target_t)

    #Z-score normalization of data from source and target domain
    data_s=scipy.stats.zscore(data_s, axis=0, ddof=0, nan_policy='propagate')
    data_t=scipy.stats.zscore(data_t, axis=0, ddof=0, nan_policy='propagate')

    #Reshape RR intervals
    all_RRintervals_s=np.array(all_RRintervals_s)
    data_RRinterval_s=np.reshape(all_RRintervals_s,(n_ecg_seg_s,1))
    all_avg_all_prev_s=np.array(all_avg_all_prev_s)
    data_prevRR_s=np.reshape(all_avg_all_prev_s,(n_ecg_seg_s,1))
    all_avg_eight_prev_s=np.array(all_avg_eight_prev_s)
    data_prev_eight_RR_s=np.reshape(all_avg_eight_prev_s,(n_ecg_seg_s,1))
    all_RRintervals_t=np.array(all_RRintervals_t)
    data_RRinterval_t=np.reshape(all_RRintervals_t,(n_ecg_seg_t,1))
    all_avg_all_prev_t=np.array(all_avg_all_prev_t)
    data_prevRR_t=np.reshape(all_avg_all_prev_t,(n_ecg_seg_t,1))
    all_avg_eight_prev_t=np.array(all_avg_eight_prev_t)
    data_prev_eight_RR_t=np.reshape(all_avg_eight_prev_t,(n_ecg_seg_t,1))

    #Normalize RR intervals [0,2]
    data_RRinterval_s=((data_RRinterval_s-np.amin(data_RRinterval_s))*2)/(np.amax(data_RRinterval_s)-np.amin(data_RRinterval_s))
    data_prevRR_s=((data_prevRR_s-np.amin(data_prevRR_s))*2)/(np.amax(data_prevRR_s)-np.amin(data_prevRR_s))
    data_prev_eight_RR_s=((data_prev_eight_RR_s-np.amin(data_prev_eight_RR_s))*2)/(np.amax(data_prev_eight_RR_s)-np.amin(data_prev_eight_RR_s))
    data_RRinterval_t=((data_RRinterval_t-np.amin(data_RRinterval_t))*2)/(np.amax(data_RRinterval_t)-np.amin(data_RRinterval_t))
    data_prevRR_t=((data_prevRR_t-np.amin(data_prevRR_t))*2)/(np.amax(data_prevRR_t)-np.amin(data_prevRR_t))
    data_prev_eight_RR_t=((data_prev_eight_RR_t-np.amin(data_prev_eight_RR_t))*2)/(np.amax(data_prev_eight_RR_t)-np.amin(data_prev_eight_RR_t))

    #Data augmentation for ventricular ectopic, supraventricular ectopic, and fusion beats

    #Repeat data, labels and RR-interals for ventricular ectopic, supraventricular ectopic, and fusion beats by factor of 2, 5, and 10, respectively
    #Source Domain
    idx_s_1=np.where(target_s==1)
    target_s_1=target_s[idx_s_1]
    data_s_1=data_s[idx_s_1]
    data_RRinterval_s_1=data_RRinterval_s[idx_s_1]
    data_prevRR_s_1=data_prevRR_s[idx_s_1]
    data_prev_eight_RR_s_1=data_prev_eight_RR_s[idx_s_1]

    tmp_1_s = np.repeat(target_s_1, 2)
    target_s = np.concatenate((target_s, tmp_1_s), axis=0)

    tmp_1_s = np.repeat(data_s_1, 2, axis=2)
    tmp_1_s_tp = np.transpose(tmp_1_s)
    tmp_1_s_tp=np.swapaxes(tmp_1_s_tp, 1, 2)
    tmp_1_s_tp=np.reshape(tmp_1_s_tp,(-1,tmp_1_s_tp.shape[2]))
    tmp_1_s_tp=np.reshape(tmp_1_s_tp,(tmp_1_s_tp.shape[0],tmp_1_s_tp.shape[1],1))
    data_s = np.concatenate((data_s, tmp_1_s_tp), axis=0)

    tmp_1_s = np.repeat(data_RRinterval_s_1, 2, axis=1)
    tmp_1_s_tp = np.transpose(tmp_1_s)
    tmp_1_s_tp=np.reshape(tmp_1_s_tp,-1)
    tmp_1_s_tp=np.reshape(tmp_1_s_tp,(tmp_1_s_tp.shape[0],1))
    data_RRinterval_s = np.concatenate((data_RRinterval_s, tmp_1_s_tp), axis=0)

    tmp_1_s = np.repeat(data_prevRR_s_1, 2, axis=1)
    tmp_1_s_tp = np.transpose(tmp_1_s)
    tmp_1_s_tp=np.reshape(tmp_1_s_tp,-1)
    tmp_1_s_tp=np.reshape(tmp_1_s_tp,(tmp_1_s_tp.shape[0],1))
    data_prevRR_s = np.concatenate((data_prevRR_s, tmp_1_s_tp), axis=0)

    tmp_1_s = np.repeat(data_prev_eight_RR_s_1, 2, axis=1)
    tmp_1_s_tp = np.transpose(tmp_1_s)
    tmp_1_s_tp=np.reshape(tmp_1_s_tp,-1)
    tmp_1_s_tp=np.reshape(tmp_1_s_tp,(tmp_1_s_tp.shape[0],1))
    data_prev_eight_RR_s = np.concatenate((data_prev_eight_RR_s, tmp_1_s_tp), axis=0)

    idx_s_2=np.where(target_s==2)
    target_s_2=target_s[idx_s_2]
    data_s_2=data_s[idx_s_2]
    data_RRinterval_s_2=data_RRinterval_s[idx_s_2]
    data_prevRR_s_2=data_prevRR_s[idx_s_2]
    data_prev_eight_RR_s_2=data_prev_eight_RR_s[idx_s_2]

    tmp_2_s = np.repeat(target_s_2, 5)
    target_s = np.concatenate((target_s, tmp_2_s), axis=0)

    tmp_2_s = np.repeat(data_s_2, 5, axis=2)
    tmp_2_s_tp = np.transpose(tmp_2_s)
    tmp_2_s_tp=np.swapaxes(tmp_2_s_tp, 1, 2)
    tmp_2_s_tp=np.reshape(tmp_2_s_tp,(-1, tmp_2_s_tp.shape[2]))
    tmp_2_s_tp=np.reshape(tmp_2_s_tp,(tmp_2_s_tp.shape[0], tmp_2_s_tp.shape[1], 1))
    data_s = np.concatenate((data_s, tmp_2_s_tp), axis=0)

    tmp_2_s = np.repeat(data_RRinterval_s_2, 5, axis=1)
    tmp_2_s_tp = np.transpose(tmp_2_s)
    tmp_2_s_tp=np.reshape(tmp_2_s_tp, -1)
    tmp_2_s_tp=np.reshape(tmp_2_s_tp,(tmp_2_s_tp.shape[0], 1))
    data_RRinterval_s = np.concatenate((data_RRinterval_s, tmp_2_s_tp), axis=0)

    tmp_2_s = np.repeat(data_prevRR_s_2, 5, axis=1)
    tmp_2_s_tp = np.transpose(tmp_2_s)
    tmp_2_s_tp=np.reshape(tmp_2_s_tp, -1)
    tmp_2_s_tp=np.reshape(tmp_2_s_tp,(tmp_2_s_tp.shape[0], 1))
    data_prevRR_s = np.concatenate((data_prevRR_s, tmp_2_s_tp), axis=0)

    tmp_2_s = np.repeat(data_prev_eight_RR_s_2, 5, axis=1)
    tmp_2_s_tp = np.transpose(tmp_2_s)
    tmp_2_s_tp=np.reshape(tmp_2_s_tp,-1)
    tmp_2_s_tp=np.reshape(tmp_2_s_tp,(tmp_2_s_tp.shape[0], 1))
    data_prev_eight_RR_s = np.concatenate((data_prev_eight_RR_s, tmp_2_s_tp), axis=0)

    idx_s_3=np.where(target_s==3)
    target_s_3=target_s[idx_s_3]
    data_s_3=data_s[idx_s_3]
    data_RRinterval_s_3=data_RRinterval_s[idx_s_3]
    data_prevRR_s_3=data_prevRR_s[idx_s_3]
    data_prev_eight_RR_s_3=data_prev_eight_RR_s[idx_s_3]

    tmp_3_s = np.repeat(target_s_3, 10)
    target_s = np.concatenate((target_s, tmp_3_s), axis=0)

    tmp_3_s = np.repeat(data_s_3, 10, axis=2)
    tmp_3_s_tp = np.transpose(tmp_3_s)
    tmp_3_s_tp=np.swapaxes(tmp_3_s_tp, 1, 2)
    tmp_3_s_tp=np.reshape(tmp_3_s_tp,(-1, tmp_3_s_tp.shape[2]))
    tmp_3_s_tp=np.reshape(tmp_3_s_tp,(tmp_3_s_tp.shape[0], tmp_3_s_tp.shape[1], 1))
    data_s = np.concatenate((data_s, tmp_3_s_tp), axis=0)

    tmp_3_s = np.repeat(data_RRinterval_s_3, 10, axis=1)
    tmp_3_s_tp = np.transpose(tmp_3_s)
    tmp_3_s_tp=np.reshape(tmp_3_s_tp, -1)
    tmp_3_s_tp=np.reshape(tmp_3_s_tp,(tmp_3_s_tp.shape[0], 1))
    data_RRinterval_s = np.concatenate((data_RRinterval_s, tmp_3_s_tp), axis=0)

    tmp_3_s = np.repeat(data_prevRR_s_3, 10, axis=1)
    tmp_3_s_tp = np.transpose(tmp_3_s)
    tmp_3_s_tp=np.reshape(tmp_3_s_tp, -1)
    tmp_3_s_tp=np.reshape(tmp_3_s_tp,(tmp_3_s_tp.shape[0], 1))
    data_prevRR_s = np.concatenate((data_prevRR_s, tmp_3_s_tp), axis=0)

    tmp_3_s = np.repeat(data_prev_eight_RR_s_3, 10, axis=1)
    tmp_3_s_tp = np.transpose(tmp_3_s)
    tmp_3_s_tp=np.reshape(tmp_3_s_tp,-1)
    tmp_3_s_tp=np.reshape(tmp_3_s_tp,(tmp_3_s_tp.shape[0], 1))
    data_prev_eight_RR_s = np.concatenate((data_prev_eight_RR_s, tmp_3_s_tp), axis=0)

    #Repeat data, target and RR-interals for ventricular ectopic, supraventricular ectopic, and fusion beats by factor of 2, 5, and 10, respectively
    #Target Domain

    idx_t_1=np.where(target_t==1)
    target_t_1=target_t[idx_t_1]
    data_t_1=data_t[idx_t_1]
    data_RRinterval_t_1=data_RRinterval_t[idx_t_1]
    data_prevRR_t_1=data_prevRR_t[idx_t_1]
    data_prev_eight_RR_t_1=data_prev_eight_RR_t[idx_t_1]

    tmp_1_t = np.repeat(target_t_1, 2)
    target_t = np.concatenate((target_t, tmp_1_t), axis=0)

    tmp_1_t = np.repeat(data_t_1, 2, axis=2)
    tmp_1_t_tp = np.transpose(tmp_1_t)
    tmp_1_t_tp=np.swapaxes(tmp_1_t_tp, 1, 2)
    tmp_1_t_tp=np.reshape(tmp_1_t_tp,(-1,tmp_1_t_tp.shape[2]))
    tmp_1_t_tp=np.reshape(tmp_1_t_tp,(tmp_1_t_tp.shape[0],tmp_1_t_tp.shape[1],1))
    data_t = np.concatenate((data_t, tmp_1_t_tp), axis=0)

    tmp_1_t = np.repeat(data_RRinterval_t_1, 2, axis=1)
    tmp_1_t_tp = np.transpose(tmp_1_t)
    tmp_1_t_tp=np.reshape(tmp_1_t_tp,-1)
    tmp_1_t_tp=np.reshape(tmp_1_t_tp,(tmp_1_t_tp.shape[0],1))
    data_RRinterval_t = np.concatenate((data_RRinterval_t, tmp_1_t_tp), axis=0)

    tmp_1_t = np.repeat(data_prevRR_t_1, 2, axis=1)
    tmp_1_t_tp = np.transpose(tmp_1_t)
    tmp_1_t_tp=np.reshape(tmp_1_t_tp,-1)
    tmp_1_t_tp=np.reshape(tmp_1_t_tp,(tmp_1_t_tp.shape[0],1))
    data_prevRR_t = np.concatenate((data_prevRR_t, tmp_1_t_tp), axis=0)

    tmp_1_t = np.repeat(data_prev_eight_RR_t_1, 2, axis=1)
    tmp_1_t_tp = np.transpose(tmp_1_t)
    tmp_1_t_tp=np.reshape(tmp_1_t_tp,-1)
    tmp_1_t_tp=np.reshape(tmp_1_t_tp,(tmp_1_t_tp.shape[0],1))
    data_prev_eight_RR_t = np.concatenate((data_prev_eight_RR_t, tmp_1_t_tp), axis=0)

    idx_t_2=np.where(target_t==2)
    target_t_2=target_t[idx_t_2]
    data_t_2=data_t[idx_t_2]
    data_RRinterval_t_2=data_RRinterval_t[idx_t_2]
    data_prevRR_t_2=data_prevRR_t[idx_t_2]
    data_prev_eight_RR_t_2=data_prev_eight_RR_t[idx_t_2]

    tmp_2_t = np.repeat(target_t_2, 5)
    target_t = np.concatenate((target_t, tmp_2_t), axis=0)

    tmp_2_t = np.repeat(data_t_2, 5, axis=2)
    tmp_2_t_tp = np.transpose(tmp_2_t)
    tmp_2_t_tp=np.swapaxes(tmp_2_t_tp, 1, 2)
    tmp_2_t_tp=np.reshape(tmp_2_t_tp,(-1, tmp_2_t_tp.shape[2]))
    tmp_2_t_tp=np.reshape(tmp_2_t_tp,(tmp_2_t_tp.shape[0], tmp_2_t_tp.shape[1], 1))
    data_t = np.concatenate((data_t, tmp_2_t_tp), axis=0)

    tmp_2_t = np.repeat(data_RRinterval_t_2, 5, axis=1)
    tmp_2_t_tp = np.transpose(tmp_2_t)
    tmp_2_t_tp=np.reshape(tmp_2_t_tp, -1)
    tmp_2_t_tp=np.reshape(tmp_2_t_tp,(tmp_2_t_tp.shape[0], 1))
    data_RRinterval_t = np.concatenate((data_RRinterval_t, tmp_2_t_tp), axis=0)

    tmp_2_t = np.repeat(data_prevRR_t_2, 5, axis=1)
    tmp_2_t_tp = np.transpose(tmp_2_t)
    tmp_2_t_tp=np.reshape(tmp_2_t_tp, -1)
    tmp_2_t_tp=np.reshape(tmp_2_t_tp,(tmp_2_t_tp.shape[0], 1))
    data_prevRR_t = np.concatenate((data_prevRR_t, tmp_2_t_tp), axis=0)

    tmp_2_t = np.repeat(data_prev_eight_RR_t_2, 5, axis=1)
    tmp_2_t_tp = np.transpose(tmp_2_t)
    tmp_2_t_tp=np.reshape(tmp_2_t_tp,-1)
    tmp_2_t_tp=np.reshape(tmp_2_t_tp,(tmp_2_t_tp.shape[0], 1))
    data_prev_eight_RR_t = np.concatenate((data_prev_eight_RR_t, tmp_2_t_tp), axis=0)

    idx_t_3=np.where(target_t==3)
    target_t_3=target_t[idx_t_3]
    data_t_3=data_t[idx_t_3]
    data_RRinterval_t_3=data_RRinterval_t[idx_t_3]
    data_prevRR_t_3=data_prevRR_t[idx_t_3]
    data_prev_eight_RR_t_3=data_prev_eight_RR_t[idx_t_3]

    tmp_3_t = np.repeat(target_t_3, 10)
    target_t = np.concatenate((target_t, tmp_3_t), axis=0)

    tmp_3_t = np.repeat(data_t_3, 10, axis=2)
    tmp_3_t_tp = np.transpose(tmp_3_t)
    tmp_3_t_tp=np.swapaxes(tmp_3_t_tp, 1, 2)
    tmp_3_t_tp=np.reshape(tmp_3_t_tp,(-1, tmp_3_t_tp.shape[2]))
    tmp_3_t_tp=np.reshape(tmp_3_t_tp,(tmp_3_t_tp.shape[0], tmp_3_t_tp.shape[1], 1))
    data_t = np.concatenate((data_t, tmp_3_t_tp), axis=0)

    tmp_3_t = np.repeat(data_RRinterval_t_3, 10, axis=1)
    tmp_3_t_tp = np.transpose(tmp_3_t)
    tmp_3_t_tp=np.reshape(tmp_3_t_tp, -1)
    tmp_3_t_tp=np.reshape(tmp_3_t_tp,(tmp_3_t_tp.shape[0], 1))
    data_RRinterval_t = np.concatenate((data_RRinterval_t, tmp_3_t_tp), axis=0)

    tmp_3_t = np.repeat(data_prevRR_t_3, 10, axis=1)
    tmp_3_t_tp = np.transpose(tmp_3_t)
    tmp_3_t_tp=np.reshape(tmp_3_t_tp, -1)
    tmp_3_t_tp=np.reshape(tmp_3_t_tp,(tmp_3_t_tp.shape[0], 1))
    data_prevRR_t = np.concatenate((data_prevRR_t, tmp_3_t_tp), axis=0)

    tmp_3_t = np.repeat(data_prev_eight_RR_t_3, 10, axis=1)
    tmp_3_t_tp = np.transpose(tmp_3_t)
    tmp_3_t_tp=np.reshape(tmp_3_t_tp,-1)
    tmp_3_t_tp=np.reshape(tmp_3_t_tp,(tmp_3_t_tp.shape[0], 1))
    data_prev_eight_RR_t = np.concatenate((data_prev_eight_RR_t, tmp_3_t_tp), axis=0)

    #print('#### Source Shape ###')
    #print(data_s.shape)
    #print(target_s.shape)
    #print(data_RRinterval_s.shape)
    #print(data_prevRR_s.shape)
    #print(data_prev_eight_RR_s.shape)
    #print('#### Target Shape ###')
    #print(data_t.shape)
    #print(target_t.shape)
    #print(data_RRinterval_t.shape)
    #print(data_prevRR_t.shape)
    #print(data_prev_eight_RR_t.shape)
	
    return data_s, target_s, data_RRinterval_s, data_prevRR_s, data_prev_eight_RR_s, data_t, target_t, data_RRinterval_t, data_prevRR_t, data_prev_eight_RR_t
