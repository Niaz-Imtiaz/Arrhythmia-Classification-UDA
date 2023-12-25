import sys
import os
import glob
import time
import wfdb
import numpy as np
from scipy import signal

class Data_Processor():


  def load_data(self, path, lead_pos):

    #Load data and labels (annotations)
    ecg_db_path = path

    db_length = len(glob.glob1(ecg_db_path,"*.dat"))

    time_start=time.time()
    all_peaks=[]
    all_records=[]
    all_ann_symbols=[]
    for index, name in enumerate(glob.glob1(ecg_db_path,"*.dat")):
        name = name[:-4] #removes the .dat (4 letters) for each file
        print("file name: "+name + "  -->  " + str(index) + " from " + str(db_length)) #A

        record = wfdb.rdrecord(ecg_db_path + name) #record holds each file name without .dat extension
        ann = wfdb.rdann(ecg_db_path + name,'atr') # #ann holds the annotation of each data with .atr extension

        record = np.transpose(record.p_signal)
        record = record[lead_pos]
        all_records.append(record)

        orig_peaks=ann.sample
        orig_symbols=ann.symbol

        N = ['N', 'L', 'R', 'e', 'j', '.'] #Normal Beat
        V = ['V', 'E']                     #Ventricular Ectopic Beat (VEB)
        S = ['A', 'a', 'J', 'S']           #Supraventricular Ectopic Beat (SVEB)
        F = ['F']                          #Fusion Beat
        n_symbols=len(orig_symbols)
        for i in range(n_symbols):
          if orig_symbols[i] in N:
            orig_symbols[i]=0      #Target=0: Normal Beat
          elif orig_symbols[i] in V:
            orig_symbols[i]=1     #Target=1: Ventricular Ectopic Beat (VEB)
          elif orig_symbols[i] in S:
            orig_symbols[i]=2     #Target=2: Supraventricular Ectopic Beat (SVEB)
          elif orig_symbols[i] in F:
            orig_symbols[i]=3    #Target=3: Fusion Beat
          else:
            orig_symbols[i]=4    #Target=4: Other Beat

        all_peaks.append(orig_peaks) #All R-peaks
        all_ann_symbols.append(orig_symbols)  #All annoations or labels
    return all_records, all_peaks, all_ann_symbols



  def resample_ecg(self, all_records, all_peaks, curr_sam_freq, new_sam_freq, n_ecg):

    #Resample (Downsample or upsample) ECG with new sampling frequency (new_sam_freq)
    all_rec_ds=[]
    all_peaks_ds=[]
    for i in range(n_ecg):
      curr_sig=all_records[i]
      curr_peaks=all_peaks[i]
      fs=curr_sam_freq
      new_fs=new_sam_freq
      factor_ds=fs/new_fs
      curr_peaks_ds=np.round((curr_peaks/factor_ds),0)
      curr_peaks_ds=curr_peaks_ds.astype(int)
      curr_sig_ds = signal.resample_poly(curr_sig, new_fs, fs)
      all_rec_ds.append(curr_sig_ds)
      all_peaks_ds.append(curr_peaks_ds)

    return all_rec_ds, all_peaks_ds



  def get_RR_intervals(self, all_peaks, n_ecg):

    #Compute all RR intervals and Average RR interval
    all_RR=[]
    all_RR_sum=0
    for i in range(n_ecg):
      Rpeaks=all_peaks[i]
      RR = np.diff(Rpeaks)
      RR=np.concatenate(([0],RR))
      all_RR_sum=all_RR_sum+np.average(RR)
      RR=RR.tolist() #RR intervals of current ecg
      all_RR.append(RR)  #All RR intervals
    all_RR_avg=int(all_RR_sum/n_ecg) #Average of all RR intervals
    return all_RR, all_RR_avg



  def segment_ecg(self, all_records, all_peaks, all_ann_symbols, all_RR, avg_all_RR, n_ecg):

    #Create ecg segments (left half and right half from R-peak)
    all_RR_avg_half=int(avg_all_RR/2)
    ecg_seg_l=2*all_RR_avg_half+1     #Length of each ecg segment length 
    all_ecg_seg=[]                    #All ecg segments
    n_seg_each_ecg=[]                 #Number of segments taken from each ecg
    n_all_beats=-1
    pos_other_beat=[]                 #Position of other beats (not 'N', 'V', 'S', 'F')

    for n in range(n_ecg):
      curr_ecg=all_records[n]
      curr_Rpeaks=all_peaks[n]
      curr_symbols=all_ann_symbols[n]
      curr_RR=all_RR[n]

      n_curr_Rpeaks=len(curr_Rpeaks)
      n_curr_RR=len(curr_RR)
      n_curr_ecg=len(curr_ecg)
      n_each_ecg=0

      for i in range(n_curr_Rpeaks):  
        ecg_seg=np.zeros((ecg_seg_l))
        l_pos=int(np.floor(curr_Rpeaks[i]-all_RR_avg_half))

        r_pos=int(np.floor(curr_Rpeaks[i]+all_RR_avg_half))
        len_signal=curr_ecg.shape[0]

        if l_pos<0:
          pad_l=0-l_pos
          ecg_seg[pad_l:]=curr_ecg[0:r_pos+1]

        elif r_pos>=len_signal:
          pad_r=r_pos-len_signal+1
          ecg_seg[0:(ecg_seg_l-pad_r)]=curr_ecg[l_pos:len_signal]

        else:
          ecg_seg[0:ecg_seg_l]=curr_ecg[l_pos:r_pos+1]

        n_each_ecg=n_each_ecg+1

        all_ecg_seg.append(ecg_seg)

        n_all_beats=n_all_beats+1

        if curr_symbols[i]==4:
          pos_other_beat.append(n_all_beats)

      n_seg_each_ecg.append(n_each_ecg)
    return all_ecg_seg, ecg_seg_l, n_seg_each_ecg, pos_other_beat



  def get_RR_avg(self, all_RR, n_ecg):

      #Calculate all pre-RR and last eight pre-RR intervals
      tmp_avg_all_prev=[]
      tmp_avg_eight_prev=[]
      for n in range(n_ecg):
        curr_RRs=all_RR[n]
        n_curr_RRs=len(curr_RRs)
        avg_all_prev=np.zeros((n_curr_RRs))  #Average of all pre-RR intervals
        avg_eight_prev=np.zeros((n_curr_RRs))  #Average of last eight pre-RR intervals
        for i in range(n_curr_RRs):
          if i>0:
            avg_all_prev[i]=np.round(np.average(curr_RRs[1:i+1]),2) #Calculate average of all previous RR intervals from the current R-peak
          if i>7:
            avg_eight_prev[i]=np.round(np.average(curr_RRs[i-7:i+1]),2) #Calculate average of previous eight RR intervals from the current R-peak
        tmp_avg_all_prev.append(avg_all_prev)        #Average of all pre-RR intervals
        tmp_avg_eight_prev.append(avg_eight_prev)    #Average of last eight pre-RR intervals
      return tmp_avg_all_prev, tmp_avg_eight_prev



  def flatten_all(self, all_peaks, all_RR, all_ann_symbols, tmp_avg_all_prev, tmp_avg_eight_prev, n_ecg):

      #Flatten all R-peaks and RR intervals
      all_Rpeaks=[]              #All R-peaks  (flattened list)
      all_RRintervals=[]         #All RR intervals  (flattened list)
      all_avg_all_prev=[]        #All pre-RR intervals from the current R-peak  (flattened list)
      all_avg_eight_prev=[]      #All last eight pre-RR intervals from the current R-peak  (flattened list)
      all_target=[]              #All labels or annotations (flattened list)

      for n in range(n_ecg):
        curr_Rpeaks=all_peaks[n]
        l_curr_Rpeak=len(curr_Rpeaks)
        curr_symbols=all_ann_symbols[n]

        curr_RR=all_RR[n]
        curr_all_prev=tmp_avg_all_prev[n]
        curr_eight_prev=tmp_avg_eight_prev[n]

        for j in range(l_curr_Rpeak):
          all_Rpeaks.append(curr_Rpeaks[j])
          all_RRintervals.append(curr_RR[j])
          all_avg_all_prev.append(curr_all_prev[j])
          all_avg_eight_prev.append(curr_eight_prev[j])
          all_target.append(curr_symbols[j])
      return all_Rpeaks, all_RRintervals, all_avg_all_prev, all_avg_eight_prev, all_target