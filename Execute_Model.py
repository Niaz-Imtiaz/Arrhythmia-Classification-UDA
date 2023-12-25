import sys
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from Data_Processor import Data_Processor
from Prepare_Data import Prepare_Data
from Prepare_Train_Test import Prepare_Train_Test
import Create_Model as cm
from Compute_Clusters import Compute_Clusters


class Execute_Model():

  def exec_model(self, G, C1, C2, train_data_batch_s, train_RRinterval_batch_s, train_prevRR_batch_s, train_prev_eightRR_batch_s, train_label_batch_s, train_data_batch_t, train_RRinterval_batch_t, train_prevRR_batch_t, train_prev_eightRR_batch_t, train_label_batch_t,test_data_batch_t, test_RRinterval_batch_t, test_prevRR_batch_t, test_prev_eightRR_batch_t, test_label_batch_t, input_shape, num_classes):

    opt_g = optim.Adam(G.parameters(), lr=0.001, weight_decay=0.0005)
    opt_c1 = optim.Adam(C1.parameters(), lr=0.001, weight_decay=0.0005)
    opt_c2 = optim.Adam(C2.parameters(), lr=0.001, weight_decay=0.0005)

    #### Pre-Training ####
    G.train()
    C1.train()
    C2.train()
    torch.cuda.manual_seed(1)
    criterion = nn.CrossEntropyLoss().cuda()

    Epochs=20
    for epoch in range(Epochs):

      for i in range(train_data_batch_s.shape[0]):
    
          train_data_batch_tmp_s=train_data_batch_s[i]
          train_data_batch_tmp_t=train_data_batch_t[i]
          train_tensor_s = torch.from_numpy(train_data_batch_tmp_s)
          train_tensor_t = torch.from_numpy(train_data_batch_tmp_t)
          train_tensor_s=train_tensor_s.to('cuda')
          train_tensor_t=train_tensor_t.to('cuda')
          train_tensor_s=train_tensor_s.float()
          train_tensor_t=train_tensor_t.float()

          train_RRinterval_batch_tmp_s=train_RRinterval_batch_s[i]
          train_prevRR_batch_tmp_s=train_prevRR_batch_s[i]
          train_prev_eightRR_batch_tmp_s=train_prev_eightRR_batch_s[i]
          train_RRinterval_batch_tmp_t=train_RRinterval_batch_t[i]
          train_prevRR_batch_tmp_t=train_prevRR_batch_t[i]
          train_prev_eightRR_batch_tmp_t=train_prev_eightRR_batch_t[i]

          train_tensor_RRinterval_s = torch.from_numpy(train_RRinterval_batch_tmp_s)
          train_tensor_prevRR_s = torch.from_numpy(train_prevRR_batch_tmp_s)
          train_tensor_prev_eightRR_s = torch.from_numpy(train_prev_eightRR_batch_tmp_s)
          train_tensor_RRinterval_t = torch.from_numpy(train_RRinterval_batch_tmp_t)
          train_tensor_prevRR_t = torch.from_numpy(train_prevRR_batch_tmp_t)
          train_tensor_prev_eightRR_t = torch.from_numpy(train_prev_eightRR_batch_tmp_t)

          train_tensor_RRinterval_s=train_tensor_RRinterval_s.to('cuda')
          train_tensor_prevRR_s=train_tensor_prevRR_s.to('cuda')
          train_tensor_prev_eightRR_s=train_tensor_prev_eightRR_s.to('cuda')
          train_tensor_RRinterval_t=train_tensor_RRinterval_t.to('cuda')
          train_tensor_prevRR_t=train_tensor_prevRR_t.to('cuda')
          train_tensor_prev_eightRR_t=train_tensor_prev_eightRR_t.to('cuda')

          train_tensor_RRinterval_s=train_tensor_RRinterval_s.float()
          train_tensor_prevRR_s=train_tensor_prevRR_s.float()
          train_tensor_prev_eightRR_s=train_tensor_prev_eightRR_s.float()
          train_tensor_RRinterval_t=train_tensor_RRinterval_t.float()
          train_tensor_prevRR_t=train_tensor_prevRR_t.float()
          train_tensor_prev_eightRR_t=train_tensor_prev_eightRR_t.float()

          train_label_batch_tmp_s=train_label_batch_s[i]

          train_label_tmp_tensor_s = torch.from_numpy(train_label_batch_tmp_s)
          train_label_tmp_one_hot_s=torch.nn.functional.one_hot(train_label_tmp_tensor_s, 4)
          train_label_tmp_tensor_s=train_label_tmp_one_hot_s.to('cuda')

          train_label_tmp_tensor_s = train_label_tmp_tensor_s.to(torch.float)

          train_tensor_s=torch.swapaxes(train_tensor_s, 1, 2)
          train_tensor_t=torch.swapaxes(train_tensor_t, 1, 2)

          reset_grad(opt_g, opt_c1, opt_c2)

          feat_s = G(train_tensor_s)
          output_C1, output_C1_prev = C1(feat_s, train_tensor_RRinterval_s, train_tensor_prevRR_s, train_tensor_prev_eightRR_s)
          output_C2, output_C2_prev = C2(feat_s, train_tensor_RRinterval_s, train_tensor_prevRR_s, train_tensor_prev_eightRR_s)

          output_C1_C2_prev=(output_C1_prev+output_C2_prev)/2
          
          loss_C1_C2_s=F.cross_entropy(output_C1_C2_prev, train_label_tmp_tensor_s)

          #Apply DRO
          groupdro_eta=1
          loss_cls= (groupdro_eta * loss_C1_C2_s).exp()

          Alpha_DIS=0.5

          loss_dis = discrepancy(output_C1_prev, output_C2_prev)
          loss=loss_cls + (Alpha_DIS * loss_dis)

          loss.backward()
          #Adjust F, C1 and C2 parameters
          opt_g.step()
          opt_c1.step()
          opt_c2.step()
          reset_grad(opt_g, opt_c1, opt_c2)


          #print('Epoch: '+str(epoch+1))
          #print('Iteration:  '+str(i+1))
          #print(loss)

    ### Pre-Training End ###

    #Compute source clusters
    data_obj=Compute_Clusters()
    all_centers_s= data_obj.get_source_centers(G, C1, C2, train_data_batch_s, train_RRinterval_batch_s, train_prevRR_batch_s, train_prev_eightRR_batch_s, train_label_batch_s)

    Epochs=10

    #### Training With Cluster Loss ####
    for epoch in range(Epochs):

        for i in range(train_data_batch_s.shape[0]):

            train_data_batch_tmp_s=train_data_batch_s[i]
            train_tensor_s = torch.from_numpy(train_data_batch_tmp_s)
            train_tensor_s=train_tensor_s.to('cuda')
            train_tensor_s=train_tensor_s.float()

            train_RRinterval_batch_tmp_s=train_RRinterval_batch_s[i]
            train_prevRR_batch_tmp_s=train_prevRR_batch_s[i]
            train_prev_eightRR_batch_tmp_s=train_prev_eightRR_batch_s[i]

            train_tensor_RRinterval_s = torch.from_numpy(train_RRinterval_batch_tmp_s)
            train_tensor_prevRR_s = torch.from_numpy(train_prevRR_batch_tmp_s)
            train_tensor_prev_eightRR_s = torch.from_numpy(train_prev_eightRR_batch_tmp_s)

            train_tensor_RRinterval_s=train_tensor_RRinterval_s.to('cuda')
            train_tensor_prevRR_s=train_tensor_prevRR_s.to('cuda')
            train_tensor_prev_eightRR_s=train_tensor_prev_eightRR_s.to('cuda')

            train_tensor_RRinterval_s=train_tensor_RRinterval_s.float()
            train_tensor_prevRR_s=train_tensor_prevRR_s.float()
            train_tensor_prev_eightRR_s=train_tensor_prev_eightRR_s.float()

            train_label_batch_tmp_s=train_label_batch_s[i]

            train_label_tmp_tensor_s = torch.from_numpy(train_label_batch_tmp_s)
            train_label_tmp_one_hot_s=torch.nn.functional.one_hot(train_label_tmp_tensor_s, 4)
            train_label_tmp_tensor_s=train_label_tmp_one_hot_s.to('cuda')

            train_label_tmp_tensor_s = train_label_tmp_tensor_s.to(torch.float)
            centers_tmp_s=all_centers_s

            centers_tmp_s = torch.from_numpy(centers_tmp_s)
            centers_tmp_s=centers_tmp_s.to('cuda')

            centers_tensor_s = centers_tmp_s.to(torch.float)

            train_tensor_s=torch.swapaxes(train_tensor_s, 1, 2)

            reset_grad(opt_g, opt_c1, opt_c2)

            feat_s = G(train_tensor_s)
            output_C1_s, output_C1_prev_s = C1(feat_s, train_tensor_RRinterval_s, train_tensor_prevRR_s, train_tensor_prev_eightRR_s)
            output_C2_s, output_C2_prev_s = C2(feat_s, train_tensor_RRinterval_s, train_tensor_prevRR_s, train_tensor_prev_eightRR_s)

            output_C1_C2_prev_s=(output_C1_prev_s+output_C2_prev_s)/2
       
            loss_C1_C2_s=F.cross_entropy(output_C1_C2_prev_s, train_label_tmp_tensor_s)

            #loss_intra: cluster-compacting loss
            #loss_inter: cluster-separating loss

            loss_cs = 0
            loss_intra = 0
            loss_inter = 0
            lr_cs=0.01

            for l in range(4):

                label_batch_s=train_label_tmp_tensor_s.data.max(1)[1]
                _idx_s=torch.where(label_batch_s==l)
                _feat_s=feat_s[_idx_s]
                if _feat_s.shape[0]!=0:
                    m_feat_s = torch.mean(_feat_s, dim=0)
                    m_feat_s=m_feat_s.to('cuda')
                    delta_cs_l = centers_tensor_s[l] - m_feat_s
                    delta_cs_l_np = delta_cs_l.cpu().detach().numpy()
                    all_centers_s[l] = all_centers_s[l] - lr_cs * delta_cs_l_np
                    loss_cs_l = L2Distance(m_feat_s, centers_tensor_s[l])
                    loss_cs += loss_cs_l

                    bs_ = _feat_s.shape[0]
                    cl_feat_s = centers_tensor_s[l].repeat((bs_, 1))
                    cl_feat_s=cl_feat_s.to('cuda')
                    loss_intra_l = L2Distance(_feat_s, cl_feat_s, dim=1) / bs_
                    loss_intra += loss_intra_l

            THR_M= 50
            c_inter=0
            for m in range(4 - 1):
                for n in range(m + 1, 4):
                    c_m=torch.count_nonzero(centers_tensor_s[m])
                    c_n=torch.count_nonzero(centers_tensor_s[n])
                    if c_m!=0 and c_n!=0:
                          loss_inter_mn = torch.max(THR_M - L2Distance(centers_tensor_s[m], centers_tensor_s[n]),
                                                    torch.FloatTensor([0]).cuda()).squeeze()
                          loss_inter += loss_inter_mn
                          c_inter=c_inter+1
            if c_inter!=0:
                loss_inter = loss_inter / c_inter

            loss_intra = loss_intra / 4

            Gamma_INTER= 0.1
            Gamma_INTRA= 0.1

            loss = loss_C1_C2_s +  (loss_intra * Gamma_INTRA) + (loss_inter * Gamma_INTER) 

            reset_grad(opt_g, opt_c1, opt_c2)
            loss.backward()
            opt_g.step()
            opt_c1.step()
            opt_c2.step()



    #### Training With Cluster Loss End ####

    #Compute source clusters
    all_centers_s= data_obj.get_source_centers(G, C1, C2, train_data_batch_s, train_RRinterval_batch_s, train_prevRR_batch_s, train_prev_eightRR_batch_s, train_label_batch_s)

    #Compute mean intra-cluster distance and mean classifier discrepancy
    mean_dist_classifier_s, mean_dist_feat_s, all_mean_dist_feat_s= data_obj.get_mean_dist(G, C1, C2, train_data_batch_s, train_RRinterval_batch_s, train_prevRR_batch_s, train_prev_eightRR_batch_s, train_label_batch_s, all_centers_s)

    #Compute target clusters
    all_centers_t = data_obj.get_target_centers(G, C1, C2, train_data_batch_t, train_RRinterval_batch_t, train_prevRR_batch_t, train_prev_eightRR_batch_t, all_centers_s, mean_dist_classifier_s, mean_dist_feat_s, all_mean_dist_feat_s)

    #Compute average clusters
    centers=np.empty((all_centers_s.shape[0],all_centers_s.shape[1]))
    for l in range(4):
        centers[l]=(all_centers_s[l]+all_centers_t[l])/2   #Mean of source and target centers for each of the 4 clusters

    #### Domain Adaptation ####
    Epochs=10

    mean_dist_classifier_s=torch.from_numpy(mean_dist_classifier_s)
    mean_dist_classifier_s=mean_dist_classifier_s.to('cuda')
    mean_dist_classifier_s = mean_dist_classifier_s.to(torch.float)
    mean_dist_classifier_s=mean_dist_classifier_s
    mean_dist_feat_s = torch.from_numpy(mean_dist_feat_s)
    mean_dist_feat_s=mean_dist_feat_s.to('cuda')
    mean_dist_feat_s = mean_dist_feat_s.to(torch.float)
    mean_dist_feat_s=mean_dist_feat_s+1

    all_mean_dist_feat_s = torch.from_numpy(all_mean_dist_feat_s)
    all_mean_dist_feat_s=all_mean_dist_feat_s.to('cuda')
    all_mean_dist_feat_s = all_mean_dist_feat_s.to(torch.float)
    all_mean_dist_feat_s[0]=all_mean_dist_feat_s[0]+1
    all_mean_dist_feat_s[1]=all_mean_dist_feat_s[1]+1
    all_mean_dist_feat_s[2]=all_mean_dist_feat_s[2]+1
    all_mean_dist_feat_s[3]=all_mean_dist_feat_s[3]+1

    #### Training with final loss function #####
    for epoch in range(Epochs):

        for i in range(train_data_batch_s.shape[0]):

            train_data_batch_tmp_s=train_data_batch_s[i]
            train_data_batch_tmp_t=train_data_batch_t[i]
            train_tensor_s = torch.from_numpy(train_data_batch_tmp_s)
            train_tensor_t = torch.from_numpy(train_data_batch_tmp_t)
            train_tensor_s=train_tensor_s.to('cuda')
            train_tensor_t=train_tensor_t.to('cuda')
            train_tensor_s=train_tensor_s.float()
            train_tensor_t=train_tensor_t.float()

            train_RRinterval_batch_tmp_s=train_RRinterval_batch_s[i]
            train_prevRR_batch_tmp_s=train_prevRR_batch_s[i]
            train_prev_eightRR_batch_tmp_s=train_prev_eightRR_batch_s[i]
            train_RRinterval_batch_tmp_t=train_RRinterval_batch_t[i]
            train_prevRR_batch_tmp_t=train_prevRR_batch_t[i]
            train_prev_eightRR_batch_tmp_t=train_prev_eightRR_batch_t[i]

            train_tensor_RRinterval_s = torch.from_numpy(train_RRinterval_batch_tmp_s)
            train_tensor_prevRR_s = torch.from_numpy(train_prevRR_batch_tmp_s)
            train_tensor_prev_eightRR_s = torch.from_numpy(train_prev_eightRR_batch_tmp_s)
            train_tensor_RRinterval_t = torch.from_numpy(train_RRinterval_batch_tmp_t)
            train_tensor_prevRR_t = torch.from_numpy(train_prevRR_batch_tmp_t)
            train_tensor_prev_eightRR_t = torch.from_numpy(train_prev_eightRR_batch_tmp_t)

            train_tensor_RRinterval_s=train_tensor_RRinterval_s.to('cuda')
            train_tensor_prevRR_s=train_tensor_prevRR_s.to('cuda')
            train_tensor_prev_eightRR_s=train_tensor_prev_eightRR_s.to('cuda')
            train_tensor_RRinterval_t=train_tensor_RRinterval_t.to('cuda')
            train_tensor_prevRR_t=train_tensor_prevRR_t.to('cuda')
            train_tensor_prev_eightRR_t=train_tensor_prev_eightRR_t.to('cuda')

            train_tensor_RRinterval_s=train_tensor_RRinterval_s.float()
            train_tensor_prevRR_s=train_tensor_prevRR_s.float()
            train_tensor_prev_eightRR_s=train_tensor_prev_eightRR_s.float()
            train_tensor_RRinterval_t=train_tensor_RRinterval_t.float()
            train_tensor_prevRR_t=train_tensor_prevRR_t.float()
            train_tensor_prev_eightRR_t=train_tensor_prev_eightRR_t.float()

            train_label_batch_tmp_s=train_label_batch_s[i]

            train_label_tmp_tensor_s = torch.from_numpy(train_label_batch_tmp_s)
            train_label_tmp_one_hot_s=torch.nn.functional.one_hot(train_label_tmp_tensor_s, 4)
            train_label_tmp_tensor_s=train_label_tmp_one_hot_s.to('cuda')

            train_label_tmp_tensor_s = train_label_tmp_tensor_s.to(torch.float)

            centers_tmp_s=all_centers_s
            centers_tmp_t=all_centers_t
            centers_tmp=centers

            centers_tmp_s = torch.from_numpy(centers_tmp_s)
            centers_tmp_s=centers_tmp_s.to('cuda')
            centers_tmp_t = torch.from_numpy(centers_tmp_t)
            centers_tmp_t=centers_tmp_t.to('cuda')
            centers_tmp = torch.from_numpy(centers_tmp)
            centers_tmp=centers_tmp.to('cuda')

            centers_tensor_s = centers_tmp_s.to(torch.float)
            centers_tensor_t = centers_tmp_t.to(torch.float)
            centers_tensor = centers_tmp.to(torch.float)

            train_tensor_s=torch.swapaxes(train_tensor_s, 1, 2)
            train_tensor_t=torch.swapaxes(train_tensor_t, 1, 2)

            reset_grad(opt_g, opt_c1, opt_c2)

            feat_s = G(train_tensor_s)
            output_C1_s, output_C1_prev_s = C1(feat_s, train_tensor_RRinterval_s, train_tensor_prevRR_s, train_tensor_prev_eightRR_s)
            output_C2_s, output_C2_prev_s = C2(feat_s, train_tensor_RRinterval_s, train_tensor_prevRR_s, train_tensor_prev_eightRR_s)
            output_C1_C2_prev_s=(output_C1_prev_s+output_C2_prev_s)/2
            loss_Cls = F.cross_entropy(output_C1_C2_prev_s, train_label_tmp_tensor_s)  #Classification Loss

            feat_t = G(train_tensor_t)
            output_C1_t, output_C1_prev_t = C1(feat_t, train_tensor_RRinterval_t, train_tensor_prevRR_t, train_tensor_prev_eightRR_t)
            output_C2_t, output_C2_prev_t = C2(feat_t, train_tensor_RRinterval_t, train_tensor_prevRR_t, train_tensor_prev_eightRR_t)
            output_C1_C2_t= (output_C1_t+output_C2_t)/2

            max_pred_val_t,max_pred_idx_t=torch.max(output_C1_C2_t,1)
            confident_pred_idx_t=torch.where(max_pred_val_t>0.99)
            confident_pred_class_t=max_pred_idx_t[confident_pred_idx_t]
            confident_pred_output_t=max_pred_val_t[confident_pred_idx_t]
            confident_feat_t=feat_t[confident_pred_idx_t]
            confident_C1_t=output_C1_prev_t[confident_pred_idx_t]
            confident_C2_t=output_C2_prev_t[confident_pred_idx_t]
            jj=0
            for ii in range(confident_feat_t.shape[0]):
                if ((L2Distance(confident_C1_t[ii], confident_C2_t[ii])<mean_dist_classifier_s)):
                    confident_pred_class_t[jj]=confident_pred_class_t[ii]
                    confident_pred_output_t[jj]=confident_pred_output_t[ii]
                    confident_feat_t[jj]=confident_feat_t[ii]
                    confident_C1_t[jj]=confident_C1_t[ii]
                    confident_C2_t[jj]=confident_C2_t[ii]
                    jj=jj+1
            confident_pred_class_t=confident_pred_class_t[0:jj]
            confident_pred_output_t=confident_pred_output_t[0:jj]
            confident_feat_t=confident_feat_t[0:jj]
            confident_C1_t=confident_C1_t[0:jj]
            confident_C2_t=confident_C2_t[0:jj]

            loss_intra_t = 0
            loss_inter_t = 0
            loss_intra_s = 0
            loss_inter_s = 0
            loss_cd=0
            loss_cmd = 0
            lr_cs=0.01
            lr_c=0.01

            pesudo_label_nums=torch.zeros(4)
            tmp_centers_t=torch.zeros(4, feat_t.shape[1])
            pesudo_label_nums=pesudo_label_nums.to('cuda')
            tmp_centers_t=tmp_centers_t.to('cuda')

            #Calculate cluster-compacting loss for target
            for l in range(4):

                _idx_t=torch.where(confident_pred_class_t==l)
                _feat_t=confident_feat_t[_idx_t]

                jj=0
                for ii in range(_feat_t.shape[0]):
                    if ((L2Distance(centers_tensor[l], _feat_t[ii]) < all_mean_dist_feat_s[l])):
                        _feat_t[jj]=_feat_t[ii]
                        jj=jj+1
                _feat_t=_feat_t[0:jj]
                pesudo_label_nums[l]=_feat_t.shape[0]

                if _feat_t.shape[0]!=0:
                    m_feat_t = torch.mean(_feat_t, dim=0)
                    tmp_centers_t[l] = m_feat_t
                    m_feat_t=m_feat_t.to('cuda')

                    delta_cs_l_t = centers_tensor_t[l] - m_feat_t
                    delta_cs_l_t_np = delta_cs_l_t.cpu().detach().numpy()
                    all_centers_t[l] = all_centers_t[l] - lr_cs * delta_cs_l_t_np

                    all_centers_tensor_t = torch.from_numpy(all_centers_t)
                    all_centers_tensor_t=all_centers_tensor_t.to('cuda')
                    all_centers_tensor_t = all_centers_tensor_t.to(torch.float)

                    bs_ = _feat_t.shape[0]
                    cl_feat_t = all_centers_tensor_t[l].repeat((bs_, 1))
                    cl_feat_t=cl_feat_t.to('cuda')
                    loss_intra_l_t = L2Distance(_feat_t, cl_feat_t, dim=1) / bs_
                    loss_intra_t += loss_intra_l_t
            loss_intra_t = loss_intra_t / 4

            true_label_nums=torch.zeros(4)
            tmp_centers_s=torch.zeros(4, feat_s.shape[1])
            true_label_nums=true_label_nums.to('cuda')
            tmp_centers_s=tmp_centers_s.to('cuda')

            #Calculate cluster-compacting loss for source
            for l in range(4):

                label_batch_s=train_label_tmp_tensor_s.data.max(1)[1]
                _idx_s=torch.where(label_batch_s==l)
                _feat_s=feat_s[_idx_s]
                true_label_nums[l]=_feat_s.shape[0]
                if _feat_s.shape[0]!=0:
                    m_feat_s = torch.mean(_feat_s, dim=0)
                    tmp_centers_s[l] = m_feat_s
                    m_feat_s=m_feat_s.to('cuda')
                    delta_cs_l = centers_tensor_s[l] - m_feat_s
                    delta_cs_l_np = delta_cs_l.cpu().detach().numpy()
                    all_centers_s[l] = all_centers_s[l] - lr_cs * delta_cs_l_np
                    all_centers_tensor_s = torch.from_numpy(all_centers_s)
                    all_centers_tensor_s=all_centers_tensor_s.to('cuda')
                    all_centers_tensor_s = all_centers_tensor_s.to(torch.float)

                    bs_ = _feat_s.shape[0]
                    cl_feat_s = all_centers_tensor_s[l].repeat((bs_, 1))
                    cl_feat_s=cl_feat_s.to('cuda')
                    loss_intra_l_s = L2Distance(_feat_s, cl_feat_s, dim=1) / bs_
                    loss_intra_s += loss_intra_l_s
            loss_intra_s = loss_intra_s / 4


            all_centers_tensor_t = torch.from_numpy(all_centers_t)
            all_centers_tensor_t=all_centers_tensor_t.to('cuda')
            all_centers_tensor_t = all_centers_tensor_t.to(torch.float)

            THR_M= 50
            c_inter=0

            #Calculate cluster-separating loss for target
            for m in range(4 - 1):
                for n in range(m + 1, 4):
                    c_m=torch.count_nonzero(all_centers_tensor_t[m])
                    c_n=torch.count_nonzero(all_centers_tensor_t[n])
                    if c_m!=0 and c_n!=0:
                        loss_inter_mn_t = torch.max(THR_M - L2Distance(all_centers_tensor_t[m], all_centers_tensor_t[n]),
                                                  torch.FloatTensor([0]).cuda()).squeeze()
                        loss_inter_t += loss_inter_mn_t
                        c_inter=c_inter+1

            if c_inter!=0:
              loss_inter_t = loss_inter_t / c_inter

            #Calculate cluster-separating loss for source
            for m in range(4 - 1):
                for n in range(m + 1, 4):
                    c_m=torch.count_nonzero(all_centers_tensor_s[m])
                    c_n=torch.count_nonzero(all_centers_tensor_s[n])
                    if c_m!=0 and c_n!=0:
                          loss_inter_mn = torch.max(THR_M - L2Distance(all_centers_tensor_s[m], all_centers_tensor_s[n]),
                                                    torch.FloatTensor([0]).cuda()).squeeze()
                          loss_inter_s += loss_inter_mn
                          c_inter=c_inter+1
            if c_inter!=0:
                loss_inter_s = loss_inter_s / c_inter

            #Calculate inter-domain cluster discrepancy loss
            for l in range(4):
                loss_cd_l = L2Distance(all_centers_tensor_s[l], all_centers_tensor_t[l])
                loss_cd += loss_cd_l
            loss_cd=loss_cd/4



            centers_tensor = torch.from_numpy(centers)
            centers_tensor=centers_tensor.to('cuda')
            centers_tensor = centers_tensor.to(torch.float)

            #Calculate running combined loss
            for l in range(4):

                    tmp_centers_sl = tmp_centers_s[l]
                    tmp_centers_tl = tmp_centers_t[l]
                    if pesudo_label_nums[l]>0 and true_label_nums[l]>0:
                        m_centers_stl = (pesudo_label_nums[l] * tmp_centers_tl + true_label_nums[l] * tmp_centers_sl) / (pesudo_label_nums[l] + true_label_nums[l])
                    else:
                        m_centers_stl= (tmp_centers_tl+tmp_centers_sl)/2
                    delta_l = centers_tensor[l] - m_centers_stl
                    delta_l_np = delta_l.cpu().detach().numpy()
                    centers[l] = centers[l] - lr_c * delta_l_np
                    centers_tensor = torch.from_numpy(centers)
                    centers_tensor=centers_tensor.to('cuda')
                    centers_tensor = centers_tensor.to(torch.float)

                    loss_cl = L2Distance(m_centers_stl, centers_tensor[l])
                    loss_cmd += loss_cl
            loss_cmd=loss_cmd/4

            BETA_INTER= 0.1
            BETA_INTRA= 0.1
            BETA_CD= 0.5
            BETA_CMD= 0.1

            loss_intra_f_s=(loss_intra_s * BETA_INTRA) #cluster-compacting loss for source
            loss_inter_f_s=(loss_inter_s * BETA_INTER) #cluster-separating loss for source
            loss_intra_f_t=(loss_intra_t * BETA_INTRA)  #cluster-compacting loss for target
            loss_inter_f_t=(loss_inter_t * BETA_INTER) #cluster-separating loss for target
            loss_cd_f= (loss_cd * BETA_CD)  # inter-domain cluster discrepancy loss
            loss_cmd_f= (loss_cmd * BETA_CMD)  #running combined loss

            loss = loss_Cls + loss_intra_f_t + loss_inter_f_t +  loss_intra_f_s + loss_inter_f_s + loss_cd_f + loss_cmd_f

            reset_grad(opt_g, opt_c1, opt_c2)
            loss.backward()
            opt_g.step()
            opt_c1.step()

            '''
            print('Epoch: '+str(epoch+1))
            print('Iteration:  '+str(i+1))
            print('-------------------')
            print('cluster-separating loss (Target):')
            print(loss_inter_f_t)
            print('cluster-separating loss (Source):')
            print(loss_inter_f_s)
            print('cluster-compacting loss (Target):')
            print(loss_intra_f_t)
            print('cluster-compacting loss (Source):')
            print(loss_intra_f_s)
            print('inter-domain cluster discrepancy loss')
            print(loss_cd_f)
            print('running combined loss')
            print(loss_cmd_f)
            print('================')
            print('Classification Loss:')
            print(loss_Cls)
            '''


    ####Testing (Target Domain)####

    G.eval()
    C1.eval()
    C2.eval()
    acc=0
    test_loss=0
    all_loss=[]
    all_acc=[]
    all_preds=[]
    all_true_labels=[]

    for i in range(test_data_batch_t.shape[0]):

          test_data_tensor = torch.from_numpy(test_data_batch_t[i])
          test_data_tensor=test_data_tensor.to('cuda')
          test_data_tensor=test_data_tensor.float()

          test_RRinterval_tensor = torch.from_numpy(test_RRinterval_batch_t[i])
          test_RRinterval_tensor=test_RRinterval_tensor.to('cuda')
          test_RRinterval_tensor=test_RRinterval_tensor.float()
          test_prevR_tensor = torch.from_numpy(test_prevRR_batch_t[i])
          test_prevR_tensor=test_prevR_tensor.to('cuda')
          test_prevR_tensor=test_prevR_tensor.float()
          test_prev_eightRR_tensor = torch.from_numpy(test_prev_eightRR_batch_t[i])
          test_prev_eightRR_tensor=test_prev_eightRR_tensor.to('cuda')
          test_prev_eightRR_tensor=test_prev_eightRR_tensor.float()

          test_data_tensor=torch.swapaxes(test_data_tensor, 1, 2)

          test_labels_tensor = torch.from_numpy(test_label_batch_t[i])
          test_labels_tensor=torch.nn.functional.one_hot(test_labels_tensor, 4)
          test_labels_tensor=test_labels_tensor.to('cuda')

          feat_test = G(test_data_tensor)
          output_test_C1, output_test_prev_C1= C1(feat_test, test_RRinterval_tensor, test_prevR_tensor, test_prev_eightRR_tensor)
          output_test_C2, output_test_prev_C2= C2(feat_test, test_RRinterval_tensor, test_prevR_tensor, test_prev_eightRR_tensor)
          output_test= (output_test_C1 + output_test_C2)/2

          criterion = nn.CrossEntropyLoss()
          test_labels_tensor = test_labels_tensor.float()
          test_loss_curr = criterion(output_test, test_labels_tensor)
          all_loss.append(test_loss_curr)

          pred1 = output_test.data.max(1)[1]
          label1 = test_labels_tensor.data.max(1)[1]
          preds_np=pred1.detach().cpu().data.numpy()
          ture_labels_np=label1.detach().cpu().data.numpy()
          all_preds.append(preds_np)
          all_true_labels.append(ture_labels_np)
          np.concatenate
          correct1= pred1.eq(label1.data).cpu().sum()
          size=pred1.shape[0]
          curr_acc=correct1/size*100
          all_acc.append(curr_acc)

    #Calculate test Loss and test accuracy
    test_acc=0
    test_loss=0
    for i in range(len(all_loss)):
          test_loss = test_loss + all_loss[i]
          test_acc = test_acc + all_acc[i]
    avg_test_loss = test_loss / len(all_loss)
    avg_test_acc = test_acc / len(all_acc)

    return avg_test_acc, avg_test_loss, G, C1, C2

def reset_grad(opt_g, opt_c1, opt_c2):
      opt_g.zero_grad()
      opt_c1.zero_grad()
      opt_c2.zero_grad()

def discrepancy(out1, out2):
          return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

def L2Distance(x, y, dim=0, if_mean=True):
    #Measure distance between 2 vectors
    if if_mean:
      distance = torch.mean(torch.sqrt(torch.sum((x - y) ** 2, dim=dim)))
    else:
      distance = torch.sqrt(torch.sum((x - y) ** 2, dim=dim))

    return distance
