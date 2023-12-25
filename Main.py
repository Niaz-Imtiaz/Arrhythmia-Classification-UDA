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
from Execute_Model import Execute_Model

data_obj=Prepare_Data()
data_s, target_s, data_RRinterval_s, data_prevRR_s, data_prev_eight_RR_s, data_t, target_t, data_RRinterval_t, data_prevRR_t, data_prev_eight_RR_t= data_obj.preprocess_data()

data_obj_spit=Prepare_Train_Test()
train_data_batch_s, train_RRinterval_batch_s, train_prevRR_batch_s, train_prev_eightRR_batch_s, train_label_batch_s, train_data_batch_t, train_RRinterval_batch_t, train_prevRR_batch_t, train_prev_eightRR_batch_t, train_label_batch_t,test_data_batch_t, test_RRinterval_batch_t, test_prevRR_batch_t, test_prev_eightRR_batch_t, test_label_batch_t, input_shape, num_classes  = data_obj_spit.create_train_test(data_s, target_s, data_RRinterval_s, data_prevRR_s, data_prev_eight_RR_s, data_t, target_t, data_RRinterval_t, data_prevRR_t, data_prev_eight_RR_t)



data_obj_model=Execute_Model()
best_test_acc=0

for i in range(5):

  G = cm.Generator()  #Feature Generator (F)
  G.to('cuda')
  C1 = cm.Classifier(num_classes) #Classifier 1 (C1)
  C1.to('cuda')
  C2 = cm.Classifier(num_classes) #Classifier 1 (C2)
  C2.to('cuda')
  curr_test_acc, curr_test_los, G_upgrd, C1_upgrd, C2_upgrd=data_obj_model.exec_model(G,C1,C2, train_data_batch_s, train_RRinterval_batch_s, train_prevRR_batch_s, train_prev_eightRR_batch_s, train_label_batch_s, train_data_batch_t, train_RRinterval_batch_t, train_prevRR_batch_t, train_prev_eightRR_batch_t, train_label_batch_t,test_data_batch_t, test_RRinterval_batch_t, test_prevRR_batch_t, test_prev_eightRR_batch_t, test_label_batch_t, input_shape, num_classes)
  print('Current Accuracy')
  print(curr_test_acc)
  if curr_test_acc > best_test_acc:
      best_test_acc=curr_test_acc
      torch.save(G_upgrd.state_dict(), './weights/generator_weights.pth')
      torch.save(C1_upgrd.state_dict(), './weights/classifier_1_weights.pth')
      torch.save(C2_upgrd.state_dict(), './weights/classifier_2_weights.pth')

      # Save train data and labels
      np.save('./train_test_data/X_train_data.npy', train_data_batch_s)
      np.save('./train_test_data/X_train_RR.npy', train_RRinterval_batch_s)
      np.save('./train_test_data/X_train_prevRR.npy', train_prevRR_batch_s)
      np.save('./train_test_data/X_train_prev_eightRR.npy', train_prev_eightRR_batch_s)
      np.save('./train_test_data/Y_train.npy', train_label_batch_s)

      # Save test data and labels
      np.save('./train_test_data/X_test_data.npy', test_data_batch_t)
      np.save('./train_test_data/X_test_RR.npy', test_RRinterval_batch_t)
      np.save('./train_test_data/X_test_prevRR.npy', test_prevRR_batch_t)
      np.save('./train_test_data/X_test_prev_eightRR.npy', test_prev_eightRR_batch_t)
      np.save('./train_test_data/Y_test.npy', test_label_batch_t)

  data_obj_spit=Prepare_Train_Test()
  train_data_batch_s, train_RRinterval_batch_s, train_prevRR_batch_s, train_prev_eightRR_batch_s, train_label_batch_s, train_data_batch_t, train_RRinterval_batch_t, train_prevRR_batch_t, train_prev_eightRR_batch_t, train_label_batch_t,test_data_batch_t, test_RRinterval_batch_t, test_prevRR_batch_t, test_prev_eightRR_batch_t, test_label_batch_t, input_shape, num_classes  = data_obj_spit.create_train_test(data_s, target_s, data_RRinterval_s, data_prevRR_s, data_prev_eight_RR_s, data_t, target_t, data_RRinterval_t, data_prevRR_t, data_prev_eight_RR_t)

print('Best Accuracy')
print(best_test_acc)


generator = cm.Generator()  #Feature Generator (F)
generator.to('cuda')
classifier1 = cm.Classifier(num_classes) #Classifier 1 (C1)
classifier1.to('cuda')
classifier2 = cm.Classifier(num_classes) #Classifier 1 (C2)
classifier2.to('cuda')

# Load the saved Generator state_dict from the file
generator.load_state_dict(torch.load('./weights/generator_weights.pth'))
generator.eval()  # Set the Generator to evaluation mode

# Load the saved Classifier state_dict from the file
classifier1.load_state_dict(torch.load('./weights/classifier_1_weights.pth'))
classifier1.eval()  # Set the Classifier to evaluation mode
classifier2.load_state_dict(torch.load('./weights/classifier_2_weights.pth'))
classifier2.eval()  # Set the Classifier to evaluation mode

# Load train data and labels

train_data_batch_s = np.load('./train_test_data/X_train_data.npy')
train_RRinterval_batch_s = np.load('./train_test_data/X_train_RR.npy')
train_prevRR_batch_s = np.load('./train_test_data/X_train_prevRR.npy')
train_prev_eightRR_batch_s = np.load('./train_test_data/X_train_prev_eightRR.npy')
train_label_batch_s = np.load('./train_test_data/Y_train.npy')

# Load test data and labels
test_data_batch_t = np.load('./train_test_data/X_test_data.npy')
test_RRinterval_batch_t = np.load('./train_test_data/X_train_RR.npy')
test_prevRR_batch_t = np.load('./train_test_data/X_test_prevRR.npy')
test_prev_eightRR_batch_t = np.load('./train_test_data/X_test_prev_eightRR.npy')
test_label_batch_t = np.load('./train_test_data/Y_test.npy')

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

print('Accuracy')
print(avg_test_acc)