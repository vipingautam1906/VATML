import random
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.transforms.transforms as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from help_code_demo import ToTensor, F1, FB, Sensitivity, Specificity, BAC, ACC, PPV, NPV
from models.model_1 import SysNet
import copy
from sklearn.metrics import confusion_matrix

seed = 222
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Augmentations
def flip_and_noise(data):
    sample = data['IEGM_seg']
    #if random.random() > 0.5:
    #    sample = -sample
    
    if random.random() > 0.5:
        peak_max = sample.max() * 0.05
        factor = random.random()
        noise = np.random.normal(0, factor * peak_max, (len(sample), 1))
        sample = sample + noise
    
    data['IEGM_seg'] = sample
    return data


def main(run, max_fb):

    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    T_runs = args.runs
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    print(f"Run:          [{run}/{T_runs}]")

    # Instantiating NN
    net = SysNet()
    net.train()
    net = net.float().to(device)
    

    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor(), flip_and_noise]))

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=4)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay = 1e-4)
    epoch_num = EPOCH

    max_test_fb = max_fb
    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = []

    train_f1_lst, train_fb_lst, train_precision_lst, train_recall_lst = [], [], [], []
    test_f1_lst, test_fb_lst, test_precision_lst, test_recall_lst = [], [], [], []


    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)
        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0

        true_label, pred_label = [], []

        for j, data in enumerate(trainloader, 0):
            inputs, labels = data['IEGM_seg'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
            accuracy += correct / BATCH_SIZE
            correct = 0.0

            running_loss += loss.item()
            i += 1

            true_label.extend(labels.detach().cpu().numpy())
            pred_label.extend(predicted.detach().cpu().numpy())

        C = confusion_matrix(true_label, pred_label)
        acc = (C[0][0] + C[1][1]) / (C[0][0] + C[0][1] + C[1][0] + C[1][1])
        precision = C[1][1] / (C[1][1] + C[0][1])
        recall = C[1][1] / (C[1][1] + C[1][0])
        f1_score = (2 * precision * recall) / (precision + recall)
        fb_score = (1 + 2 ** 2) * (precision * recall) / ((2 ** 2) * precision + recall)


        print('[Epoch %d] \nTrain|| Loss: %.5f' %
              (epoch + 1, running_loss / len(trainloader.dataset)))
        print("        Precision: {:.5f}, Recall: {:.5f}, F1_score: {:.5f}, Fb_score: {:.5f}".format(precision, recall, f1_score,
                                                                                           fb_score))

        Train_loss.append(running_loss / len(trainloader.dataset))
        Train_acc.append(acc.item())
        train_f1_lst.append(f1_score.item())
        train_fb_lst.append(fb_score.item())
        train_precision_lst.append(precision.item())
        train_recall_lst.append(recall.item())

        running_loss = 0.0
        accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0
        true_label_test, pred_label_test = [], []

        net.eval()
        with torch.no_grad():
            for data_test in testloader:
                IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
                IEGM_test = IEGM_test.float().to(device)
                labels_test = labels_test.to(device)

                outputs_test = net(IEGM_test)

                _, predicted_test = torch.max(outputs_test.data, 1)
                total += labels_test.size(0)
                correct += (predicted_test == labels_test).sum()

                loss_test = criterion(outputs_test, labels_test)

                running_loss_test += loss_test.item()
                true_label_test.extend(labels_test.detach().cpu().numpy())
                pred_label_test.extend(predicted_test.detach().cpu().numpy())
                i += 1
        

        test_C = confusion_matrix(true_label_test, pred_label_test)
        test_acc = (test_C[0][0] + test_C[1][1]) / (test_C[0][0] + test_C[0][1] + test_C[1][0] + test_C[1][1])
        test_precision = test_C[1][1] / (test_C[1][1] + test_C[0][1])
        test_recall = test_C[1][1] / (test_C[1][1] + test_C[1][0])
        test_f1_score = (2 * test_precision * test_recall) / (test_precision + test_recall)
        test_fb_score = (1 + 2 ** 2) * (test_precision * test_recall) / ((2 ** 2) * test_precision + test_recall)

        print(f'Test || Loss: {(running_loss_test / len(testloader.dataset)):.5f}')
        print(f'        Precision: {test_precision:.5f} Recall: {test_recall:.5f} F1_score: {test_f1_score:.5f} Fb_score: {test_fb_score:.5f} ')

        Test_loss.append(running_loss_test / len(testloader.dataset))
        Test_acc.append(test_acc.item())
        test_f1_lst.append(test_f1_score.item())
        test_fb_lst.append(test_fb_score.item())
        test_precision_lst.append(test_precision.item())
        test_recall_lst.append(test_recall.item())

        if test_fb_score >= max_test_fb:
            max_test_fb = max(max_test_fb, test_fb_score)
            torch.save(net, f'./saved_models/SysNet_best.pkl')
            torch.save(net.state_dict(), f'./saved_models/SysNet_state_dict_best.pkl')

    file = open('./saved_models/loss_acc.txt', 'w')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Test_acc))
    file.write('\n\n')
    return max_test_fb

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--runs', type=int, default=1)
    argparser.add_argument('--path_data', type=str, default='./Training_data/tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')

    args = argparser.parse_args()
    device = torch.device("cuda:" + str(args.cuda))
    print("device is --------------", device)

    max_fb = 0.0
    for run in range(1,args.runs+1):
        current_max_fb = main(run, max_fb)
        if current_max_fb >= max_fb:
            max_fb = max(current_max_fb,max_fb)
        print("Current F_beta score\t", max_fb)
    print('Training finished..Best Model Saved!!')
    print('F_beta Score acheived\t',max_fb)

