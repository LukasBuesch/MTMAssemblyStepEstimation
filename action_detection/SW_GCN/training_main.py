import os.path

import torch
import numpy as np
import xlsxwriter
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json

from action_labels import action_labels
# import sophisticated optimizer (loaded from GitHub)
from Ranger.ranger.ranger import Ranger
from network_model import *
import action_labels as al


class TrainModel:
    def __init__(self, train_data, val_data, train_file_name):
        """
        create instacne for training to free up memory space later
        :param train_data:
        :param val_data:
        """

        self.train_data = train_data
        self.val_data = val_data
        self.train_file_name = train_file_name

    def start_training(self):
        # import settings
        with open('config_train.json') as json_file:
            settings = json.load(json_file)

        # set gpu device
        gpu_device = start_cuda()

        # data loader
        train_set_loader, val_set_loader = get_loader(settings["batch_size"], self.train_data,
                                                      self.val_data)

        # load list of action class names
        classes_names = al.action_labels

        # list of class appearances
        label_appearance = np.zeros(len(al.action_labels))
        for batch_idx, (data, label) in enumerate(train_set_loader):
            for x in classes_names:
                # count appearance of action label in train data set
                label_appearance[classes_names[x]] += np.count_nonzero(label == classes_names[x])

        # normalize to largest number of appearances and calc reciprocal value -> set higher value to labels with lower
        # number of appearances to prevent only optimizing for labels with large appearance
        label_weight = np.zeros(len(label_appearance))
        for i in range(len(label_appearance)):
            if label_appearance[i] == 0:
                label_weight[i] = 0
            else:
                label_weight[i] = ((1 / label_appearance[i]) * max(label_appearance))
        #
        label_weight = torch.from_numpy(np.array(label_weight, dtype='float32'))
        label_weight = label_weight.cuda()

        # worksheet to store progress in
        workbook = xlsxwriter.Workbook(settings["workbook"] + "_" + self.train_file_name + ".xlsx")
        worksheet = workbook.add_worksheet()

        # train model
        torch.cuda.empty_cache()
        model = Model(settings["num_coord_per_joint"], len(classes_names), classes_names, gpu_device)

        # load existing weights
        if settings["train_on_exist"] != "":
            print("load existing weights: {}".format(settings["train_on_exist"]))
            model.load_state_dict(torch.load(settings["train_on_exist"]))

        model = model.cuda()
        train_epochs(model, gpu_device, worksheet, label_weight, train_set_loader, val_set_loader,
                     settings["num_epochs"],
                     settings["train_model_store"] + "_on_" + self.train_file_name + "_with_3_fcl_" + ".torch")

        # close workbook
        workbook.close()


def train(model, device, train_set_loader, optimizer, epoch, worksheet, label_weight, logging_interval=100):
    """
    main function for training network
    :param label_weight:
    :param model:
    :param device:
    :param train_set_loader:
    :param optimizer:
    :param epoch:
    :param worksheet:
    :param logging_interval:
    :return:
    """
    # set model to train mode
    model.train()

    # to calculate progress and performance
    acc_sum = 0
    acc_loss = 0
    counter_train = 0

    # iterate over whole training data
    for batch_idx, train_set in enumerate(train_set_loader):

        counter_train += 1

        # set data and label from train_set tuple
        data = train_set[0]
        label = train_set[1]
        output = torch.Tensor(np.empty((len(data), len(action_labels))))

        # push data to cuda
        data = data.to(device)
        label = label.to(device)
        output = output.to(device)

        # train network on spatio temporal graph but not on training set length -> iterate over first dimension of data
        for i in range(len(data)):
            output[i] = model(data[i])
        # start optimizing
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # ToDO: max norm - put to config if necessary
        loss = F.cross_entropy(output, label, weight=label_weight)
        loss.backward()
        optimizer.step()

        # safe in predefined log intervals
        if batch_idx % logging_interval == 0:
            guess = output.max(1, keepdim=True)[1]  # get the index of the label with max probability

            # counts how many correct estimates in training set
            correct = guess.eq(label.view_as(guess)).float().mean().item()  # compare to label mid of window
            acc_sum += correct
            acc_loss += loss.item()

            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_set_loader.dataset), 100. * batch_idx / len(train_set_loader),
                loss.item(), 100. * correct))

    # end of training epoch - post processing
    acc_sum = (acc_sum * 100) / counter_train
    acc_loss /= counter_train
    print("")
    print('Train epoch accuracy : {:.2f}%'.format(acc_sum))
    worksheet.write(epoch, 4, acc_sum)
    worksheet.write(epoch, 3, acc_loss)


def train_epochs(model, device, worksheet, label_weight, train_set_loader, val_set_loader, num_epochs,
                 train_model_store):
    """
    call train methods for number of epochs
    :param train_model_store:
    :param model:
    :param device:
    :param worksheet:
    :param label_weight:
    :param train_set_loader:
    :param val_set_loader:
    :param num_epochs:
    :return:
    """
    # best accuracy reached
    best_acc = 0

    # iterate over epochs
    for epoch in range(num_epochs):
        # set optimizer to be Ranger (sophisticated optimizer loaded from GitHub)
        optimizer = Ranger(model.parameters(), lr=0.005, weight_decay=0, betas=(0.95, 0.999))

        # start training
        train(model, device, train_set_loader, optimizer, epoch, worksheet, label_weight, logging_interval=1)

        # validate training of epoch
        best_acc = validate(model, device, val_set_loader, epoch, best_acc, train_model_store, worksheet)


def validate(model, device, val_set_loader, epoch, best_acc, train_model_store, worksheet):
    """
    validate model
    :param model:
    :param device:
    :param val_set_loader:
    :param epoch:
    :param best_acc:
    :param train_model_store:
    :return:
    """
    # set model to evaluation mode
    model.eval()

    # to calculate progress and performance
    test_loss = 0
    # calculate correctness per label to get a more meaningful value with less influence of label occurrence
    correct = 0
    correct_per_label = [0]*len(action_labels)
    label_count = [0]*len(action_labels)

    # disable gradient calculation for performance reasons
    with torch.no_grad():

        # iterate over validation set
        for data, target in val_set_loader:
            # data, target = data, target
            output = torch.Tensor(np.empty((len(data), len(action_labels))))

            # push tensor to cuda
            data = data.to(device)
            target = target.to(device)
            output = output.to(device)

            # validate network on spatio temporal graph but not on training set length
            # -> iterate over first dimension of data
            for i in range(len(data)):
                # run model
                output[i] = model(data[i])

            # Note: with `reduce=True`, I'm not sure what would happen with a final batch size
            # that would be smaller than regular previous batch sizes. For now it works.
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            guess = output.max(1, keepdim=True)[1]  # get the index of the label with max probability
            for i in range(len(guess)):
                a = guess[i]
                b = target.view_as(guess)[i]
                label_count[target.view_as(guess)[i]] += 1
                if guess[i] == target.view_as(guess)[i]:
                    correct_per_label[target.view_as(guess)[i]] += 1
            correct += guess.eq(target.view_as(guess)).sum().item()

    # evaluate validation
    test_loss /= len(val_set_loader.dataset)

    val_acc = 100. * correct / len(val_set_loader.dataset)
    # calculate median accuracy over all labels
    val_acc_med = 0
    val_acc_per_label = []
    for i in range(len(correct_per_label)):
        label_acc = correct_per_label[i]/label_count[i] * 100
        val_acc_med += label_acc
        val_acc_per_label.append(label_acc)
    val_acc_med /= len(correct_per_label)

    print("")
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Accuracy mean over labels: {:.2f}%'.format(
        test_loss, correct, len(val_set_loader.dataset), val_acc, val_acc_med))
    print("")

    # store model if best accuracy reached so far
    if best_acc < val_acc_med:
        torch.save(model.state_dict(), train_model_store)
        best_acc = val_acc_med

    # post processing
    print('max accuracy : {:.2f}%\n\n'.format(best_acc))
    worksheet.write(epoch, 0, val_acc)
    worksheet.write(epoch, 1, val_acc_med)
    worksheet.write(epoch, 2, test_loss)
    # store accuracy per label
    for i in range(len(val_acc_per_label)):
        worksheet.write(epoch, 6+i, val_acc_per_label[i])
    return best_acc


def get_loader(batch_size, file_train, file_val):
    """
    generate data loader for loaded .pt file
    :return:
    """

    training_set = torch.load(file_train)
    testing_set = torch.load(file_val)

    train_set_loader = torch.utils.data.DataLoader(
        dataset=training_set,
        batch_size=batch_size,
        shuffle=True)

    val_set_loader = torch.utils.data.DataLoader(
        dataset=testing_set,
        batch_size=batch_size,
        shuffle=False)

    return train_set_loader, val_set_loader





if __name__ == '__main__':
    # load training data:
    # path to data
    path_to_folder_1 = ""  # ToDo: insert path
    path_to_folder_2 = ""  # ToDo: insert path
    path_to_folder = [path_to_folder_1, path_to_folder_2]

    for path in path_to_folder:

        train_name = "InHARD_train"
        val_name = "InHARD_val"

        # iterate over folders
        a = os.listdir(path)
        for folder in os.listdir(path):

            # load train and val data
            folder_path = path + folder
            train_data = None
            val_data = None
            train_file_name = None
            for filename in os.listdir(folder_path):
                if filename[0:len(train_name)] == train_name:
                    # train data
                    train_data = folder_path + "\\" + filename
                    train_file_name = filename[:-3]  # without ending ".pt"
                elif filename[0:len(val_name)] == val_name:
                    # validation data
                    val_data = folder_path + "\\" + filename
            if train_data is None or val_data is None:
                continue

            # do training
            print("Start training for " + train_data)
            model_training = TrainModel(train_data, val_data, train_file_name)
            model_training.start_training()

            # delete instance afterwards to free memory
            del model_training
