import numpy as np
import pandas as pd
import math
from random import sample
np.random.seed(0)
class ProcessData:

    def __init__(self, para_all_data, para_all_label, para_data_with_label, para_init_class_rate, para_new_class_rate):
        """
        para_init_class_rate: ratio of initial data
        para_new_class_rate: ratio of new class
        """
        self.all_data = para_all_data
        self.all_label = para_all_label
        self.data_with_label = para_data_with_label
        self.init_class_rate = para_init_class_rate
        self.new_class_rate = para_new_class_rate
        self.label_list = np.unique(self.all_label)
        self.remain_label = []

    def initial_class(self):
        """
        initial the number of classes
        """
        # label_list = set(self.all_label) # list
        init_class_num = math.ceil(len(self.label_list)*self.init_class_rate) # int
        # init_class_label = sample(list(self.label_list), init_class_num) # 随机抽取一定类别 arr
        init_class_label = self.label_list[:init_class_num] # 指定抽取
        self.remain_label = [x for x in self.label_list if x not in init_class_label]
        init_data = []
        init_label = []
        for i in range(init_class_num):
            temp_index = np.where(self.all_label == init_class_label[i])[0][:]
            if i == 0:
                init_data = self.all_data[temp_index]  # arr
                init_label = self.all_label[temp_index]
            else:
                init_data = np.vstack((init_data, self.all_data[temp_index]))
                init_label = np.vstack((init_label,  self.all_label[temp_index]))
        return init_data, init_label, init_class_label

    def new_class(self):
        """
        simulate the addition of new classes in the open world
        """
        new_data_list = []
        new_label_list = []
        new_class_label_list = []

        while len(self.remain_label) != 0:
            new_class_num = math.ceil(len(self.label_list)*self.new_class_rate)
            if new_class_num > len(self.remain_label):
                new_class_num = len(self.remain_label)
            single_class_label = sample(list(self.remain_label), new_class_num)
            new_class_label_list.append(single_class_label)
            self.remain_label = [x for x in self.remain_label if x not in single_class_label]
            single_new_data = []
            signal_new_label = []

            for i in range(new_class_num):
                temp_index = np.where(self.all_label==single_class_label[i])[0][:]
                if i == 0:
                    single_new_data = self.all_data[temp_index]  # arr
                    signal_new_label = self.all_label[temp_index]
                else:
                    single_new_data = np.vstack((single_new_data, self.all_data[temp_index]))
                    signal_new_label = np.vstack((signal_new_label, self.all_label[temp_index]))
            new_data_list.append(single_new_data)
            new_label_list.append(signal_new_label)

        return new_data_list, new_label_list, new_class_label_list







