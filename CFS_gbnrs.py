import numpy as np

from load_data import ReadData as rd
from process_data import ProcessData
import cross_validation as cr
from compare import Comparison
import time
import HSNRS
import GBNRS
import open_learning_gbnrs

import sys
import os
# np.random.seed(0)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

open_learning = open_learning_gbnrs
def cfs(path, name):
    os_path = os.path.abspath(os.path.dirname(__file__))  # 获取当前py文件的父目录
    type = sys.getfilesystemencoding()
    # Read data
    all_data, all_label, data_with_label = rd(path, 1).read_data()
    allfeatures = list(range(all_data.shape[1]))

    k_folds = 10
    temp_train_index, temp_test_index = cr.cross_validation(all_data.shape[0], k_folds)

    init_data_ratio = [0.6]
    new_data_ratio = [0.4, 0.1]
    result1 = []
    for f in range(5):  # change the Gaussian kernel parameter
        for g in range(1):
            for l in range(2):

                para_init_data_ratio = init_data_ratio[g]
                para_new_data_ratio = new_data_ratio[l]
                print("set the data ratio:", para_init_data_ratio, para_new_data_ratio)

                # Simulate open world data, 0.6 is the ratio of initial data, 0.3 is the ratio of new class
                pro_data = ProcessData(all_data, all_label, data_with_label, para_init_data_ratio, para_new_data_ratio)
                init_data, init_label, init_class_label = pro_data.initial_class()
                new_data_list, new_label_list, new_class_label_list = pro_data.new_class()

                # all data feature selection
                original_select, original_remian, _ = GBNRS.get_attribute_reduction(all_data, all_label)
                print("original_select,original_remain", original_select, original_remian)

                start_time = time.time()
                select_att, remain_att, init_granular_balls = GBNRS.get_attribute_reduction(init_data, init_label)
                print("this turn select_att, remain_att:", select_att, remain_att)

                all_end_time = time.time() - start_time

                # start_time = time.time()
                # Open learning for new data sequences
                for i in range(len(new_data_list)):
                    # Class recognition on the open-set
                    each_turn_time = time.time()
                    class_ident = open_learning.ClassIdentification(init_granular_balls, new_data_list[i],
                                                                    init_class_label)
                    unknown_data, known_data = class_ident.class_identification(allfeatures)

                    # Clustering unknown data and build new granular balls
                    new_gb, cluster_data, cluster_label = open_learning.BallUpdate(
                        unknown_data).new_balls("dbscan", i)
                    ball_update = open_learning.BallUpdate(unknown_data).knowledge_update(init_granular_balls, new_gb)

                    # Update feature selection subspace
                    temp_data, temp_label = open_learning.data_merge(init_data, cluster_data, init_label, cluster_label)
                    if len(remain_att) == 0:
                        break
                    else:
                        select_att, remain_att = open_learning.SelectFeature(select_att, remain_att,
                                                                             ball_update).select_fea(
                            temp_data, temp_label)
                    end_each_turn_time = time.time()-each_turn_time
                    print("each_turn_time:", end_each_turn_time)

                # record a subset of historical reductions
                select_record = []
                # record the accuracy, f1_score and time
                temp_accuracy_comparison = []
                temp_f1_score = []
                temp_cost_time = []

                temp_accuracy_selected_comparison = []
                temp_f1_selected_score = []
                temp_selected_cost_time = []

                HNRS_accuracy_selected_comparison = []
                HNRS_f1_selected_score = []
                HNRS_selected_cost_time = []

                # select = [1,36,7,13,31,25,19,37,2,38,14,8,20,15,32,21,26,9,3,39,33,16,27,10,22,4,28,34,41,17,11,23,5,29]

                print("new_select_att, remain_att, all_end_time", select_att, remain_att, all_end_time)
                if len(select_att) == len(all_data[0]):
                    continue

                for j in range(10):
                    # print(j)
                    train_data = all_data[temp_train_index[j]]
                    train_label = all_label[temp_train_index[j]]

                    test_data = all_data[temp_test_index[j]]
                    test_label = all_label[temp_test_index[j]]
                    # initial data
                    temp_accuracy_list_original, temp_f1_score_list_original, temp_cost_time_original = Comparison(
                        train_data,
                        train_label,
                        test_data,
                        test_label).comparison()
                    temp_accuracy_comparison.append(temp_accuracy_list_original)
                    temp_f1_score.append(temp_f1_score_list_original)
                    temp_cost_time.append(temp_cost_time_original)
                    # selected data

                    HNRS_accuracy_list_select, HNRS_f1_score_list_select, HNRS_cost_time_select = Comparison(
                        train_data[:, original_select],
                        train_label, test_data[:, original_select], test_label).comparison()

                    temp_accuracy_list_select, temp_f1_score_list_select, temp_cost_time_select = Comparison(
                        train_data[:, select_att],
                        train_label, test_data[:, select_att], test_label).comparison()

                    temp_accuracy_selected_comparison.append(temp_accuracy_list_select)
                    temp_f1_selected_score.append(temp_f1_score_list_select)
                    temp_selected_cost_time.append(temp_cost_time_select)

                    HNRS_accuracy_selected_comparison.append(HNRS_accuracy_list_select)
                    HNRS_f1_selected_score.append(HNRS_f1_score_list_select)
                    HNRS_selected_cost_time.append(HNRS_cost_time_select)

                for k in range(5):
                    final_score_comparison = []
                    final_f1_score = []
                    final_cost_time = []

                    final_score_selected_comparison = []
                    final_f1_score_selected = []
                    final_cost_time_selected = []

                    HNRS_score_selected_comparison = []
                    HNRS_f1_score_selected = []
                    HNRS_cost_time_selected = []

                    for z in range(10):
                        final_score_comparison.append(temp_accuracy_comparison[z][k])
                        final_f1_score.append(temp_f1_score[z][k])
                        final_cost_time.append(temp_cost_time[z][k])

                        final_score_selected_comparison.append(temp_accuracy_selected_comparison[z][k])
                        final_f1_score_selected.append(temp_f1_selected_score[z][k])
                        final_cost_time_selected.append(temp_selected_cost_time[z][k])

                        HNRS_score_selected_comparison.append(HNRS_accuracy_selected_comparison[z][k])
                        HNRS_f1_score_selected.append(HNRS_f1_selected_score[z][k])
                        HNRS_cost_time_selected.append(HNRS_selected_cost_time[z][k])

                    score_test_comparison = sum(final_score_comparison) / 10
                    temp_standard_comparison = np.std(final_score_comparison)

                    print("ever_acc::", final_score_selected_comparison)

                    f1 = sum(final_f1_score) / 10
                    temp_f1_standard = np.std(final_f1_score)
                    average_time = sum(final_cost_time) / 10
                    print("ori acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$ ,time:$%0.9lf$" % (
                    score_test_comparison,
                    temp_standard_comparison,
                    f1,
                    temp_f1_standard,
                    average_time))
                    score_HNRS_selected_comparison = sum(HNRS_score_selected_comparison) / 10
                    HNRS_standard_selected_comparison = np.std(HNRS_score_selected_comparison)
                    HNRS_f1_selected = sum(HNRS_f1_score_selected) / 10
                    HNRS_f1_sel_standard = np.std(HNRS_f1_score_selected)
                    HNRS_average_time_selected = sum(HNRS_cost_time_selected) / 10
                    print("hnr acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$ ,time:$%0.9lf$" % (
                        score_HNRS_selected_comparison,
                        HNRS_standard_selected_comparison,
                        HNRS_f1_selected,
                        HNRS_f1_sel_standard,
                        HNRS_average_time_selected))

                    score_test_selected_comparison = sum(final_score_selected_comparison) / 10
                    temp_standard_selected_comparison = np.std(final_score_selected_comparison)
                    f1_selected = sum(final_f1_score_selected) / 10
                    temp_f1_sel_standard = np.std(final_f1_score_selected)
                    average_time_selected = sum(final_cost_time_selected) / 10
                    print("cfs acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$ ,time:$%0.9lf$" % (
                        score_test_selected_comparison,
                        temp_standard_selected_comparison,
                        f1_selected,
                        temp_f1_sel_standard,
                        average_time_selected))
                    result0 = {
                         # "s": para_s,
                        "init_r": init_data_ratio[g],
                        "new_r": new_data_ratio[l],
                        "k_classify": k,
                        "ever_acc": final_score_selected_comparison,
                        "o_a": score_test_comparison,
                        "o_s": temp_standard_comparison,
                        "o_f1": f1,
                        "o_fs": temp_f1_standard,
                        "h_a": score_HNRS_selected_comparison,
                        "h_s": HNRS_standard_selected_comparison,
                        "h_f1": HNRS_f1_selected,
                        "h_fs": HNRS_f1_sel_standard,
                        "cfs_a": score_test_selected_comparison,
                        "cfs_s": temp_standard_selected_comparison,
                        "cfs_f1": f1_selected,
                        "cfs_fs": temp_f1_sel_standard,

                    }
                    result1.append(result0)

                    print("-----")
                print("!!!!!!!new data ratio!!!!!!!!!!")
            print("--------------------each init data ratio---------------")
        print("--------------------each gaussian kernel parameter---------------")
    print("******************************************************************************")

    return result1