import HSNRS
from sklearn import cluster
import numpy as np
from scipy.spatial import distance


class ClassIdentification:
    """
    identify the new data
    """
    def __init__(self, para_hypersphere, data, para_s, class_num, class_list):
        self.hypersphere = para_hypersphere
        self.data = data
        self.s = para_s
        self.class_num = class_num
        self.class_list = class_list

    def get_matrix(self, para_a, para_b):
        """
        calculate the Gaussian kernel distance between two vectors
        """
        dist_sq = distance.cdist(para_a, para_b)
        cur_sim_vec = np.exp(-np.square(dist_sq) / (2.0 * self.s * self.s))
        return cur_sim_vec

    def get_distance(self, att):
        """
        calculate the distance between the new data and the center of each hypersphere
        return:
        """
        distance_list = []
        for i in range(self.class_num):
            a = self.get_matrix(self.data[:, att], self.hypersphere["sv"][i][:, att])
            term_2 = -2 * np.dot(self.get_matrix(self.data[:, att], self.hypersphere["sv"][i][:, att]), self.hypersphere["alpha"][i])
            cur_sim_vec = self.get_matrix(self.hypersphere["sv"][i][:, att], self.hypersphere["sv"][i][:, att])
            term_3 = np.dot(np.dot(self.hypersphere["alpha"][i].T, cur_sim_vec), self.hypersphere["alpha"][i])
            a = 1 + term_2 + term_3
            temp_distance = np.sqrt(1 + term_2 + term_3)
            distance_list.append(temp_distance)
        return distance_list

    def class_identification(self, att):
        """

        """
        dis_list = self.get_distance(att)
        data_index = list(range(self.data.shape[0]))
        unknown_data_index = data_index
        known_data = []
        for i in range(self.class_num):
            compare_distance = np.where(dis_list[i] <= self.hypersphere["radius"][i])[0]  # 超球是否会存在空间重叠情况？
            # find the unknown data
            if compare_distance.all() not in unknown_data_index:
                continue
            if len(compare_distance) == 0:
                unknown_data_index = unknown_data_index
            else:
                known_data_label = np.tile(np.array(self.class_list[i]), len(compare_distance)).reshape((len(compare_distance)), 1)
                known_data = np.hstack((self.data[compare_distance, :], known_data_label))
                unknown_data_index = list(set(unknown_data_index) - set(compare_distance))
        if len(unknown_data_index) == 0:
            unknown_data_index = list(range(self.data.shape[0]))

        unknown_data = self.data[unknown_data_index, :]

        return unknown_data, known_data

class Cluster:

    def __init__(self, para_data):
        self.data = para_data

    def k_means(self, para_k):
        kcluster = cluster.KMeans(para_k).fit(self.data)
        temp_labels = list(set(list(kcluster.labels_)))
        labels_list = []
        for i in range(len(temp_labels)):
            labels_list.append(temp_labels[i]+100)  # 给伪标签一个很大的数值，避免其与已有标签混淆
        return kcluster.labels_, labels_list

    def dbscan(self, frequency):
        db = cluster.DBSCAN(eps=0.2, min_samples=10).fit(self.data)
        temp_labels = db.labels_
        index = np.where(temp_labels != -1)[0]
        temp_labels = np.array(temp_labels)[index]
        normal_data = self.data[index]
        labels = []
        n_clusters_ = len(set(temp_labels)) - (1 if -1 in temp_labels else 0)
        n_noise_ = list(labels).count(-1)
        for i in range(len(temp_labels)):
            labels.append(temp_labels[i]+frequency+100)
        labels_num = list(set(labels))
        labels = np.array(labels).reshape(len(labels), 1)

        return normal_data, labels, labels_num

class BallUpdate:

    def __init__(self, para_data):
        self.data = para_data

    def new_balls(self, para_cluster_method, para_s, frequency):

        temp_cluster = Cluster(self.data)
        if para_cluster_method == "kmeans":
            data_label, label_list = temp_cluster.k_means(3)
        if para_cluster_method == "dbscan":
            self.data, data_label, label_list = temp_cluster.dbscan(frequency)
        hypersphere = HSNRS.HyperSphere(self.data, data_label, len(label_list),
                                        label_list, para_s).init_hyper_sphere()
        return hypersphere, self.data, data_label, label_list

    def knowledge_update(self, init_hypersphere, new_hypersphere):
        """
        merge all granular balls
        """
        for i in range(len(new_hypersphere["radius"])):
            init_hypersphere["radius"].append(new_hypersphere["radius"][i])
            init_hypersphere["center"].append(new_hypersphere["center"][i])
            init_hypersphere["sv"].append(new_hypersphere["sv"][i])
            init_hypersphere["sv_index"].append(new_hypersphere["sv_index"][i])
            init_hypersphere["alpha"].append(new_hypersphere["alpha"][i])

        return init_hypersphere


def data_merge(para_init_data, para_new_data, para_init_label, para_new_label):
    data = np.vstack((para_init_data, para_new_data))
    label = np.vstack((para_init_label, para_new_label))
    return data, label


class SelectFeature:
    """
    select important features from candidate features
    """
    def __init__(self, para_select_att, para_remain_att, para_hypersphere):
        self.select_att = para_select_att
        self.remain_att = para_remain_att
        self.hypersphere = para_hypersphere

    def select_feature(self, data, label, para_s):
        i = 0
        class_list = list(set(list(label[:, 0])))
        class_num = len(class_list)
        attributes_reduction = list(range(data.shape[1]))
        hyper_sphere = HSNRS.HyperSphere(data, label, class_num, class_list, para_s) # Just initialize an object
        while len(self.remain_att) != 0:
            if i+1 < len(self.remain_att):
                # Delete attributes one by one to judge the purity of balls
                remove_i = attributes_reduction.index(self.remain_att[i])
                attributes_reduction.remove(self.remain_att[i])
                distance_list = hyper_sphere.get_purity(remove_i, self.hypersphere)
                num_mis = 0
                for j in range(class_num):
                    num_mis += len(distance_list[j])
                # print(num_mis / data.shape[0])  # or num_mis/data.shape[0] < 0.01
                #if num_mis == 0:
                if num_mis/data.shape[0] == 0:
                    hyper_sphere = HSNRS.HyperSphere(data[:, attributes_reduction], label, class_num, class_list,
                                                     para_s)
                    self.hypersphere = hyper_sphere.init_hyper_sphere()
                else:
                    attributes_reduction.append(self.remain_att[i])
                    attributes_reduction.sort()
            else:
                break
            i += 1
        remain_attributes = list(set(list(range(data.shape[1]))) - set(attributes_reduction))

        return attributes_reduction, remain_attributes





