import GBNRS
from sklearn import cluster
import numpy as np
from scipy.spatial import distance
import HSNRS
import statistics

class ClassIdentification:
    """
    identify the new data
    """
    def __init__(self, para_gb_list, data, class_list):
        self.gb_list = para_gb_list
        self.data = data
        self.class_list = class_list
        self.gb_num = len(self.gb_list.granular_balls)

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
        for i in range(self.gb_num):
            temp_distance = np.linalg.norm(self.data[:, att] - self.gb_list.granular_balls[i].center[att], axis=1, ord=2)
            distance_list.append(np.array(temp_distance.reshape(self.data.shape[0], 1)))
        return distance_list

    def class_identification(self, att):
        """

        """
        dis_list = self.get_distance(att)
        data_index = list(range(self.data.shape[0]))
        unknown_data_index = data_index
        known_data = []
        for i in range(self.gb_num):
            compare_distance = np.where(dis_list[i] <= self.gb_list.granular_balls[i].radius)[0]  #
            # find the unknown data
            if compare_distance.all() not in unknown_data_index:
                continue
            if len(compare_distance) == 0:
                unknown_data_index = unknown_data_index
            else:
                known_data_label = np.tile(np.array(self.gb_list.granular_balls[i].label), len(compare_distance)).reshape((len(compare_distance)), 1)
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
        db = cluster.DBSCAN(eps=0.3, min_samples=10).fit(self.data)
        temp_labels = db.labels_
        index = np.where(temp_labels != -1)[0]
        temp_labels = np.array(temp_labels)[index]
        normal_data = self.data[index]
        labels = []
        n_clusters_ = len(set(temp_labels)) - (1 if -1 in temp_labels else 0)
        n_noise_ = list(labels).count(-1)

        if n_clusters_ == 0:
            index = np.where(temp_labels == -1)[0]
            normal_data = self.data[index]
            labels = np.array(temp_labels)[index]
            multiplier = frequency + 100
            labels = [x * multiplier for x in labels]
            labels_num = list(set(labels))
            labels = np.array(labels).reshape(len(labels), 1)


        if n_noise_ == 0:
            normal_data = self.data
            labels = np.full(self.data.shape[0], frequency + 100)
            labels_num = list(set(labels))
            labels = np.array(labels).reshape(len(labels), 1)


        else:
            for i in range(len(temp_labels)):
                labels.append(frequency + 100)
            labels_num = list(set(labels))
            labels = np.array(labels).reshape(len(labels), 1)


        return normal_data, labels, labels_num

class BallUpdate:

    def __init__(self, para_data):
        self.data = para_data

    def new_balls(self, para_cluster_method, frequency):

        temp_cluster = Cluster(self.data)
        if para_cluster_method == "kmeans":
            data_label, label_list = temp_cluster.k_means(3)
        if para_cluster_method == "dbscan":
            self.data, data_label, label_list = temp_cluster.dbscan(frequency)

        if len(data_label) > 1 and len(label_list) > 0:
            # print("gbc data_len", len(data_label))
            GB, _ = GBNRS.create_gblist(self.data, data_label)
            return GB, self.data, data_label
        else:
            return False, False, False


    def knowledge_update(self, init_GB_list, new_GB_list):
        """
        merge all granular balls
        """
        if not new_GB_list:
            init_GB_list = init_GB_list
        else:
            temp_num = len(new_GB_list.granular_balls)
            # Combined data
            init_GB_list.data = np.vstack((init_GB_list.data[:, :-1], new_GB_list.data[:, :-1]))
            num, dim = init_GB_list.data[:, :-1].shape
            index = np.array(range(num)).reshape(num, 1)  # column of index
            init_GB_list.data = np.hstack((init_GB_list.data, index))
            # Combined granular-balls
            for i in range(temp_num):
                init_GB_list.granular_balls.append(new_GB_list.granular_balls[i])

        # Return all
        return init_GB_list


def data_merge(para_init_data, para_new_data, para_init_label, para_new_label):

    if type(para_new_data) == bool and not para_new_data:
        data = para_init_data
        label = para_init_label
    else:
        data = np.vstack((para_init_data, para_new_data))
        label = np.vstack((para_init_label, para_new_label))

    return data, label


class SelectFeature:
    """
    select important features from candidate features
    """
    def __init__(self, para_select_att, para_remain_att, para_gb_list):
        self.select_att = para_select_att
        self.remain_att = para_remain_att
        self.gb_list = para_gb_list


    def select_feature(self, data, label, para_s):
        i = 0
        class_list = list(set(list(label[:, 0])))
        class_num = len(class_list)
        attributes_reduction = list(range(data.shape[1]))
        hyper_sphere = HSNRS.HyperSphere(data, label, class_num, class_list, para_s) #
        while len(self.remain_att) != 0:
            if i+1 < len(self.remain_att):
                # Delete attributes one by one to judge the purity of balls
                remove_i = attributes_reduction.index(self.remain_att[i])
                attributes_reduction.remove(self.remain_att[i])
                distance_list = hyper_sphere.get_purity(remove_i, self.hypersphere)
                num_mis = 0
                for j in range(class_num):
                    num_mis += len(distance_list[j])
                #print(num_mis / data.shape[0])  # or num_mis/data.shape[0] < 0.01
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

    def select_fea(self, data, label):
        i = 0
        class_list = list(set(list(label[:, 0])))
        data = np.hstack((data, label))
        num, dim = data[:, :-1].shape
        index = np.array(range(num)).reshape(num, 1)  # column of index
        data = np.hstack((data, index))
        attributes_reduction = list(range(data.shape[1]-2))
        while len(self.remain_att) != 0:
            if i + 1 < len(self.remain_att):
                # Delete attributes one by one to judge the purity of balls
                # remove_i = attributes_reduction.index(self.remain_att[i])
                remove_i = self.remain_att[i]
                #print("the ith feature:", remove_i)
                attributes_reduction.remove(remove_i)
                granular_balls = GBNRS.GBList(data, False, self.gb_list.granular_balls)
                gb_division = granular_balls.re_division(remove_i)
                purity = [round(ball.purity, 3) for ball in gb_division]
                # if sum(purity) == len(purity)-0.05:  # if the ith attribute can be reduced statistics
                # print("purity", statistics.mean(purity))
                if statistics.mean(purity) >= 0.7:
                    # Recreate the new list granular balls with attributes after the reduction
                    # step 1.
                    granular_balls = GBNRS.GBList(np.hstack((data[:, attributes_reduction], data[:, -2:])))
                    granular_balls.init_granular_balls()
                    granular_balls.del_balls(num_data=2)

                    # step 2.
                    granular_balls.re_k_means()
                    granular_balls.del_balls(purity=1)

                else:  # If the ith attribute is can't be reduced, then add it back.
                    attributes_reduction.append(remove_i)
                    attributes_reduction.sort()
            else:
                break
            i += 1
        remain_attributes = list(set(list(range(data.shape[1]-2))) - set(attributes_reduction))
        return attributes_reduction, remain_attributes



























