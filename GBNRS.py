
"""
granular ball neighborhood rough set
"""
import numpy as np
from collections import Counter
from sklearn.cluster import k_means
import warnings
import statistics
warnings.filterwarnings("ignore")


class GranularBall:
    """class of the granular ball"""
    def __init__(self, data):
        """
        :param data:  Labeled data set, the "-2" column is the class label, the last column is the index of each line
        and each of the preceding columns corresponds to a feature
        """
        self.data = data[:, :]
        self.data_no_label = data[:, :-2]
        self.num, self.dim = self.data_no_label.shape
        self.center = self.data_no_label.mean(0)
        self.label, self.purity = self.__get_label_and_purity()
        self.radius = self.get_radius()

    def __get_label_and_purity(self):
        """
        :return: the label and purity of the granular ball.
        """
        #cprint("data len", len(self.data[:-2]))

        count = Counter(self.data[:, -2])
        # print(count)
        if len(count) != 0:
            label = max(count, key=count.get)
            purity = count[label] / self.num
        else:
            label = 0
            purity = 0

        return label, purity


    def get_radius(self):

        # 计算每个点到中心点的距离
        distances = np.linalg.norm(self.data_no_label - self.center, axis=1, ord=2)

        # 找到最大距离和对应的点索引
        max_distance = np.max(distances)
        return max_distance
    def split_2balls(self):
        """
        split the granular ball to 2 new balls by using 2_means.
        """
        label_cluster = k_means(X=self.data_no_label, n_clusters=2)[1]
        if sum(label_cluster == 0) and sum(label_cluster == 1):
            ball1 = GranularBall(self.data[label_cluster == 0, :])
            ball2 = GranularBall(self.data[label_cluster == 1, :])
        else:
            ball1 = GranularBall(self.data[0:1, :])
            ball2 = GranularBall(self.data[1:, :])
        return ball1, ball2


class GBList:
    """class of the list of granular ball"""
    def __init__(self, data=None, original_gb=True, gb_list=None):
        self.data = data[:, :]
        if original_gb:
            self.granular_balls = [GranularBall(self.data)]  # gbs is initialized with all data
        else:
            self.granular_balls = gb_list

    def init_granular_balls(self, purity=1.0, min_sample=1):
        """
        Split the balls, initialize the balls list.
        :param purity: If the purity of a ball is greater than this value, stop splitting.
        :param min_sample: If the number of samples of a ball is less than this value, stop splitting.
        """
        ll = len(self.granular_balls)
        i = 0
        while True:
            if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
                split_balls = self.granular_balls[i].split_2balls()
                self.granular_balls[i] = split_balls[0]
                self.granular_balls.append(split_balls[1])
                ll += 1
            else:
                i += 1
            if i >= ll:
                break
        self.data = self.get_data()

    def get_data_size(self):
        return list(map(lambda x: len(x.data), self.granular_balls))

    def get_purity(self):
        return list(map(lambda x: x.purity, self.granular_balls))

    def get_center(self):
        """
        :return: the center of each ball.
        """
        return np.array(list(map(lambda x: x.center, self.granular_balls)))

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.data for ball in self.granular_balls]
        return np.vstack(list_data)

    def del_balls(self, purity=0, num_data=0):
        """
        Deleting the balls that meets following conditions from the list, updating self.granular_balls and self.data.
        :param purity: delete the balls that purity is large than this value.
        :param num_data: delete the balls that the number of samples is large than this value.
        :return: None
        """
        self.granular_balls = [ball for ball in self.granular_balls if ball.purity >= purity and ball.num >= num_data]
        self.data = self.get_data()

    def re_k_means(self):
        """
        Global k-means clustering for data with the center of the ball as the initial center point.
        """
        k = len(self.granular_balls)
        label_cluster = k_means(X=self.data[:, :-2], n_clusters=k, init=self.get_center())[1]
        for i in range(k):
            self.granular_balls[i] = GranularBall(self.data[label_cluster == i, :])

    def re_division(self, i):
        """
        Data division with the center of the ball.
        :return: a list of new granular balls after divisions.
        """
        k = len(self.granular_balls)
        attributes = list(range(self.data.shape[1] - 2))
        attributes.remove(i)
        label_cluster = k_means(X=self.data[:, attributes], n_clusters=k,
                                init=self.get_center()[:, attributes], max_iter=1)[1]
        granular_balls_division = []
        for i in set(label_cluster):
            granular_balls_division.append(GranularBall(self.data[label_cluster == i, :]))
        return granular_balls_division

def create_gblist(data, label):

    data_with_label = np.hstack((data, label))
    num, dim = data_with_label[:, :-1].shape
    index = np.array(range(num)).reshape(num, 1)  # column of index
    data = np.hstack((data_with_label, index))  # Add the index column to the last column of the data

    # step 1.
    granular_balls = GBList(data, original_gb=True)  # create the list of granular balls
    granular_balls.init_granular_balls()  # initialize the list
    granular_balls.del_balls(num_data=2)  # delete the ball with 1 (less than 2) sample

    # step 2.
    granular_balls.re_k_means()  # Global k-means clustering as fine tuning.
    granular_balls.del_balls(purity=1)

    return granular_balls, data

def get_attribute_reduction(data, label):
    """
    The main function of attribute reduction.
    :param data: data set
    :return: reduced attribute set
    """
    granular_balls, data = create_gblist(data, label)
    init_granular_balls = granular_balls
    # step 3.
    attributes_reduction = list(range(data.shape[1] - 2))
    for i in range(data.shape[1] - 2):
        if len(attributes_reduction) <= 1:
            break

        the_remove_i = attributes_reduction.index(i)
        attributes_reduction.remove(i)  # remove the ith attribute
        gb_division = granular_balls.re_division(the_remove_i)  # divide the data with center of granular balls
        purity = [round(ball.purity, 3) for ball in gb_division]  # get the purity of the divided granular balls

        # if sum(purity) == len(purity):  # if the ith attribute can be reduced
        if statistics.mean(purity) >=0.95 :
            # Recreate the new list granular balls with attributes after the reduction
            # step 1.
            granular_balls = GBList(np.hstack((data[:, attributes_reduction], data[:, -2:])), original_gb=True)
            granular_balls.init_granular_balls()
            granular_balls.del_balls(num_data=3)

            # step 2.
            granular_balls.re_k_means()
            granular_balls.del_balls(purity=1)

        else:  # If the ith attribute is can't be reduced, then add it back.
            attributes_reduction.append(i)
            attributes_reduction.sort()
    attributes_reduction = attributes_reduction
    remain_attributes = list(set(list(range(data.shape[1] - 2))) - set(attributes_reduction))
    return attributes_reduction, remain_attributes, init_granular_balls

def open_attribute_reduction(GB_list):
    """
    The main function of attribute reduction.
    :param data: data set
    :return: reduced attribute set
    """
    granular_balls, data = GB_list.granular_balls, GB_list.data

    init_granular_balls = granular_balls
    # step 3.
    attributes_reduction = list(range(data.shape[1] - 2))
    for i in range(data.shape[1] - 2):
        if len(attributes_reduction) <= 1:
            break

        the_remove_i = attributes_reduction.index(i)
        attributes_reduction.remove(i)  # remove the ith attribute
        gb_division = granular_balls.re_division(the_remove_i)  # divide the data with center of granular balls
        purity = [round(ball.purity, 3) for ball in gb_division]  # get the purity of the divided granular balls

        if statistics.mean(purity):  # if the ith attribute can be reduced
            # Recreate the new list granular balls with attributes after the reduction
            # step 1.
            granular_balls = GBList(np.hstack((data[:, attributes_reduction], data[:, -2:])), original_gb=True)
            granular_balls.init_granular_balls()
            granular_balls.del_balls(num_data=3)

            # step 2.
            granular_balls.re_k_means()
            granular_balls.del_balls(purity=1)

        else:  # If the ith attribute is can't be reduced, then add it back.
            attributes_reduction.append(i)
            attributes_reduction.sort()
    attributes_reduction = attributes_reduction
    remain_attributes = list(set(list(range(data.shape[1]))) - set(attributes_reduction))
    return attributes_reduction, remain_attributes, init_granular_balls
