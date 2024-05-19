import CFS_hnrs
import CFS_gbnrs
import warnings
import os
import sys




warnings.filterwarnings("ignore")


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



os_path = os.path.abspath(os.path.dirname(__file__))  # 获取当前py文件的父目录
type = sys.getfilesystemencoding()

if __name__ == "__main__":

    dataset = [ "zoo"]

    for name in dataset:
        path = r'data\\' + name + '.csv'
        sys.stdout = Logger(r'Optimal_Result/' + name + '.txt')
        model = CFS_gbnrs #CFS_hnrs #
        result = model.cfs(path, name)
        temp_len = len(result)
        init_data_ratio = [0.3]
        new_data_ratio = [0.4]
        for j in init_data_ratio:
            for h in new_data_ratio:
                temp_acc0=temp_f10=temp_acc1=temp_f11=temp_acc2=temp_f12=temp_acc3=temp_f13= temp_acc4= temp_f14 = 0
                Timer0= Timer1= Timer2= Timer3= Timer4= Timer5= Timer6= Timer7= Timer8= Timer9 = 0
                for i in range(temp_len):
                    if result[i]["init_r"] == j and result[i]["new_r"] == h and result[i]["k_classify"] == 0:
                        if result[i]["cfs_a"] > temp_acc0:
                            temp_acc0 = result[i]["cfs_a"]
                            Timer0 = i
                        if result[i]["cfs_f1"] > temp_f10:
                            temp_f10 = result[i]["cfs_f1"]
                            Timer1 = i

                    if result[i]["init_r"] == j and result[i]["new_r"] == h and result[i]["k_classify"] == 1:
                        if result[i]["cfs_a"] > temp_acc1:
                            temp_acc1 = result[i]["cfs_a"]
                            Timer2 = i
                        if result[i]["cfs_f1"] > temp_f11:
                            temp_f11 = result[i]["cfs_f1"]
                            Timer3 = i

                    if result[i]["init_r"] == j and result[i]["new_r"] == h and result[i]["k_classify"] == 2:
                        if result[i]["cfs_a"] > temp_acc2:
                            temp_acc2 = result[i]["cfs_a"]
                            Timer4 = i
                        if result[i]["cfs_f1"] > temp_f12:
                            temp_f12 = result[i]["cfs_f1"]
                            Timer5 = i

                    if result[i]["init_r"] == j and result[i]["new_r"] == h and result[i]["k_classify"] == 3:
                        if result[i]["cfs_a"] > temp_acc3:
                            temp_acc3 = result[i]["cfs_a"]
                            Timer6 = i
                        if result[i]["cfs_f1"] > temp_f13:
                            temp_f13 = result[i]["cfs_f1"]
                            Timer7 = i

                    if result[i]["init_r"] == j and result[i]["new_r"] == h and result[i]["k_classify"] == 4:
                        if result[i]["cfs_a"] > temp_acc4:
                            temp_acc4 = result[i]["cfs_a"]
                            Timer8 = i
                        if result[i]["cfs_f1"] > temp_f14:
                            temp_f14 = result[i]["cfs_f1"]
                            Timer9 = i
                print("___________KNN___________")
                print("ori acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$" % (
                result[Timer0]["o_a"], result[Timer0]["o_s"], result[Timer1]["o_f1"], result[Timer1]["o_fs"]))
                print("s, init_ratio, new_ratio:", result[Timer0]["s"], j, h)
                print("cfs acc:$%0.2lf_{\pm %.2lf}$ " % (temp_acc0, result[Timer0]["cfs_s"]))
                print("s, init_ratio, new_ratio:", result[Timer1]["s"], j, h)
                print("cfs f1:$%0.2lf_{\pm %.2lf}$" % (temp_f10,  result[Timer1]["cfs_fs"]))

                print("___________SVM___________") #SVM, DT, RF, LGR
                print("ori acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$" % (
                result[Timer2]["o_a"], result[Timer2]["o_s"], result[Timer3]["o_f1"], result[Timer3]["o_fs"]))
                print("s, init_ratio, new_ratio:", result[Timer2]["s"], j, h)
                print("cfs acc:$%0.2lf_{\pm %.2lf}$ " % (temp_acc1, result[Timer2]["cfs_s"]))
                print("s, init_ratio, new_ratio:", result[Timer3]["s"], j, h)
                print("cfs f1:$%0.2lf_{\pm %.2lf}$" % (temp_f11,  result[Timer3]["cfs_fs"]))

                print("___________DT___________")
                print("ori acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$" % (
                result[Timer4]["o_a"], result[Timer4]["o_s"], result[Timer5]["o_f1"], result[Timer5]["o_fs"]))
                print("s, init_ratio, new_ratio:", result[Timer4]["s"], j, h)
                print("cfs acc:$%0.2lf_{\pm %.2lf}$ " % (temp_acc2, result[Timer4]["cfs_s"]))
                print("s, init_ratio, new_ratio:", result[Timer5]["s"], j, h)
                print("cfs f1:$%0.2lf_{\pm %.2lf}$" % (temp_f12,  result[Timer5]["cfs_fs"]))

                print("___________RF___________")
                print("ori acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$" % (
                result[Timer6]["o_a"], result[Timer6]["o_s"], result[Timer7]["o_f1"], result[Timer7]["o_fs"]))
                print("s, init_ratio, new_ratio:", result[Timer6]["s"], j, h)
                print("cfs acc:$%0.2lf_{\pm %.2lf}$ " % (temp_acc3, result[Timer6]["cfs_s"]))
                print("s, init_ratio, new_ratio:", result[Timer7]["s"], j, h)
                print("cfs f1:$%0.2lf_{\pm %.2lf}$" % (temp_f13,  result[Timer7]["cfs_fs"]))

                print("___________LGR___________")
                print("ori acc:$%0.2lf_{\pm %.2lf}$,  f1:$%0.2lf_{\pm %.2lf}$" % (
                result[Timer8]["o_a"], result[Timer8]["o_s"], result[Timer9]["o_f1"], result[Timer9]["o_fs"]))
                print("s, init_ratio, new_ratio:", result[Timer8]["s"], j, h)
                print("cfs acc:$%0.2lf_{\pm %.2lf}$ " % (temp_acc4, result[Timer8]["cfs_s"]))
                print("s, init_ratio, new_ratio:", result[Timer9]["s"], j, h)
                print("cfs f1:$%0.2lf_{\pm %.2lf}$" % (temp_f14,  result[Timer9]["cfs_fs"]))

                print("*******************************************************")








