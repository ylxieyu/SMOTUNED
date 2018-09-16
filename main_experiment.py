# 主实验
# 比较No-SMOTE、MAHAKIL、SMOTE、SMOTUNED 的性能
# 采用5*5 交叉验证
from Tools.DataSetTool import DataSetTool
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from Tools.EvaluationTool import EvaluationTool
from MAHAKIL.mahakil import MAHAKIL
from SMOTUNED.smote import SMOTE
from SMOTUNED.smotuned import SMOTUNED

def main_experiment(data_list, label_list, sampling_method, kf):
    for i in range(len(data_list)):
        for train_index, test_index in kf.split(data_list[i]):
            data_train, data_test = data_list[i][train_index], data_list[i][test_index]
            label_train, label_test = label_list[i][train_index], label_list[i][test_index]
            if sampling_method == 'NO_SMOTE':
                clf = LogisticRegression()
                clf.fit(data_train, label_train)
                predictions = clf.predict(data_test)
                EvaluationTool().get_output(predictions, label_test, index=sampling_method)
            if sampling_method == 'MAHAKIL':
                data_train_, label_train_ = MAHAKIL().fit_sample(data_train, label_train)
                clf = LogisticRegression()
                clf.fit(data_train_, label_train_)
                predictions = clf.predict(data_test)
                EvaluationTool().get_output(predictions, label_test, index=sampling_method)
            if sampling_method == 'SMOTE':
                data_train_, label_train_ = SMOTE().fit_sample(data_train, label_train)
                clf = LogisticRegression()
                clf.fit(data_train_, label_train_)
                predictions = clf.predict(data_test)
                EvaluationTool().get_output(predictions, label_test, index=sampling_method)
            if sampling_method == 'SMOTUNED':
                data_bin, de_data, label_bin, de_label = train_test_split(data_train, label_train, test_size=0.25, random_state=0)
                k, m, r = SMOTUNED().DE(de_data, de_label)
                data_bin_, label_bin_ = SMOTE(k=k, m=m, r=r).fit_sample(data_bin, label_bin)
                clf = LogisticRegression()
                clf.fit(data_bin_, label_bin_)
                predictions = clf.predict(data_test)
                EvaluationTool().get_output(predictions, label_test, index=sampling_method)

data_list, label_list = DataSetTool.init_data('D:\\data\\txt\\', 20, False, False)
kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
# main_experiment(data_list, label_list, 'NO_SMOTE', kf)
# main_experiment(data_list, label_list, 'MAHAKIL', kf)
#  TimeoutError: [WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。
#  34个结果
# main_experiment(data_list, label_list, 'SMOTE', kf)
main_experiment(data_list, label_list, 'SMOTUNED', kf)