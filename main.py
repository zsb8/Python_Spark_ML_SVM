from pyspark import SparkConf, SparkContext
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from time import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.feature import StandardScaler


def create_spark_context():
    global sc, path
    sc = SparkContext(conf=SparkConf().setAppName('test'))
    path = "hdfs://node1:8020/input/"


def read_data():
    global lines, categories_map
    raw_data_with_header = sc.textFile(path + "train.tsv")
    # print(f"raw_data_with_Header=={raw_data_with_header.take(2)}")
    header = raw_data_with_header.first()
    raw_data = raw_data_with_header.filter(lambda x: x != header)
    r_data = raw_data.map(lambda x: x.replace("\"", ""))
    lines = r_data.map(lambda x: x.split('\t'))
    categories_map = lines.map(lambda fields: fields[3]).distinct().zipWithIndex().collectAsMap()


def convert_float(x):
    if x == "?":
        result = 0
    else:
        result = float(x)
    return result


def extract_features(field, categories_map, feature_end):
    category_idx = categories_map[field[3]]
    category_features = np.zeros(len(categories_map))
    category_features[category_idx] = 1
    numerical_features = [convert_float(field) for field in field[4: feature_end]]
    result = np.concatenate((category_features, numerical_features))
    return result


def extract_label(field):
    label = field[-1]
    result = float(label)
    return result


def prepare_data():
    print("Before standard:")
    label_RDD = lines.map(lambda x: extract_label(x))
    feature_RDD = lines.map(lambda r: extract_features(r, categories_map, len(r)-1))
    for i in feature_RDD.first():
        print(f"{i},")
    print("After standard:")
    std_scaler = StandardScaler(withMean=True, withStd=True).fit(feature_RDD)
    scaler_feature_RDD = std_scaler.transform(feature_RDD)
    for i in scaler_feature_RDD.first():
        print(i)
    label_point = label_RDD.zip(scaler_feature_RDD)
    label_point_RDD = label_point.map(lambda x: LabeledPoint(x[0], x[1]))
    result = label_point_RDD.randomSplit([8, 1, 1])
    return result


def evaluate_model(model, validation_data):
    score = model.predict(validation_data.map(lambda x: x.features))
    score_and_labels = score.zip(validation_data.map(lambda x: x.label)).map(lambda x: (float(x[0]), float(x[1])))
    metrics = BinaryClassificationMetrics(score_and_labels)
    auc = metrics.areaUnderROC
    return auc


def train_evaluation_model(train_data,
                           validation_data,
                           num_iterations,
                           step_size,
                           mini_batch_fraction):
    start_time = time()
    model = SVMWithSGD.train(train_data, num_iterations, step_size, mini_batch_fraction)
    auc = evaluate_model(model, validation_data)
    # print(f"AUC ==: {auc}")
    duration = time() - start_time
    # print(f"The time to train was: {duration}")
    return auc, duration, num_iterations, step_size, mini_batch_fraction, model


def eval_parameter(train_data, validation_data):
    num_iterations_list = [1, 3, 5, 15, 25]
    step_size_list = [10, 50, 100, 200]
    reg_param_list = [0.01, 0.1, 1]
    my_metrics = [
        train_evaluation_model(train_data, validation_data, num_iterations, step_size, reg_param)
        for num_iterations in num_iterations_list
        for step_size in step_size_list
        for reg_param in reg_param_list
    ]
    s_metrics = sorted(my_metrics, key=lambda x: x[0], reverse=True)
    best_parameter = s_metrics[0]
    print(f"The best numIterations_list is:{best_parameter[2]}\n"
          f"The best stepSize:{best_parameter[3]}\n"
          f"The best regParam is:{best_parameter[4]}\n"
          f"The best AUC is:{best_parameter[0]}\n")
    best_auc = best_parameter[0]
    best_model = best_parameter[5]
    return best_auc, best_model


def predict_data(best_model):
    raw_data_with_header = sc.textFile(path + "test.tsv")
    header = raw_data_with_header.first()
    raw_data = raw_data_with_header.filter(lambda x: x != header)
    r_data = raw_data.map(lambda x: x.replace("\"", ""))
    lines_test = r_data.map(lambda x: x.split('\t'))
    data_rdd = lines_test.map(lambda x: (x[0], extract_features(x, categories_map, len(x))))
    dic_desc = {
        0: 'temp web',
        1: 'evergreen web'
    }
    for data in data_rdd.take(10):
        result_predict = best_model.predict(data[1])
        print(f"web:{data[0]}, \n predict:{result_predict}, desc: {dic_desc[result_predict]}")


def show_chart(df, eval_parm, bar_parm, line_parm, y_min=0.5, y_max=1.0):
    ax = df[bar_parm].plot(kind='bar', title=eval_parm, figsize=(10, 6), legend=True, fontsize=12)
    ax.set_xlabel(eval_parm, fontsize=12)
    ax.set_ylim([y_min, y_max])
    ax.set_ylabel(bar_parm, fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df[line_parm].values, linestyle='-', marker='o', linewidth=2, color='r')
    plt.show()


def draw_graph(train_data, validation_data, eval_parm):
    num_iterations_list = [1, 3, 5, 15, 25]
    step_size_list = [10, 50, 100, 200]
    reg_param_list = [0.01, 0.1, 1]
    if eval_parm == "numIterations":
        index_list = num_iterations_list
        step_size_list = [100]
        reg_param_list = [1]
    elif eval_parm == "stepSize":
        index_list = step_size_list
        num_iterations_list = [25]
        reg_param_list = [1]
    elif eval_parm == "regParam":
        index_list = reg_param_list
        num_iterations_list = [25]
        step_size_list = [100]
    my_metrics = [
        train_evaluation_model(train_data, validation_data, num_iterations, step_size, reg_param)
        for num_iterations in num_iterations_list
        for step_size in step_size_list
        for reg_param in reg_param_list
    ]
    df = pd.DataFrame(my_metrics,
                      index=index_list,
                      columns=['AUC', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    show_chart(df, eval_parm, 'AUC', 'duration', 0.5, 0.7)


if __name__ == "__main__":
    s_time = time()
    create_spark_context()
    print("Reading data stage".center(60, "="))
    read_data()
    train_d, validation_d, test_d = prepare_data()
    print(train_d.first())

    print("Draw the graph - iteration".center(60, "="))
    draw_graph(train_d, validation_d, "numIterations")
    print("Draw the graph - setSize".center(60, "="))
    draw_graph(train_d, validation_d, "stepSize")
    print("Draw the graph - regParam".center(60, "="))
    draw_graph(train_d, validation_d, "regParam")

    print("Evaluate parameter".center(60, "="))
    b_auc, b_model = eval_parameter(train_d, validation_d)
    print(f"The best AUC is: {b_auc}")
    print("Test".center(60, "="))
    test_data_auc = evaluate_model(b_model, test_d)
    print(f"best auc is:{format(b_auc, '.4f')}, test_data_auc is: {format(test_data_auc, '.4f')}, "
          f"they are only slightly different:{format(abs(float(b_auc) - float(test_data_auc)), '.4f')}")
    print("Predict".center(60, "="))
    predict_data(b_model)









