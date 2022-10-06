Data is pubic.    

# Python_Spark_ML_SVM
Use SVM algorithm to machine learning, find the best mode to predict the validation of website.

Running environment is Spark + Hadoop + PySpark    
Used the algorithm is Support Vetcor Machine (SVM).     
Used the library is pyspark.mllib. 

# Stage1:  Read data
Placed the tsv on hadoop. Built 3 data sets: (1) Train data, (2) Validation data, (3) Sub_test data.


## Compare the parameters
"numIterations".  
Set the step_size=10 and reg_param=1, draw the graph for the numIterations. The AUC is the highest when num_iteration is 1. But the AUCs are similar, only little difference.
~~~
    num_iterations_list = [1, 3, 5, 15, 25]
    step_size_list = [100]
    reg_param_list = [1]
~~~
![image](https://user-images.githubusercontent.com/75282285/194394064-a366c96c-a1e4-481e-b00a-c67dfedf1357.png)



"stepSize"
Set the mum_iteration=15 and mini_batch_fraction=0.8, draw the graph for the stepSize. The AUC is the hightest when step_size is 10.
~~~
    num_iterations_list = [15]
    step_size_list = [10, 50, 100, 200]
    mini_batch_fraction_list = [0.8]
~~~
![image](https://user-images.githubusercontent.com/75282285/194132258-339e74ad-a243-4652-836a-8d49783c321d.png)


"miniBatchFraction".
Set the num_iteration=15 and step_size=10, draw the graph for the miniBatchFraction. The AUC is the hightest when mini_batch_fraction is 0.8.
~~~
    num_iterations_list = [15]
    step_size_list = [10]
    mini_batch_fraction_list = [0.5, 0.8, 1]
~~~
![image](https://user-images.githubusercontent.com/75282285/194132553-c1165925-9e03-4428-aac9-3549852a9262.png)



# Stage2: Train and evaluate   
Created the model using train data set.   
Calculated the AUC using validation data set.
Sorted the metrics.    
Found the best parameters includ the best AUC and the best model.   
![image](https://user-images.githubusercontent.com/75282285/194131381-d1165b01-d03c-4b8d-ba75-81ef9eb4601d.png)


# Stage3: Test
Used the sub_test data set and the best model to calculate the AUC. If testing AUC is similare as the best AUC, it is OK.
As the result, the best AUC is  0.6610, use the test data set to calcuate AUC is 0.6565, the difference is 0.0045, so it has not overfitting issue. 
![image](https://user-images.githubusercontent.com/75282285/194131330-67a56a36-75d5-4501-b8f7-6bdcefb960ba.png)


# Stage4: Predict
Use the test data (in Hadoop, test.tsv) and the model (calculated after Stage2) to predict.
~~~
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
~~~
![image](https://user-images.githubusercontent.com/75282285/194137552-20d00455-5121-4207-9712-bfe8c663883e.png)

# Spark monitor
http://node1:4040/jobs/
![image](https://user-images.githubusercontent.com/75282285/194138884-6a3329fa-083b-47e6-8657-42f8008baf88.png)
![image](https://user-images.githubusercontent.com/75282285/194139282-0bd3e21d-7076-49fd-8b8e-5d70cf22ace7.png)     
In Spark, it ran at least 3213 jobs.      
![image](https://user-images.githubusercontent.com/75282285/194139415-f61330f4-2825-4edd-8043-82a0e39cc832.png)

http://node1:8080/
![image](https://user-images.githubusercontent.com/75282285/194138246-08d1c8a2-749e-4f72-b91b-f1fa5d69dc51.png)
