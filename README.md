# net-int-det
Dataset short description:
Total number of labeled connections : 1885519
Normal connections : 1816609
Attack connections : 68910 (~3.65%)
Attack types with number of instances: Bruteforce (12 june: 2086 + 16 june: 14), Infilterating network from inside (13 june : 20358), HTTP DoS (14 june : 3777), DDoS using IRC Botnet (15 june : 37461), Bruteforce SSH (17 june : 5261). 

1 -- xmlparser.sh: This script extract the connection-wise attributes from the network labeled flows of ISCX datasets.
Run the following in terminal. 
bash xmlparser.sh <file-name>.xml 

The ouput is saved in a csv file with the name "input-file-name".csv. 

xmlstarlet is used to extract the attribute values /parse xml files. For instance, to extract the labels, we may use;

xmlstarlet sel -t -v "//Tag" 

2-- seq_features.py: This python script takes the payloads (source and destination) of each connection and the respective labels as file1 and file2. Then generate n-grams of selected size, dictionarize them and finally maps the "text" payloads to vector space. 

How to run:
python seq_features.py file1 file2 k

Datasets description:

1. The train-test-with-seeds.tar.gz contains training set (90%), test set (10 %) and the seed (binary array) to generate them. All the experiments needs to be done on the training set and it needs to be considered the ultimate and only available data for time being.

2. The training set from above needs to be further split into development and test set with 80-20 ratio. Since each experiment is to be repeated five times and scores to be averaged, for the split use the seeds provided in the "developement-sets-seeds.gz". Use one column for each iteration.

3. For each train-test split in the development set, the training is to be done using 5-fold cross-validation. The "cross-val-seeds.tar.gz" contains five binary arrays each one for each development set in (2). Each binary array contains five columns where each column corresponds to a different 80-20 split for cross-validation.

4. Overall, there would be five experiments with training and testing, and each training is done using 5-fold cross-validation.

5. The labels are of six types and for binary classification labels other than "Normal" can be merged together.

6. For split in each scenario, all the samples where the corresponding index in seeds is zero would go into training and the ones with seed 1 would be used for testing.
For instance X_train=data[seed==0,:-1], X_test=data[seed==1,:-1], Y_train=data[seed==0,-1], Y_test=data[seed==1,-1].
