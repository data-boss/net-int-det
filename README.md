# net-int-det
Dataset short description:
Total number of labeled connections : 1885519
Normal connections : 1816609
Attack connections : 68910 (~3.65%)
Attack types : Bruteforce, Infilterating network from inside, HTTP DoS, DDoS using IRC Botnet, Bruteforce SSH. 

1 -- xmlparser.sh: This script extract the connection-wise attributes from the network labeled flows of ISCX datasets.
Run the following in terminal. 
bash xmlparser.sh <file-name>.xml 

The ouput is saved in a csv file with the name "input-file-name".csv. 

xmlstarlet is used to extract the attribute values /parse xml files. For instance, to extract the labels, we may use;

xmlstarlet sel -t -v "//Tag" 

2-- seq_features.py: This python script takes the payloads (source and destination) of each connection and the respective labels as file1 and file2. Then generate n-grams of selected size, dictionarize them and finally maps the "text" payloads to vector space. 

How to run:
python seq_features.py file1 file2 k

