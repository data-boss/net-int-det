#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:35:40 2017

@author: farhan
"""
import pandas as pd

d1=pd.read_csv("TestbedSatJun12Flows.xml.csv",header=0)
d2=pd.read_csv("TestbedSunJun13Flows.xml.csv",header=0)
d3=pd.read_csv("TestbedMonJun14Flows.xml.csv",header=0)
d4=pd.read_csv("TestbedTueJun15-1Flows.xml.csv",header=0)
d5=pd.read_csv("TestbedTueJun15-2Flows.xml.csv",header=0)
d6=pd.read_csv("TestbedTueJun15-3Flows.xml.csv",header=0)
d7=pd.read_csv("TestbedWedJun16-1Flows.xml.csv",header=0)
d8=pd.read_csv("TestbedWedJun16-2Flows.xml.csv",header=0)
d9=pd.read_csv("TestbedWedJun16-3Flows.xml.csv",header=0)
d10=pd.read_csv("TestbedThuJun17-1Flows.xml.csv",header=0)
d11=pd.read_csv("TestbedThuJun17-2Flows.xml.csv",header=0)
d12=pd.read_csv("TestbedThuJun17-3Flows.xml.csv",header=0)

    

d1=d1.replace("Attack","bruteforce",regex=True)
d2=d2.replace("Attack","insider",regex=True)     
d3=d3.replace("Attack","http-dos",regex=True)
d4=d4.replace("Attack","botnet-ddos",regex=True)  
d5=d5.replace("Attack","botnet-ddos",regex=True) 
d6=d6.replace("Attack","botnet-ddos",regex=True)
d7=d7.replace("Attack","bruteforce",regex=True)  
d8=d8.replace("Attack","bruteforce",regex=True)
d9=d9.replace("Attack","bruteforce",regex=True)
d10=d10.replace("Attack","bruteforce-ssh",regex=True)
d11=d11.replace("Attack","bruteforce-ssh",regex=True)
d12=d12.replace("Attack","bruteforce-ssh",regex=True)
data=[d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12]

pd.concat(data,axis=0).to_csv("all_connection_labeled_csv.csv")