#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 02:11:59 2017

@author: farhan
"""
import pandas as pd
import xml.etree.ElementTree as ET
import csv
import sys
tree = ET.parse(sys.argv[1])
root = tree.getroot()

# open a file for writing

output = open(str(sys.argv[1])+'.csv', 'w')

# create the csv writer object

csvwriter = csv.writer(output)
file_head = []

count = 0
for member in root.findall(root[0].tag):
    file = []
    if count == 0:
        appname = member.find('appName').tag
        file_head.append(appname)
        srcb = member.find('totalSourceBytes').tag
        file_head.append(srcb)
        dstb = member.find('totalDestinationBytes').tag
        file_head.append(dstb)
        srcp=member.find('totalSourcePackets').tag
        file_head.append(srcp)
        dstp=member.find('totalDestinationPackets').tag
        file_head.append(dstp)
        srcpl=member.find('sourcePayloadAsBase64').tag
        file_head.append(srcpl)
        dstpl=member.find('destinationPayloadAsBase64').tag
        file_head.append(dstpl)
        direction=member.find('direction').tag
        file_head.append(direction)
        srcfl=member.find('sourceTCPFlagsDescription').tag
        file_head.append(srcfl)
        dstfl=member.find('destinationTCPFlagsDescription').tag
        file_head.append(dstfl)
        src=member.find('source').tag
        file_head.append(src)
        proto=member.find('protocolName').tag
        file_head.append(proto)
        srcpo=member.find('sourcePort').tag
        file_head.append(srcpo)
        dst=member.find('destination').tag
        file_head.append(dst)
        dstpo=member.find('destinationPort').tag
        file_head.append(dstpo)
        start=member.find('startDateTime').tag
        file_head.append(start)
        end=member.find('stopDateTime').tag
        file_head.append(end)
        tag=member.find('Tag').tag
        file_head.append(tag)
        csvwriter.writerow(file_head)
        count = count + 1
        
    appname = member.find('appName').text
    file.append(appname)
    srcb = member.find('totalSourceBytes').text
    file.append(srcb)
    dstb = member.find('totalDestinationBytes').text
    file.append(dstb)
    srcp=member.find('totalSourcePackets').text
    file.append(srcp)
    dstp=member.find('totalDestinationPackets').text
    file.append(dstp)
    srcpl=member.find('sourcePayloadAsBase64').text
    file.append(srcpl)
    dstpl=member.find('destinationPayloadAsBase64').text
    file.append(dstpl)
    direction=member.find('direction').text
    file.append(direction)
    srcfl=member.find('sourceTCPFlagsDescription').text
    file.append(srcfl)
    dstfl=member.find('destinationTCPFlagsDescription').text
    file.append(dstfl)
    src=member.find('source').text
    file.append(src)
    proto=member.find('protocolName').text
    file.append(proto)
    srcpo=member.find('sourcePort').text
    file.append(srcpo)
    dst=member.find('destination').text
    file.append(dst)
    dstpo=member.find('destinationPort').text
    file.append(dstpo)
    start=member.find('startDateTime').text
    file.append(pd.to_datetime(start[:10]+"-"+start[11:]))
    end=member.find('stopDateTime').text
    file.append(pd.to_datetime(end[:10]+"-"+end[11:]))
    tag=member.find('Tag').text
    file.append(tag)
	
    csvwriter.writerow(file)
output.close()

