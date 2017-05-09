#!/bin/bash

files=`ls ~/Desktop/NET*/*.xml`
{
  for xmlfile in $files 
  do
    xmlstarlet sel -t -v "//sourcePayloadAsBase64" "$xmlfile" > 1.csv
    xmlstarlet sel -t -v "//destinationPayloadAsBase64" "$xmlfile" > 2.csv
    xmlstarlet sel -t -v "//Tag" "$xmlfile" > 3.csv
    paste -d, 1.csv 2.csv 3.csv 
  done
} > final.csv

