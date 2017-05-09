#!/bin/bash

files=`ls *.xml`

for xmlfile in $files
do
  xmlstarlet sel -t -v "//sourcePayloadAsBase64" "$xmlfile" > ./work/1.csv
  xmlstarlet sel -t -v "//destinationPayloadAsBase64" "$xmlfile" >./work/2.csv
  xmlstarlet sel -t -v "//Tag" "$xmlfile" > ./work/3.csv
  paste -d, ./work/1.csv ./work/2.csv ./work/3.csv > ./work/$xmlfile.csv
  rm -rf ./work/{1,2,3}.csv
done

