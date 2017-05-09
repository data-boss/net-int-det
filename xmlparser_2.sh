#!/bin/bash

file="$1"
rd="$(xmlstarlet el "$1" | sed -s -n 2p)"
xmlstarlet sel -T \
-t -m '$rd' \
-v 'concat(appName;',';totalSourceBytes;',';totalDestinationBytes;',';totalDestinationPackets;',';totalSourcePackets;',';sourcePayloadAsBase64;',';sourcePayloadAsUTF;',';destinationPayloadAsBase64;',';destinationPayloadAsUTF;',';direction;',';sourceTCPFlagsDescription;',';destinationTCPFlagsDescription;',';source;',';protocolName;',';sourcePort;',';destination;',';destinationPort;',';startDateTime;',';stopDateTime;',';Tag)'\ "$1" > $1.csv

