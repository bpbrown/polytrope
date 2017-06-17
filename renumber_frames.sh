#!/usr/bin/env bash

INCR=100
while getopts ":i:" opt; do
    case $opt in
        i)
            INCR=${OPTARG}
            shift 2
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "OPTION -$OPTARG requires an integer to increment frame numbers by." >&2
            exit 1
            ;;
    esac
done
echo "INCR is $INCR"

DIRNAME=$1

OUTPUTDIR=`dirname $DIRNAME`
if [ ! -d $OUTPUTDIR/combined_frames/ ]; then
    mkdir $OUTPUTDIR/combined_frames
fi

for f in `ls $DIRNAME/frames/*png`; do
    field_num=`grep -o "/" <<< $f | wc -l`
    let field_num=field_num+1
    VAR=`echo $f| cut -d/ -f $field_num | sed -rn 's/snapshot_([0-9]+).png/\1/p'`
    let VAR=10#$VAR+$INCR
    printf -v FILENAME "snapshot_%06i.png" $VAR
    cp $f $OUTPUTDIR/combined_frames/$FILENAME
done
