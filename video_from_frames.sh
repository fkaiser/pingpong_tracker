#!/bin/bash
frame_rate=$2
output_name=$1"/output.mp4"
for d in $1/*; do
    [[ $d =~ .*image(0*)([1-9]+).png ]]
    if [[ ${BASH_REMATCH[2]} != "" ]]; then
        match_number=${BASH_REMATCH[2]}
        if [[ ${start_frame} == "" ]]; then
            start_frame=${match_number}
            digits=${#BASH_REMATCH[1]} + ${#BASH_REMATCH[2]}
            echo ${BASH_REMATCH[0]}
        fi
        if [[ ${start_frame} -gt  ${match_number} ]]; then    
             start_frame=${match_number}
             digits=${#BASH_REMATCH[1]} + ${#BASH_REMATCH[2]}
        fi
    fi
done

digits=$((${digits}))
digits=4
frame_match=$1"/image%0"${digits}"d.png"
echo  "Start frame is: " ${start_frame}
echo "digits: " ${digits}
echo "name of video file: " ${output_name}

ffmpeg -start_number ${start_frame} -framerate ${frame_rate} -i ${frame_match} -pix_fmt yuv420p ${output_name}