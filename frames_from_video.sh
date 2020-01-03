#!/bin/bash
ffmpeg -i $1 $2/image%04d.png -hide_banner -vsync vfr
touch $2/timestamps.json
ffprobe -v quiet -f lavfi -print_format json -i "movie=$1" -show_frames -show_entries frame=pkt_pts_time,pkt_dts_time > $2timestamps.json

