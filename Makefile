download_videos:
	aws s3 cp s3://niro-prv-assets/input/neharot/rananVsHrzl /tmp/neharot/rananVsHrzl --recursive
list_files:
	ls /tmp/neharot/rananVsHrzl | tr '\n' ' '

run_create_video:
	export PYTHONPATH=/home/ubuntu/tracker/tracker:$PYTHONPATH && \
	export LOG_LEVEL=INFO && \
	python src/main.py \
	--create-video \
	--attach-sound \
	--out-vid-len 10 \
	--start-frame 0 \
	--process-folder /tmp/natanya \
	--process-file GX011658.MP4 GX011659.MP4 GX011660.MP4 GX011661.MP4 GX011662.MP4 GX011663.MP4 GX011665.MP4 GX011666.MP4 GX011667.MP4 GX011668.MP4 GX011669.MP4 GX011670.MP4
	

run_create_tracking:
	export PYTHONPATH=/home/ubuntu/tracker/tracker:$PYTHONPATH && \
	python src/main.py \
	--force-create-tracking \
	--tracking-frames-limit 0 \
	--process-folder /tmp/neharot/rananVsHrzl \
	--process-file GX011658.MP4 GX011659.MP4 GX011660.MP4 GX011661.MP4 GX011662.MP4 GX011663.MP4 GX011665.MP4 GX011666.MP4 GX011667.MP4 GX011668.MP4 GX011669.MP4 GX011670.MP4


upload_tracking_data:
	aws s3 cp /tmp/neharot/rananVsHrzl/tracking s3://niro-prv-assets/input/neharot/rananVsHrzl/tracking --recursive

set_sound:
	ffmpeg -i /tmp/natanya/GX011645.MP4 -vn -acodec copy output_audio.aac
	ffmpeg -i output/video_GX011645_0.mp4 -i output_audio.aac -c:v copy -c:a aac output/video_GX011645_0_aud.mp4