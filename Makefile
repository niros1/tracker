run_create_video:
	export PYTHONPATH=/home/ubuntu/tracker/tracker:$PYTHONPATH && \
	export LOG_LEVEL=INFO && \
	python src/main.py \
	--create-video \
	--attach-sound \
	--out-vid-len 10 \
	--start-frame 0 \
	--process-folder output \
	--process-file GX011620.MP4





# GX011644, 41,

run_create_tracking:
	export PYTHONPATH=/home/ubuntu/tracker/tracker:$PYTHONPATH && \
	python src/main.py \
	--force-create-tracking \
	--tracking-frames-limit 0 \
	--process-folder /opt/dlami/nvme/s3sync/natanya \
	--process-file GX011642.MP4 GX011640.MP4 GX011643.MP4 GX011645.MP4

download_videos:
	aws s3 cp s3://niro-prv-assets/input/natanya /tmp/natanya --recursive

upload_tracking_data:
	aws s3 cp output/natanya s3://niro-prv-assets/input/natanya/tracking --recursive

set_sound:
	ffmpeg -i /tmp/natanya/GX011645.MP4 -vn -acodec copy output_audio.aac
	ffmpeg -i output/video_GX011645_0.mp4 -i output_audio.aac -c:v copy -c:a aac output/video_GX011645_0_aud.mp4