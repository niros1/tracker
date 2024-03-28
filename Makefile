run_create_video:
	export PYTHONPATH=/home/ubuntu/tracker/tracker:$PYTHONPATH && \
	export LOG_LEVEL=INFO && \
	python src/main.py \
	--create-video \
	--out-vid-len 300 \
	--start-frame 3000 \
	--process-file /Users/niro/dev/github/tracker/input/GX011611.MP4 \

# GX011644, 41, 

run_create_tracking:
	export PYTHONPATH=/home/ubuntu/tracker/tracker:$PYTHONPATH && \
	python src/main.py \
	--force-create-tracking \
	--tracking-frames-limit 0 \
	--process-folder /opt/dlami/nvme/s3sync/natanya \
	--process-file GX011642.MP4 GX011640.MP4 GX011643.MP4 GX011645.MP4

download_data:
	aws s3 cp s3://niro-prv-assets/input/natanya /opt/dlami/nvme/s3sync/natanya --recursive

upload_data:
	aws s3 cp output/natanya s3://niro-prv-assets/input/natanya/tracking --recursive
