run_create_video:
	export PYTHONPATH=/home/ubuntu/tracker/tracker:$PYTHONPATH && \
	export LOG_LEVEL=INFO && \
	python src/main.py \
	--process-file /Users/niro/dev/github/tracker/input/GX011611.MP4 \
	--create-video \
	--out-vid-len 300 \
	--start-frame 3000

# GX011644, 41, 

run_create_tracking:
	export PYTHONPATH=/home/ubuntu/tracker/tracker:$PYTHONPATH && \
	python src/main.py \
	--force-create-tracking \
	--process-file /opt/dlami/nvme/s3sync/natanya/GX011638.MP4 \
	--tracking-frames-limit 0

sync_data:
	 aws s3 cp s3://niro-prv-assets/input/natanya /opt/dlami/nvme/s3sync/natanya --recursive