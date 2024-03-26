run_create_video:
	export PYTHONPATH=/home/ubuntu/tracker/tracker:$PYTHONPATH && \
	python src/main.py \
	--process-file /Users/niro/dev/github/tracker/input/GX011620.MP4 \
	--create-video \
	--out-vid-len 10000 \
	--start-frame 5500


run_create_tracking:
	export PYTHONPATH=/home/ubuntu/tracker/tracker:$PYTHONPATH && \
	python src/main.py \
	--force-create-tracking \
	--process-file /opt/dlami/nvme/s3sync/GX011620.MP4 \
	--tracking-frames-limit 0
