run:
	export PYTHONPATH=/home/ubuntu/tracker/tracker:$PYTHONPATH && \
	python src/main.py --process-file /opt/dlami/nvme/s3sync/GX011620.MP4

run_force_create_tracking:
	export PYTHONPATH=/home/ubuntu/tracker/tracker:$PYTHONPATH && \
	python src/main.py --force-create-tracking \
	--process-file /opt/dlami/nvme/s3sync/GX011620.MP4 \
	--tracking-frames-limit 0
