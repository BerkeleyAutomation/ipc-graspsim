#!/usr/bin/env bash

OBJ=$1		# e.g., cube_3cm
GRASP_IND=$2	# e.g., 0

CPU_IND=0

E_VAL=1e8
MU_VAL=0.4

IMAGE_NAME="ipc-graspsim"

# to do single run
sudo docker run --rm -it \
	-v $(pwd)/tools:/IPC_sim/tools/ \
	-v $(pwd)/output:/IPC_sim/output/ \
	-v $(pwd)/dexgrasp_data/:/IPC_sim/dexgrasp_data/ \
	--cpuset-cpus ${CPU_IND} -m 4g \
	${IMAGE_NAME} /bin/bash -c "PYOPENGL_PLATFORM=osmesa \
	python3 tools/grasping/phys_grasp_ipc.py ${OBJ} ${GRASP_IND}"
