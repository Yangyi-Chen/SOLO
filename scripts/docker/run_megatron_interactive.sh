#!/bin/bash

DOCKER_IMG=xingyaoww/pt-megatron-llm:v1.2
WORK_DIR=`pwd`

# initialize GIDS
GIDS=($(id -G))

echo "${GIDS[@]}"

# Example of using GIDS in a for-loop
for gid in "${GIDS[@]}"; do
    echo "Group ID: $gid"
    # You can add other commands here that use $gid
done

docker run \
    -e UID=$(id -u) \
    -e GIDS="$(IFS=,; echo "${GIDS[*]}")" \
    -e WANDB_API_KEY \
    -e HUGGING_FACE_HUB_TOKEN \
    --gpus all -it \
    --rm \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$WORK_DIR:/workspace" \
    -v /scratch:/scratch \
    -v /shared:/shared \
    "$DOCKER_IMG" \
    /bin/bash -c '
        set -x
        useradd --shell /bin/bash -u $UID -o -c "" -m user
        IFS="," read -ra GIDS_ARRAY <<< "$GIDS"
        for gid in "${GIDS_ARRAY[@]}"; do
            groupadd -g $gid group$gid
            usermod -a -G $gid user
        done
        cd /workspace
        su user -c "git config --global credential.helper store"
        su user
    '
