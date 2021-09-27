#!/usr/bin/env bash
set -e
set -u
SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/.."

sudo docker build -t ipc-graspsim -f docker/Dockerfile .
