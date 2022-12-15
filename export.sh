#!/usr/bin/env bash

. ./build.sh

docker save jawfracnet_processing | gzip -c > jawfracnet_v0.10.tar.gz
