#!/bin/bash

export PYTHONPATH=/home/tim/devel/nn-202302/src

python -m nnexp.cmd.`basename $0` "$@"
