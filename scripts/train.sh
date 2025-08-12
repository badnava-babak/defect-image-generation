#!/bin/sh
python -m torch.distributed.launch \
--nproc_per_node=1 \
/home/b502b586/workspaces/SiemensEnergy/main.py --distributed True
