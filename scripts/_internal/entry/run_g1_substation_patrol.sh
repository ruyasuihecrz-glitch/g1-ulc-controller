#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

LOOPS="${LOOPS:-0}"
PAUSE="${PAUSE:-0.2}"
# 默认蛇形展示路线：
#   1. 先走上侧主通道，展示长距离巡航；
#   2. 再从第 1/2 列柜之间下穿；
#   3. 继续从第 2/3 列柜之间上穿；
#   4. 最后走底侧通道回到左端。
# 右端收在 x=13.8，不去最右侧柜角附近掉头；速度提高后这样更稳。
POINTS="${POINTS:-0.8,0.0,0.0;-0.5,0.0,0.0;-0.5,5.6,0.0;6.3,5.6,0.0;6.3,1.9,0.0;10.7,1.9,0.0;10.7,5.6,0.0;13.8,5.6,0.0;10.7,5.6,0.0;10.7,-1.9,0.0;6.3,-1.9,0.0;6.3,-5.6,0.0;-0.5,-5.6,0.0;-0.5,0.0,0.0;0.8,0.0,0.0}"

env -u PYTHONHOME -u PYTHONPATH -u PYTHONEXECUTABLE bash --noprofile --norc -lc \
  'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 scripts/_internal/tasks/patrol_g1_nav2.py --points "$1" --loops "$2" --pause "$3" --wait_timeout 60 --continue_on_failure' \
  bash "${POINTS}" "${LOOPS}" "${PAUSE}"
