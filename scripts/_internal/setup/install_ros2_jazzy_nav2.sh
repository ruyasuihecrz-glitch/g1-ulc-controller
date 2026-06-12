#!/usr/bin/env bash
set -euo pipefail

# Install ROS 2 Jazzy + Nav2 tooling for Ubuntu 24.04.
#
# This script follows the official ROS 2 Jazzy Debian-package installation flow
# and adds the packages needed by the Isaac Lab G1 cmd_vel bridge:
#   - rclpy / geometry_msgs / nav_msgs / tf2_ros / rosgraph_msgs
#   - Nav2 bringup
#   - slam_toolbox
#
# It intentionally does not install Unitree hardware ROS 2 packages. The Isaac Lab
# Nav2 demo only needs standard ROS 2 topics. Unitree hardware comms should be added
# later in a separate real-robot workspace or container.

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
INSTALL_DESKTOP="${INSTALL_DESKTOP:-false}"

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

if [[ ! -f /etc/os-release ]]; then
  echo "[ERROR] /etc/os-release not found."
  exit 1
fi

# shellcheck disable=SC1091
. /etc/os-release
if [[ "${VERSION_ID:-}" != "24.04" ]]; then
  echo "[ERROR] ROS 2 Jazzy deb packages target Ubuntu 24.04. Detected VERSION_ID=${VERSION_ID:-unknown}."
  exit 1
fi

echo "[INFO] Installing ROS 2 ${ROS_DISTRO} for Ubuntu ${VERSION_ID} (${VERSION_CODENAME:-unknown})"

${SUDO} apt update
${SUDO} apt install -y locales software-properties-common curl ca-certificates gnupg
${SUDO} locale-gen en_US en_US.UTF-8
${SUDO} update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

${SUDO} add-apt-repository -y universe

if ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
  ROS_APT_SOURCE_VERSION="${ROS_APT_SOURCE_VERSION:-1.2.0~${VERSION_CODENAME}}"
  ROS_APT_SOURCE_URL="${ROS_APT_SOURCE_URL:-https://repo.ros2.org/ubuntu/main/pool/main/r/ros-apt-source/ros2-apt-source_${ROS_APT_SOURCE_VERSION}_all.deb}"
  echo "[INFO] Installing ros2-apt-source from ${ROS_APT_SOURCE_URL}"
  curl --fail --location --connect-timeout 20 --retry 5 --retry-delay 5 \
    -o /tmp/ros2-apt-source.deb \
    "${ROS_APT_SOURCE_URL}"
  ${SUDO} dpkg -i /tmp/ros2-apt-source.deb
fi

${SUDO} apt update

if [[ "${INSTALL_DESKTOP,,}" == "true" || "${INSTALL_DESKTOP}" == "1" ]]; then
  ${SUDO} apt install -y ros-${ROS_DISTRO}-desktop
else
  ${SUDO} apt install -y ros-${ROS_DISTRO}-ros-base
fi

${SUDO} apt install -y \
  python-is-python3 \
  ros-${ROS_DISTRO}-navigation2 \
  ros-${ROS_DISTRO}-nav2-bringup \
  ros-${ROS_DISTRO}-slam-toolbox \
  ros-${ROS_DISTRO}-tf2-ros \
  ros-${ROS_DISTRO}-tf2-tools \
  ros-${ROS_DISTRO}-robot-state-publisher \
  ros-${ROS_DISTRO}-xacro \
  ros-${ROS_DISTRO}-rosgraph-msgs \
  ros-${ROS_DISTRO}-geometry-msgs \
  ros-${ROS_DISTRO}-nav-msgs \
  ros-${ROS_DISTRO}-sensor-msgs \
  ros-dev-tools

if ! grep -q "/opt/ros/${ROS_DISTRO}/setup.bash" "${HOME}/.bashrc"; then
  echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> "${HOME}/.bashrc"
fi

mkdir -p "${HOME}/.ros/log"

set +u
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

echo "[INFO] ROS 2 version:"
ros2 --help >/dev/null
ros2 pkg prefix rclpy
ros2 pkg prefix nav2_bringup
ros2 pkg prefix slam_toolbox

echo "[OK] ROS 2 ${ROS_DISTRO}, Nav2, and slam_toolbox installed."
echo "[NEXT] Open a new shell or run: source /opt/ros/${ROS_DISTRO}/setup.bash"
