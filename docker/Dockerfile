ARG ISAACLAB_BASE_IMAGE_ARG=isaac-lab-base

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG=/workspace/isaaclab_extension_template
ENV DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV ROS_SETUP=/opt/ros/jazzy/setup.bash
ENV ROS_HOME=/workspace/.ros
ENV ROS_LOG_DIR=/workspace/.ros/log

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY ../ ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

# Install ROS 2 Jazzy + Nav2 in the image so container restarts stay reproducible.
RUN bash ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/scripts/install_ros2_jazzy_nav2.sh && \
    apt install -y python-is-python3 && \
    mkdir -p /workspace/.ros/log && \
    { grep -qxF "source /opt/ros/jazzy/setup.bash" /root/.bashrc || echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc; }

# # Install whatever you need as additional dependencies.
RUN bash -i -c "source ${HOME}/.bashrc && \
    cd ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/unitree_rl_lab && \
    pip install -e ."

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR /workspace
