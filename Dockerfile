# ROS2 base image
FROM rwthika/ros2-cuda:kilted-desktop-full-v25.08 AS base

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Install dependencies with apt
RUN apt update && \
  DEBIAN_FRONTEND=noninteractive apt install -y keyboard-configuration && \
  apt install -y git \
  apt-utils \
  software-properties-common \
  desktop-file-utils \
  ros-dev-tools \
  python3-colcon-common-extensions \
  python3-pip \
  python3-numpy \
  python3-shapely \
  libpcap-dev \
  gnuplot \
  libboost-all-dev \
  libpcl-dev \
  libncurses5-dev libncursesw5-dev \
  ros-$ROS_DISTRO-yaml-cpp-vendor ros-$ROS_DISTRO-xacro ros-$ROS_DISTRO-foxglove-bridge \
  ros-$ROS_DISTRO-pcl-ros ros-$ROS_DISTRO-camera-info-manager ros-$ROS_DISTRO-diagnostic-updater \
  ros-$ROS_DISTRO-image-transport ros-$ROS_DISTRO-image-transport-plugins \
  pybind11-dev \
  ffmpeg

RUN pip install pyyaml \
  subprocess32 \
  numpy \
  scipy \
  matplotlib \
  numba \
  tqdm \
  rosnumpy \
  ruamel.yaml \
  panda3d \
  panda3d-gltf \
  panda3d-simplepbr

RUN apt update
RUN apt install -y \
    gdb \
    gdbserver \
    ros-$ROS_DISTRO-backward-ros \
    vim \ 
    tmux \
    htop \
    bash-completion \
    graphviz

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc
RUN echo "source /root/pacsim_ws/install/setup.bash" >> /root/.bashrc
RUN echo "export COLCON_DEFAULTS_FILE=/root/workspace/configs/colcon_config.yaml" >> /root/.bashrc

ENV SHELL /bin/bash

RUN pip install --ignore-installed gymnasium tensorboard tqdm termcolor tyro line_profiler rich
RUN pip install --ignore-installed torch torchvision torchmetrics
RUN pip install --ignore-installed matplotlib==3.7.0
RUN pip install "numpy<2"
RUN pip install onnx

COPY pacsim /root/pacsim_ws/src/pacsim
WORKDIR /root/pacsim_ws
RUN source /opt/ros/$ROS_DISTRO/setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

CMD ["/bin/bash"]
