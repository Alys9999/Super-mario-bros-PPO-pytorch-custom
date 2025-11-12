# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
# ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# RUN apt-get update -y && apt-get install -y libglib2.0-dev libsm6 libxext6 libxrender-dev freeglut3-dev ffmpeg



# Remove NVIDIA repos that cause the GPG error
# RUN rm -f /etc/apt/sources.list.d/*nvidia*.list /etc/apt/sources.list.d/*cuda*.list \
#  && apt-get update -y \
#  && apt-get install -y --no-install-recommends \
#       libglib2.0-dev libsm6 libxext6 libxrender-dev freeglut3-dev ffmpeg \
#  && rm -rf /var/lib/apt/lists/**




# RUN pip install gym-super-mario-bros==7.3.2 opencv-python==4.3.0.36 future==0.18.2 pyglet==1.5.7

# RUN pip install gym==0.21.0 nes-py==8.1.8 gym-super-mario-bros==7.3.2 numpy==1.24.4 pyglet==1.5.7 opencv-python==4.3.0.36 future==0.18.2


FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update -y && apt-get install -y \
    libglib2.0-dev libsm6 libxext6 libxrender-dev freeglut3-dev ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Keep numpy <1.24 and pyglet <=1.5.21 for nes-py; bump OpenCV to a Py3.10 wheel
RUN pip install --no-cache-dir "setuptools<67" "wheel<0.40" \
 && pip install --no-cache-dir \
    "gym==0.21.0" \
    "nes-py==8.2.1" \
    "gym-super-mario-bros==7.3.0" \
    "numpy==1.23.5" \
    "pyglet==1.5.21" \
    "opencv-python==4.7.0.72"

WORKDIR /Super-mario-bros-PPO-pytorch
