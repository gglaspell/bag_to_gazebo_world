FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install numpy \
                open3d \
                Pillow \
                rosbags \
                scipy \
                tqdm

WORKDIR /app

COPY bag_to_gazebo_world.py /app/bag_to_gazebo_world.py
RUN chmod +x /app/bag_to_gazebo_world.py

ENTRYPOINT ["python3", "/app/bag_to_gazebo_world.py"]

