FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# OpenCV
RUN apt update
RUN apt -y install ffmpeg libsm6 libxext6

# non-root user
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
RUN mkdir /opt/algorithm
RUN chown -R algorithm:algorithm /opt/algorithm /opt/conda
USER algorithm

# conda
RUN conda update -c defaults -y conda

# pip requirements
WORKDIR /opt/algorithm
COPY --chown=algorithm:algorithm requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

# copy package and main file
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/conda/lib/
COPY --chown=algorithm:algorithm jawfrac/ jawfrac/
COPY --chown=algorithm:algorithm train_fractures.py .

# script to run
ENTRYPOINT ["python", "/opt/algorithm/train_fractures.py"]
