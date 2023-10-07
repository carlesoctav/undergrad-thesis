install java and javac.

export XRT_TPU_CONFIG="localservice;0;localhost:51011"

RUN ON SUDO
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9