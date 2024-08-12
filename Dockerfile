ARG IMAGE_NAME=python:3.9

FROM $IMAGE_NAME

# make system capable of building python
RUN apt update
RUN apt install -y ninja-build # need ninja for torch to compile the extension
RUN apt install -y python3-pip
RUN apt install -y libreadline-dev 
RUN apt install -y libssh-dev 
RUN apt install -y libbz2-dev 
RUN apt install -y liblzma-dev 
RUN apt install -y libsqlite3-dev

# pyenv 
RUN curl https://pyenv.run | bash
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PATH:$PYENV_ROOT/bin"
ENV PATH="$PYENV_ROOT/shims:$PATH"
RUN for pver in 3.9 3.10 3.11 3.12; do pyenv install $pver \
  && pyenv global $pver \
  && pip install -U pip uv; \
  done


# copy the actual app
WORKDIR /app
COPY . .

ARG JAX_VERSION
ARG JAX_OPT=cpu

ENV JAX_VERSION=$JAX_VERSION JAX_OPT=$JAX_OPT JAX_ENABLE_X64="True" 
CMD ["bash", "./scripts/_install_and_test.sh"]
