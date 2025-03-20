# Shared arguments
ARG USERNAME="vscode"
ARG USER_UID=1000
ARG USER_GID=$USER_UID


FROM gadgetron_cudabuild AS gadgetron_cudabuild_env
ARG USER_UID
USER ${USER_UID}
WORKDIR /opt

RUN mkdir -p /opt/code/lit_gadgetron
COPY --chown=$USER_UID:conda lit_gadgetron/ /opt/code/lit_gadgetron/
SHELL ["/bin/bash", "-c"]

# Update the conda env
RUN . /opt/conda/etc/profile.d/conda.sh && umask 0002 && conda activate gadgetron && umask 0002 && /opt/conda/bin/mamba \
env update --file /opt/code/lit_gadgetron/environment.yml

FROM gadgetron_cudabuild_env AS gadgetron_nhlbicudabuild
ARG USER_UID
USER ${USER_UID}
WORKDIR /opt

RUN mkdir -p /opt/code/lit_gadgetron
COPY --chown=$USER_UID:conda lit_gadgetron/ /opt/code/lit_gadgetron/
SHELL ["/bin/bash", "-c"]

RUN . /opt/conda/etc/profile.d/conda.sh && umask 0002 && conda activate gadgetron && sh -x && \
    cd /opt/code/lit_gadgetron && \
    mkdir build && \
    cd build && \
    cmake ../ -GNinja -DUSE_MKL=ON -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/package -DCMAKE_PREFIX_PATH=/opt/package && \
    ninja && \
    ninja install

RUN echo "LC_ALL=C" >> ${HOME}/.bashrc
RUN echo "unset LANGUAGE" >> ${HOME}/.bashrc

FROM gadgetron_cudabuild_env AS gadgetron_nhlbi_rt_cuda
ARG USER_UID
USER ${USER_UID}

RUN mkdir -p /opt/config_nnUNet
COPY --chown=$USER_UID:conda config_nnUNet/ /opt/config_nnUNet/

COPY --from=gadgetron_nhlbicudabuild --chown=$USER_UID:conda /opt/package /opt/conda/envs/gadgetron/
COPY --from=gadgetron_nhlbicudabuild --chown=$USER_UID:conda /opt/code/gadgetron/docker/entrypoint.sh /opt/

RUN chmod +x /opt/entrypoint.sh
RUN sudo mkdir -p /opt/integration-test && sudo chown ${USER_GID}:${USER_UID} /opt/integration-test
COPY --from=gadgetron_cudabuild --chown=$USER_UID:conda /opt/code/gadgetron/test/integration /opt/integration-test/

ENTRYPOINT [ "/tini", "--", "/opt/entrypoint.sh" ]