# Shared arguments
ARG USERNAME="vscode"
ARG USER_UID=1000
ARG USER_GID=$USER_UID


FROM gadgetron_cudabuild AS gadgetron_cudabuild_env
ARG USER_UID
USER ${USER_UID}
WORKDIR /opt

RUN mkdir -p /opt/code/NHLBI-GT-Non-Cartesian
COPY --chown=$USER_UID:conda NHLBI-GT-Non-Cartesian/ /opt/code/NHLBI-GT-Non-Cartesian/
SHELL ["/bin/bash", "-c"]

# Update the conda env
RUN . /opt/conda/etc/profile.d/conda.sh && umask 0002 && conda activate gadgetron && umask 0002 && /opt/conda/bin/mamba \
env update --file /opt/code/NHLBI-GT-Non-Cartesian/environment.yml

COPY --chown=$USER_UID:conda OpticalFlow3d/ /opt/code/OpticalFlow3d/
RUN . /opt/conda/etc/profile.d/conda.sh && umask 0002 && conda activate gadgetron && sh -x && \
    cd /opt/code/OpticalFlow3d/ && \
    pip install -e . && \
    pip install numpy==1.23

FROM gadgetron_cudabuild_env AS gadgetron_nhlbicudabuild
ARG USER_UID
USER ${USER_UID}
WORKDIR /opt


RUN mkdir -p /opt/GIRF/
COPY --chown=$USER_UID:conda NHLBI-GT-Non-Cartesian/GIRF/ /opt/GIRF/


RUN mkdir -p /opt/GIRF/
COPY --chown=$USER_UID:conda NHLBI-GT-Non-Cartesian/GIRF/ /opt/GIRF/

RUN mkdir -p /opt/code/NHLBI-GT-Non-Cartesian
COPY --chown=$USER_UID:conda NHLBI-GT-Non-Cartesian/ /opt/code/NHLBI-GT-Non-Cartesian/
SHELL ["/bin/bash", "-c"]

RUN . /opt/conda/etc/profile.d/conda.sh && umask 0002 && conda activate gadgetron && sh -x && \
    cd /opt/code/NHLBI-GT-Non-Cartesian && \
    mkdir build && \
    cd build && \
    cmake ../ -GNinja -DUSE_MKL=ON -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/package -DCMAKE_PREFIX_PATH=/opt/package && \
    ninja && \
    ninja install


# Needed due to conflict b/w optical flow and cupy
#RUN /opt/conda/envs/gadgetron/bin/pip uninstall -y cupy-cuda113
#RUN /opt/conda/envs/gadgetron/bin/pip uninstall -y cupy-cuda11x
#RUN /opt/conda/envs/gadgetron/bin/pip	install cupy-cuda11x


RUN echo "LC_ALL=C" >> ${HOME}/.bashrc
RUN echo "unset LANGUAGE" >> ${HOME}/.bashrc

FROM gadgetron_cudabuild_env AS gadgetron_nhlbi_rt_cuda
ARG USER_UID
USER ${USER_UID}
RUN mkdir -p /opt/data/GIRF/
COPY --chown=$USER_UID:conda NHLBI-GT-Non-Cartesian/GIRF/ /opt/GIRF/
COPY --from=gadgetron_nhlbicudabuild --chown=$USER_UID:conda /opt/package /opt/conda/envs/gadgetron/
COPY --from=gadgetron_nhlbicudabuild --chown=$USER_UID:conda /opt/code/gadgetron/docker/entrypoint.sh /opt/
# Needed due to conflict b/w optical flow and cupy
#RUN /opt/conda/envs/gadgetron/bin/pip uninstall -y cupy-cuda113
#RUN /opt/conda/envs/gadgetron/bin/pip uninstall -y cupy-cuda11x
#RUN /opt/conda/envs/gadgetron/bin/pip   install cupy-cuda11x
#COPY --from=gadgetron_nhlbicudabuild --chown=$USER_UID:conda /opt/code/gadgetron/docker/set_matlab_paths.sh /opt/
RUN chmod +x /opt/entrypoint.sh
RUN sudo mkdir -p /opt/integration-test && sudo chown ${USER_GID}:${USER_UID} /opt/integration-test
COPY --from=gadgetron_cudabuild --chown=$USER_UID:conda /opt/code/gadgetron/test/integration /opt/integration-test/

ENTRYPOINT [ "/tini", "--", "/opt/entrypoint.sh" ]