Installation instructions
=========================

We recommend to different ways of obtaining and running this project : using a conda environment and build it or using Docker container

Installing in conda environment
-------------------------------

First of all, you will need to install Gadgetron. Information are available over here : 

`Gadgetron repository <https://gadgetron.readthedocs.io/en/latest/obtaining.html>`_ and `Gadgetron documentation <https://github.com/gadgetron/gadgetron>`_

.. code-block:: console

    mamba activate gadgetron
    mamba env update -f environment.yml

The code can then be built 

.. code-block:: console

    mkdir build && cd build && cmake ../ -GNinja -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DC_PREFIX_PATH=${CONDA_PREFIX} -DUSE_CUDA=ON -DUSE_MKL=ON


Once built, the package can be used with gadgetron using the config xml files provided with this repository (`config files repository <https://github.com/NHLBI/lit_gadgetron/tree/auto_venc_calibration/config>`_).

After activating the environment (with conda activate gadgetron), you should be able to check that everything is working with ``gadgetron --info`` and ``gadgetron_ismrmrd_client_feedback -h``

Docker container 
----------------

Alternatively, you can test the code by pulling the provided docker image using the following command:

.. code-block:: console

    docker pull gadgetronnhlbi/ubuntu_2204_cuda124_lit:auto_venc_calibration_rt


This image can be deployed with: 

.. code-block:: console

    docker run --gpus all  --name=deploy_rt -ti -p 9063:9002 --volume=[LOCAL_DATA_FOLDER]:/opt/data --restart unless-stopped --detach gadgetronnhlbi/ubuntu_2204_cuda124_lit:auto_venc_calibration_rt`

where **LOCAL_DATA_FOLDER** is the path to a folder containing raw data that can be used for testing the reconstruction. 

Test the code 
-------------

Once the docker container is running, you can start a bash terminal inside the container using: 

.. code-block:: console

    docker exec -ti deplot_rt bash 

and you can simply ou can simply navigate to `/opt/data/` and test the code :

.. code-block:: console

    cd /opt/data
    gadgetron_ismrmrd_client_feedback -p 9002 -f DATA_FILE -c spiral_2d_AO.xml -o OUTPUT_FILENAME.h5` 


In another terminal session you can monitor the logs from the container 

.. code-block:: console

    docker logs -f deploy_rt`


Please note that if you are using the gadgetron_ismrmrd_client_feedback from outside the container then you may need to specify the server address with **-a SERVER_ADDRESS** and the port **-p 9063**

.. code-block:: console

    cd LOCAL_DATA_FOLDER
    gadgetron_ismrmrd_client_feedback -a SERVER_ADDRESS -p 9063 -f DATA_FILE -c spiral_2d_AO.xml -o OUTPUT_FILENAME.h5` 


.. warning::

    gadgetron_ismrmrd_client will not work with our `config files repository <https://github.com/NHLBI-MR/SNR-driven-flow/tree/main/config>`_ 
    because the gadget **ImageSNRExtractionAndFeedback** is sending 
    a Feedback message which MessageID is unknown for Gadgetron alone.

Dataset
-------

The test data can be downloaded from zenodo: 10.5281/zenodo.10525047
