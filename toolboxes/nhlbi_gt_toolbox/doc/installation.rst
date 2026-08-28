Installation instructions
=========================

We recommend two different ways of obtaining and running this project : using a conda environment and build it or using Docker container

Installing in conda environment
-------------------------------

First of all, you will install Gadgetron :

.. code-block:: console

    git clone -b cardiopulmonary_bstar git@github.com:NHLBI/lit_gadgetron.git
    cd gadgetron
    conda env create -f environment.yml
    conda activate gadgetron
    mkdir build && cd build && cmake ../ -GNinja -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DC_PREFIX_PATH=${CONDA_PREFIX} -DUSE_CUDA=ON -DUSE_MKL=ON
    ninja && ninja install
    
Once built, the package can be used with gadgetron using the config xml files provided with this repository (`config files repository <https://github.com/NHLBI/lit_gadgetron/tree/cardiopulmonary_bstar/toolboxes/nhlbi_gt_toolbox/config>`_).

Validate Gadgetron installation
+++++++++++++++++++++++++++++++
First, validate that the Gadgetron is installed and working.After activating the environment (with ``conda activate gadgetron``), the command ``gadgetron --info`` should give you information
about your installed version of the Gadgetron and it would look something like this::

    $ gadgetron --info
    Gadgetron Version Info
      -- Version            : 4.6.0
      -- Git SHA1           : NA
      -- System Memory size : 112705 MB
      -- Python Support     : YES
      -- Julia Support      : NO
      -- Matlab Support     : NO
      -- CUDA Support       : YES
      -- NVCC Flags         : -gencode arch=compute_70,code=sm_70;-gencode arch=compute_75,code=sm_75;-gencode arch=compute_80,code=sm_80;-gencode arch=compute_86,code=sm_86 --std=c++17
        * Number of CUDA capable devices: 1
          - Device 0: Tesla P100-PCIE-16GB
             + CUDA Driver Version / Runtime Version: 11.6/11.6
             + CUDA Capability Major / Minor version number: 6.0
             + Total amount of global GPU memory: 16280 MB

The output may vary on your specific setup, but you will see error messages if the Gadgetron is not installed or not installed correctly.

Validate image reconstruction pipelines
+++++++++++++++++++++++++++++++++++++++
To validate that the Gadgetron is working correctly with the NHLBI toolbox, you first need to download the test data using the following command:

.. code-block:: console

    conda activate gadgetron
    python test/nhlbi_integration_tests/get_nhlbi_data.py download

Then, you can run the following command to test the bSTAR pulmonary image reconstruction pipeline for example:

.. code-block:: console

    cd test/nhlbi_integration_tests/
    python run_nhlbi_tests.py cases/mocolr_bSTAR.cfg -F

The expected output of the test should look like this::

    Downloading test data...
    Verified: /opt/code/gadgetron/test/nhlbi_integration_tests/data/mocolr_bSTAR/bstar_450mm_1.20mm_true0_TR2.32ms_rf200_i110_67k_FA40_WASP_self0_fid0_noise0.seq
    Verified: /opt/code/gadgetron/test/nhlbi_integration_tests/data/mocolr_bSTAR/noise_data.h5
    Verified: /opt/code/gadgetron/test/nhlbi_integration_tests/data/mocolr_bSTAR/baseline_output.h5
    Verified: /opt/code/gadgetron/test/nhlbi_integration_tests/data/mocolr_bSTAR/traj_bstar_450mm_1.20mm_true0_TR2.32ms_rf200_i110_67k_FA40_WASP_self0_fid0_noise0.h5
    Verified: /opt/code/gadgetron/test/nhlbi_integration_tests/data/mocolr_bSTAR/recon_data.h5
    Querying Gadgetron capabilities...

    Test 1 of 1: cases/mocolr_bSTAR.cfg

    Running Gadgetron test cases/mocolr_bSTAR.cfg with:
    -- ISMRMRD_HOME    : None
    -- GADGETRON_HOME  : None
    -- TEST CASE       : cases/mocolr_bSTAR.cfg
    Starting MRD Storage Server on port 9113
    Starting Gadgetron instance on port 9003
    Copying prepared ISMRMRD data: /opt/code/gadgetron/test/nhlbi_integration_tests/data/mocolr_bSTAR/noise_data.h5 -> test/dependency.siemens.copied.mrd
    Passing data to Gadgetron: test/dependency.siemens.copied.mrd -> test/dependency.client.output.mrd
    Gadgetron processing time: 0.13 s
    Copying prepared ISMRMRD data: /opt/code/gadgetron/test/nhlbi_integration_tests/data/mocolr_bSTAR/recon_data.h5 -> test/reconstruction.siemens.copied.mrd
    Passing data to Gadgetron: test/reconstruction.siemens.copied.mrd -> test/reconstruction.client.output.mrd
    Gadgetron processing time: 378.75 s
    reconstruction.test.1      [OK] (Norm: 3.5e-05 [0.01] Scale: 6.0e-08 [0.01])
    reconstruction.test.1      [OK] (Output headers matched reference)
    reconstruction.test.2      [OK] (Norm: 7.3e-04 [0.01] Scale: 3.0e-05 [0.01])
    reconstruction.test.2      [OK] (Output headers matched reference)
    Test status: Passed
    SPEED REGRESSION: 378.9s vs baseline 179.6s (+111.0%, threshold 50%)

    Speed regressions:
            cases/mocolr_bSTAR.cfg

    1 tests passed. 0 tests failed. 0 tests skipped. 0 missing baselines. 1 speed regressions.
    Total processing time: 378.88 seconds.


Docker container 
----------------

Alternatively, you can test the code by pulling the provided docker image located in `packages repository <https://github.com/NHLBI/lit_gadgetron/pkgs/container/litgt_cardio_pulmonary_bstar_rt>`_ using the following command:

.. code-block:: console

    docker pull ghcr.io/nhlbi/litgt_cardio_pulmonary_bstar_rt:20260205


This image can be deployed with: 

.. code-block:: console

    docker run --gpus all  --name=cardio_pulmonary_bstar_rt -ti -p 9063:9002 --volume=[LOCAL_DATA_FOLDER]:/opt/data --restart unless-stopped --detach ghcr.io/nhlbi/litgt_cardio_pulmonary_bstar_rt:20260205`

where **LOCAL_DATA_FOLDER** is the path to a folder containing raw data that can be used for testing the reconstruction. 

Test the code 
-------------

Once the docker container is running, you can start a bash terminal inside the container using: 

.. code-block:: console

    docker exec -ti cardio_pulmonary_bstar_rt bash 

and you can simply validate the image reconstruction pipeline using our integration tests (See precedent paragraph) or you can navigate to `/opt/data/` and test the code using the following command:

.. code-block:: console

    cd /opt/data
    gadgetron_ismrmrd_client -p 9002 -f noise/noise_Freemax_XL_NIH_2025-03-06-112829_FID016823_bstar_1_20mm_FA40_FOV450.h5 -c default_measurement_dependencies.xml
    gadgetron_ismrmrd_client -p 9002 -f h5/Freemax_XL_NIH_2025-03-06-112829_FID016823_bstar_1_20mm_FA40_FOV450.h5 -c pulmonary_echo0.xml -o OUTPUT_FILENAME.h5


In another terminal session you can monitor the logs from the container 

.. code-block:: console

    docker logs -f cardio_pulmonary_bstar_rt


Please note that if you are using the gadgetron_ismrmrd_client from outside the container then you may need to specify the server address with **-a SERVER_ADDRESS** and the port **-p 9063**

.. code-block:: console

    cd LOCAL_DATA_FOLDER
    gadgetron_ismrmrd_client -a SERVER_ADDRESS -p 9063  -f noise/noise_Freemax_XL_NIH_2025-03-06-112829_FID016823_bstar_1_20mm_FA40_FOV450.h5 -c default_measurement_dependencies.xml
    gadgetron_ismrmrd_client -a SERVER_ADDRESS -p 9063  -f h5/Freemax_XL_NIH_2025-03-06-112829_FID016823_bstar_1_20mm_FA40_FOV450.h5 -c pulmonary_echo0.xml -o OUTPUT_FILENAME.h5

Dataset
-------

The test data can also be downloaded from zenodo: `18461603 <https://zenodo.org/records/18461603>`_

.. note::

    More Information on Gadgetron are available over here : 
    `Gadgetron repository <https://gadgetron.readthedocs.io/en/latest/obtaining.html>`_ and `Gadgetron documentation <https://github.com/gadgetron/gadgetron>`_
