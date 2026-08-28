Installation instructions
=========================

We recommend two different ways of obtaining and running this project : using a conda environment and build it or using Docker container

Installing in conda environment
-------------------------------

First of all, you will install Gadgetron :

.. code-block:: console

    git clone -b volumetric_rt_mri git@github.com:NHLBI/lit_gadgetron.git
    cd gadgetron
    conda env create -f environment.yml
    conda activate gadgetron
    mkdir build && cd build && cmake ../ -GNinja -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DC_PREFIX_PATH=${CONDA_PREFIX} -DUSE_CUDA=ON -DUSE_MKL=ON
    ninja && ninja install
    
Once built, the package can be used with gadgetron using the config xml files provided with this repository (`config files repository <https://github.com/NHLBI/lit_gadgetron/tree/volumetric_rt_mri/toolboxes/nhlbi_gt_toolbox/config>`_).

Validate installation
+++++++++++++++++++++
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
To validate that the Gadgetron is working correctly with the NHLBI toolbox, you can run the following command to test the prescan reference image reconstruction pipeline for example:

.. code-block:: console

    conda activate gadgetron
    cd test/nhlbi_integration_tests/
    python run_nhlbi_tests.py cases/csm_prescan.cfg -F

The expected output of the test should look like this::

    Downloading test data...
    Downloading: csm_prescan/noise_data.h5
    Downloading: csm_prescan/recon_data.h5
    Downloading: csm_prescan/baseline_output.h5
    Saved: /opt/code/gadgetron/test/nhlbi_integration_tests/data/csm_prescan/baseline_output.h5
    Saved: /opt/code/gadgetron/test/nhlbi_integration_tests/data/csm_prescan/noise_data.h5
    Saved: /opt/code/gadgetron/test/nhlbi_integration_tests/data/csm_prescan/recon_data.h5
    Querying Gadgetron capabilities...

    Test 1 of 1: cases/csm_prescan.cfg

    Running Gadgetron test cases/csm_prescan.cfg with:
    -- ISMRMRD_HOME    : None
    -- GADGETRON_HOME  : None
    -- TEST CASE       : cases/csm_prescan.cfg
    Starting MRD Storage Server on port 9113
    Starting Gadgetron instance on port 9003
    Copying prepared ISMRMRD data: /opt/code/gadgetron/test/nhlbi_integration_tests/data/csm_prescan/noise_data.h5 -> test/dependency.siemens.copied.mrd
    Passing data to Gadgetron: test/dependency.siemens.copied.mrd -> test/dependency.client.output.mrd
    Gadgetron processing time: 0.12 s
    Copying prepared ISMRMRD data: /opt/code/gadgetron/test/nhlbi_integration_tests/data/csm_prescan/recon_data.h5 -> test/reconstruction.siemens.copied.mrd
    Passing data to Gadgetron: test/reconstruction.siemens.copied.mrd -> test/reconstruction.client.output.mrd
    Gadgetron processing time: 24.14 s
    reconstruction.test        [OK] (Norm: 9.2e-06 [0.01] Scale: 0.0e+00 [0.01])
    reconstruction.test        [OK] (Output headers matched reference)
    Test status: Passed
    Speed improved: 24.3s vs baseline 56.2s (-56.8%)

    1 tests passed. 0 tests failed. 0 tests skipped. 0 missing baselines. 0 speed regressions.
    Total processing time: 24.26 seconds.

Or the 3D real-time image reconstruction pipeline:

.. code-block:: console

    cd test/nhlbi_integration_tests/
    python run_nhlbi_tests.py cases/3drt.cfg -F

The expected output of the test should look like this::

    Downloading test data...
    Downloading: 3drt/noise_data.h5
    Downloading: 3drt/recon_data.h5
    Downloading: 3drt/prescan-reference_noise.h5
    Downloading: 3drt/prescan-reference.h5
    Saved: /opt/code/gadgetron/test/nhlbi_integration_tests/data/3drt/prescan-reference_noise.h5
    Downloading: 3drt/baseline_output.h5
    Saved: /opt/code/gadgetron/test/nhlbi_integration_tests/data/3drt/noise_data.h5
    Saved: /opt/code/gadgetron/test/nhlbi_integration_tests/data/3drt/recon_data.h5
    Saved: /opt/code/gadgetron/test/nhlbi_integration_tests/data/3drt/prescan-reference.h5
    Warning: Could not download data for test '3drt': Downloaded file /opt/code/gadgetron/test/nhlbi_integration_tests/data/3drt/baseline_output.h5 failed validation. Expected SHA256 04e6d61654c5040e9589cb1ef02019e445990ef72c4381e151f2342e95ba4542. Actual SHA256 48011834c0279b9f4ef82a92b967015dff65ead9c3eba26b8ed081540a88b47c
    Querying Gadgetron capabilities...

    Test 1 of 1: cases/3drt.cfg

    Running Gadgetron test cases/3drt.cfg with:
    -- ISMRMRD_HOME    : None
    -- GADGETRON_HOME  : None
    -- TEST CASE       : cases/3drt.cfg
    Starting MRD Storage Server on port 9113
    Starting Gadgetron instance on port 9003
    Copying prepared ISMRMRD data: /opt/code/gadgetron/test/nhlbi_integration_tests/data/3drt/noise_data.h5 -> test/dependency.siemens.copied.mrd
    Passing data to Gadgetron: test/dependency.siemens.copied.mrd -> test/dependency.client.output.mrd
    Gadgetron processing time: 0.12 s
    Copying prepared ISMRMRD data: /opt/code/gadgetron/test/nhlbi_integration_tests/data/3drt/prescan-reference_noise.h5 -> test/dependency.siemens.1.copied.mrd
    Passing data to Gadgetron: test/dependency.siemens.1.copied.mrd -> test/dependency.client.1.output.mrd
    Gadgetron processing time: 0.12 s
    Copying prepared ISMRMRD data: /opt/code/gadgetron/test/nhlbi_integration_tests/data/3drt/prescan-reference.h5 -> test/dependency.siemens.2.copied.mrd
    Passing data to Gadgetron: test/dependency.siemens.2.copied.mrd -> test/dependency.client.2.output.mrd
    Gadgetron processing time: 31.04 s
    Copying prepared ISMRMRD data: /opt/code/gadgetron/test/nhlbi_integration_tests/data/3drt/recon_data.h5 -> test/reconstruction.siemens.copied.mrd
    Passing data to Gadgetron: test/reconstruction.siemens.copied.mrd -> test/reconstruction.client.output.mrd
    Gadgetron processing time: 41.94 s
    reconstruction.test        [OK] (Norm: 3.0e-03 [0.01] Scale: 7.3e-06 [0.01])
    reconstruction.test        [OK] (Output headers matched reference)
    Test status: Passed
    SPEED REGRESSION: 73.2s vs baseline 42.3s (+73.1%, threshold 50%)

    Speed regressions:
            cases/3drt.cfg

    1 tests passed. 0 tests failed. 0 tests skipped. 0 missing baselines. 1 speed regressions.
    Total processing time: 73.22 seconds.

Docker container 
----------------

Alternatively, you can test the code by pulling the provided docker image located in `packages repository <https://github.com/NHLBI/lit_gadgetron/pkgs/container/litgt_volumetric_rt_mri_rt>`_ using the following command:

.. code-block:: console

    docker pull ghcr.io/nhlbi/litgt_volumetric_rt_mri_rt:20260828


This image can be deployed with: 

.. code-block:: console

    docker run --gpus all --name=volumetric_rt_mri -ti -p 9063:9002 -p 9072:9004 --volume=[LOCAL_DATA_FOLDER]:/opt/data --restart unless-stopped --detach ghcr.io/nhlbi/litgt_volumetric_rt_mri_rt:20260828`

where **LOCAL_DATA_FOLDER** is the path to a folder containing raw data that can be used for testing the reconstruction. 

Test the code 
-------------

Once the docker container is running, you can start a bash terminal inside the container using: 

.. code-block:: console

    docker exec -ti volumetric_rt_mri bash 

and you can simply validate the image reconstruction pipeline using our integration tests (See precedent paragraph) or you can navigate to `/opt/data/` and test the code for the prescan reference image reconstruction pipeline for example:

.. code-block:: console

    cd /opt/data
    gadgetron_ismrmrd_client -p 9002 -f prescan-reference_noise.h5 -c default_measurement_dependencies.xml
    gadgetron_ismrmrd_client -p 9002 -f prescan-reference.h5 -c spiral_3DRT_CSM_test.xml -o OUTPUT_FILENAME.h5 


In another terminal session you can monitor the logs from the container 

.. code-block:: console

    docker logs -f volumetric_rt_mri


Please note that if you are using the gadgetron_ismrmrd_client from outside the container then you may need to specify the server address with **-a SERVER_ADDRESS** and the port **-p 9063**

.. code-block:: console

    cd LOCAL_DATA_FOLDER
    gadgetron_ismrmrd_client -a SERVER_ADDRESS -p 9063 -f prescan-reference_noise.h5 -c default_measurement_dependencies.xml
    gadgetron_ismrmrd_client -a SERVER_ADDRESS -p 9063 -f prescan-reference.h5 -c spiral_3DRT_CSM_test.xml -o OUTPUT_FILENAME.h5 


3D Slicer Integration
---------------------

If the SlicerGadget is enabled, `3D Slicer <https://slicer.readthedocs.io/en/latest/index.html>`_ with the `SlicerIGT <https://www.slicerigt.org/wp/>`_ extension can be used for live data streaming. In order to use, setup SlicerIGT on the host device with the corresponding forwarded port to the container (ex. 9072) in the OpenIGTLinkIF module, as shown in the picture below.

.. image:: igt.jpg
   :width: 50%

Dataset
-------

The test data can be downloaded from zenodo: `19005977 <https://zenodo.org/records/19005977>`_

.. note::
    More Information on Gadgetron are available over here : 
    `Gadgetron repository <https://gadgetron.readthedocs.io/en/latest/obtaining.html>`_ and `Gadgetron documentation <https://github.com/gadgetron/gadgetron>`_
