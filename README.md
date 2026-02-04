# lit_gadgtron : Multi-Project Gadgetron repository

This repository hosts public facing code for gadgetron from the laboratory of imaging technology, NHLBI, NIH.
It contains multiple research projects. **Each branch corresponds to a distinct project and its associated journal article (publication).**  
Browse branches to access the code, data, and manuscript for each publication.

## Repository structure

- **Main branch:** Overview and instructions.
- **Project branches:** Each branch is named after a project/publication and contains:
  - Source code
  - Data (if applicable)

## Available projects

| Branch Name | Publication Title | Brief description| Article Link |
| :---: | :---: | :---: | :---: |
| [`cardiac_4D_imoco`](https://github.com/NHLBI/lit_gadgetron/tree/cardiac_4D_imoco)         | Isotropic 3D cardiac cine imaging at 0.55 T using stack-of-spiral sampling and four-dimensional iterative motion compensation (4D iMoCo) | 4D iterative motion compensated image reconstruction for cardiac resolved imaging using stack of spirals| [10.1016/j.jocmr.2026.102698](https://doi.org/10.1016/j.jocmr.2026.102698) |
| [`auto_venc_calibration`](https://github.com/NHLBI/lit_gadgetron/tree/auto_venc_calibration)    | Automatic velocity encoding (VENC) calibration for quantitative flow measurement to improve precision | Cartesian phase contrast image reconstruction with feedback messaging to automatically estimate  velocity encoding limit (VENC). | In preparation |
| [`cardiopulmonary_bstar`](https://github.com/NHLBI/lit_gadgetron/tree/cardiopulmonary_bstar)  | “Two-for-one”: 3D cardiac and pulmonary MR imaging from a single acquisition using bSTAR | Dual image reconstruction pipeline (cardiac-resolved whole heart and respiratory resolved lung) for bSTAR acquisition (3D radial sampling, Pulseq) | In preparation |

## How to access a project

1. Clone the repository:
   ```bash
   git clone git@github.com:NHLBI/lit_gadgetron.git
   cd lit_gadgetron

2. List all branches:
   ```bash
   git branch -a
3. Switch to a project branch (e.g. cardiopulmonary_bstar branch):
4. ```bash
   git checkout cardiopulmonary_bstar

## Related external Gadgetron projects

Below are links to former or related Gadgetron projects hosted in other repositories on the [NHLBI-MR github group](https://github.com/NHLBI-MR):

| External Repository | Publication Title | Brief description| Article Link |
| :---: | :---: | :---: | :---: |
| [`SNR-driven-flow`](https://github.com/NHLBI-MR/SNR-driven-flow)        | Inline automatic quality control of 2D phase-contrast flow MR imaging for subject-specific scan time adaptation (SNR-driven-flow) | 2D non-Cartesian phase contrast image reconstruction with feedback messaging for an automatic SNR-driven quality control| [10.1002/mrm.30083](https://doi.org/10.1002/mrm.30083) |
| [`icomoco`](https://github.com/NHLBI-MR/icomoco)        | Increasing the scan-efficiency of pulmonary imaging at 0.55 T using iterative concomitant field and motion-corrected reconstruction | Iterative concomitant field and motion-corrected image reconstruction for pulmonary stack of spiral imaging | [10.1002/mrm.30054](https://doi.org/10.1002/mrm.30054) |

> Please refer to each repository for details, documentation, and publications.


## Gadgetron Image Reconstruction Framework
The Gadgetron is an open source project for medical image reconstruction. <br>
Gadgetron documentation is available at [https://gadgetron.readthedocs.io](https://gadgetron.readthedocs.io)

## Citation
If you find any project useful in your research, please cite the corresponding publication (see corresponding project branch for citation details) and Gadgetron article:

Hansen MS, Sørensen TS. Gadgetron: An Open Source Framework for Medical Image Reconstruction. Magn Reson Med. 2013 Jun;69(6):1768-76.

