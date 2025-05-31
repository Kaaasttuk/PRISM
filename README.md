# PRISM: Prescreening Rapid Identification of Single-cell Genotypes in Microfluidic Devices

This repository contains the code and related files for the IIB Project in the Department of Engineering, University of Cambridge. This project aims to develop a workflow for multiplexed genotype experiments in the Mother Machine using a novel chemical barcoding method, PRISM (Prescreening Rapid Identification of Single-cell Genotypes in Microfluidic Devices).

## Project Overview

The PRISM workflow involves two main components that were developed and optimised:
1.  **Wet-Lab Protocol:** For generating distinct fluorescence barcode images from cells. (Experimental procedures are detailed in the main project report).
2.  **Image Analysis Pipeline:** An automated pipeline to decode each cell’s barcode from microscope images and subsequently track dynamic phenotypes.

This repository primarily houses the code for the image analysis pipeline, synthetic data generation, and any analysis scripts used to produce results for the report.

## Directory Structure and File Guide

The repository is organised as follows. Key files and their purposes are highlighted:

* **`./` (Root Directory)**
    * `README.md`: This file – providing an overview and guide to the repository.
    * `requirements.txt`: A list of Python packages required to run the scripts and notebooks in this project. Users should install these dependencies, preferably in a virtual environment.
        ```bash
        pip install -r requirements.txt
        ```
    * `utils.py`: A Python script containing utility functions mainly for synthetic fluorescent image generation.
    * `nd2-extractor.ipynb`: Jupyter Notebook used for extracting meta data and png images from `.nd2` microscope image files (a common format for Nikon microscope data).
    * `Omni_cell_segment.ipynb`: Jupyter Notebook used to experiment with cell segmentation model Omnipose, and contains analysis of growth rate comparison between barcoded cells and non-barcoded cells, and dye leakage experiments.
    * `Report_Figure.ipynb`: Jupyter Notebook used to generate some of the figures presented in the final project report.

* **`Omnipose_train/`**: This directory contains files related to training the Omnipose cell segmentation model.
    * `Omnipose_training_gen.ipynb`: Main Jupyter Notebook for training the Omnipose model .
    * `PC_Gen.ipynb`: Jupyter Notebook for generating synthetic Phase Contrast (PC) images using SyMBac as a training data.
    * `empty_tr.ipynb`: Jupyter Notebook for including empty trenches from real images in the training set.
    * `recycle.ipynb`: Jupyter Notebook to include well-segmented real images in the refined training set.

* **`PRISM_model/`**: This directory contains files for training and testing the PRISM barcode classification model.
    * `TrainPRISM.ipynb`: Jupyter Notebook for training the PRISM Convolutional Neural Network (CNN) model to classify barcode patterns.
    * `make_train_data.ipynb`: Jupyter Notebook used for inspecting, preparing and structuring the training dataset for the PRISM CNN.

* **`Synthetic_Fluorescent_Image/`**: This directory contains scripts and notebooks for generating synthetic fluorescent images used for training the PRISM CNN.
    * `cell.py`: Python script for cell model.
    * `dataGenerator.py`: Python script responsible for the core logic of generating synthetic image data.
    * `microscope.py`: Python script simulating microscope optics for realistic synthetic image generation.
    * `synthetic_image_gen.ipynb`: Jupyter Notebook for generating the synthetic fluorescent image similar to user's real data.

* **`Trench_Extraction/`**: This directory contains files related to the trench extraction stage of the image analysis pipeline.
    * `PyMMM.ipynb`: Jupyter Notebook implementing the trench localisation using the Python Mother Machine Manager (PyMMM) approach for single Z-plane images.
    * `PyMMM_z_stack.ipynb`: Jupyter Notebook, similar to `PyMMM.ipynb`, but adapted for handling image stacks with multiple Z-planes (focal planes) and selecting the best-focused image.
    * **`PyMMM_main/`**: Subdirectory containing the core Python scripts for PyMMM.

* **`growth_rate/`**: This directory contains data extracted in Omni_cell_segment.ipynb for cell growth rate analysis.

* **`leakage_exp/`**: This directory contains data from dye leakage control experiments for membrane dyes that had to be manually collected. Nucleoid dye leakage experiment is dealt with Omni_cell_segment.ipynb.

## For Readers

* The main project report should be consulted for the detailed scientific background, methodology, results, and discussion.
* Raw experimental data (e.g., large `.nd2` files) are typically not stored directly in Git repositories due to size. Please contact the Bakshi Lab.
* path in each file must be reconfigured in your own local computer.
* This repository provides the supporting code. Key stages to review might include:
    * Trench extraction logic (in `Trench_Extraction/`).
    * Cell segmentation model training and application (in `Omnipose_train/` and `Omni_cell_segment.ipynb`).
    * Barcode classification model training and data preparation (in `PRISM_model/`).
    * Synthetic data generation, which supports model training (in `Synthetic_Fluorescent_Image/`).


## Contact

Katsutaka Suzuki

