# Pediatric CXR Model

## Introduction

This is the final project for APS360 at the University of Toronto. The goal is to build a Pediatric Chest X-Ray Model to classify and localize thoracic disorders.

## Structure

The items in individual folders are described below.

- `checkpoints`: Contains model weights and model accuracies/loss for key checkpoints. Currently contain the best interim checkpoint.
- `colab_scripts`: Scripts that benefitted from Colab's high RAM and GPU are found here. They should be executed in Colab. Note that this requires the cleaned data to be zipped and uploaded to your google drive.
- `data_cleaned`: Output folder which contains cleaned images and annotation labels. Images are not included in the repository due to dataset agreement rules.
- `data_raw`: Folder which contains raw images and labels from each of the datasets used. Data is not included in the repository due to dataset agreement rules.
- `models`: Contains baseline models, primary architecture, and pytorch dataset definition. This repository is depracated. Please refer to the `colab_scripts` folder for latest updates.
- `other`: Contains miscellaneous items (mainly to generate visualizes for the course report).
- `processing`: Contains data formatting and augmentation scripts.
