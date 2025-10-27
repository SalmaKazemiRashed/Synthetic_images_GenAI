## A generative image project focused on high-content fluorescence microscopy, designed to generate full cell images using only the nuclei channel.

This project focuses on generating the cell channel of a fluorescence microscopy dataset from the nuclei channel to address a batch of missing data. The dataset is shared on Zenodo, with nuclei and cell data described at [nuclei](https://www.sciencedirect.com/science/article/pii/S2352340922009726) and [Cell](https://www.sciencedirect.com/science/article/pii/S2352340924011107).

For this project, several UNet and GAN models were trained with different hyperparameters. To improve UNet performance, we explored advanced loss functions, learning rate schedules, and deeper network architectures. UNets were primarily used for simple image-to-image (I2I) translation tasks.

Two UNet models are presented here, along with preprocessing code, model architectures, and training functions. Finally, predictions are visualized by loading the saved models.