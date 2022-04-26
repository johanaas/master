# Master project

This repository is the master project of [Kim Midtlid](https://github.com/kamidtli) and [Johannes Åsheim](https://github.com/johanaas). 

## Setup

HSJA attack requires Tensorflow 1, so the project was installed on Windows 10 in a conda environment.

Prereqiuesites:
1. Download [Miniconda Installer](https://docs.conda.io/en/latest/miniconda.html)
2. Download the [ImageNet validation dataset](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php). First you'll need to create a user. Select  **Validation images (all tasks)**. This takes a loong time.
---
3. Clone this repo with `git clone` to a folder of your choice.
4. Start the miniconda prompt from the start menu, not the standard cmd prompt.
5. Create a conda environment with python version 3.6 `conda create --name [ENV_NAME] python=3.6`
6. Press `y` when asked.
7. Activate the conda environment `conda activate [ENV_NAME]`
8. Navigate to the cloned repo folder.
9. Install the requirements with `pip install -r requirements.txt`
10. Update the `IMAGENET_PATH` in `config.py` to where you downloaded the ImageNet in _step 2_.
11. Run `python main.py`
