# MCIT: Multi-level cross-modal interactive transformer for RGBT tracking
The paper was accepted by the Neurocomputing.

## Citation
If our work is useful for your research, please consider citing:

```Bibtex
@article{MCIT,
title = {MCIT: Multi-level cross-modal interactive transformer for RGBT tracking},
journal = {Neurocomputing},
volume = {649},
pages = {130758},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2025.130758},
url = {https://www.sciencedirect.com/science/article/pii/S0925231225014304},
author = {Yu Qin and Jianming Zhang and Shimeng Fan and Zikang Liu and Jin Wang},
}
```

## Install the environment
Install virtual environment and dependency packages.
```bash
conda create -n MCIT python=3.7
conda activate MCIT
pip install -r requirements.txt
```

Create the default environment setting files.
```
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

Then set the paths of the project and dataset in "ltr/admin/local.py" and "pytracking/evaluation/local.py".

## Training
Set the training parameters in  "ltr/train_settings/MCIT/MCIT_settings.py".

Then run:
```
python ltr/run_training.py
```

## Testing
Set the model weight path in "pytracing/parameter/MCIT/MCIT.py".

Then run:
```
python pytracking/run_tracker.py --dataset_name rgbt234
```
## Tracking results
Download the tracking results from [Baidu Netdisk](https://pan.baidu.com/s/1n31MZ32ZNzSuYhaRsd-X5Q?pwd=pm6p) code: pm6p

Download the model weights from [Baidu Netdisk](https://pan.baidu.com/s/1koibm_DHj194yHpihfyf8Q?pwd=143t) code: 143t

## Acknowledgments
Thanks for the [PyTracking](https://github.com/visionml/pytracking) and [OSTrack](https://github.com/botaoye/OSTrack) library, which helps us to quickly implement our ideas.
