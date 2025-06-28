from ltr.train_settings.MCIT.MCIT_settings import get_tracker_settings
from ltr.models.tracking.MCIT import vit_tracker
from pytracking.utils import TrackerParams
import torch

def parameters():
    params = TrackerParams()
    params.debug = 0
    params.visualization = False
    params.use_gpu = True
    params.checkpoint = '/home/qinyu/project/MCIT/checkpoints/checkpoints/ltr/MCIT/MCIT/MCIT_ep0160.pth.tar'
    params.settings = get_tracker_settings()
    params.device = torch.device("cuda:0")
    params.net = vit_tracker(params.settings)

    return params
