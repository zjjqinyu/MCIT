class EnvironmentSettings:
    def __init__(self):
        self.project_dir = '/home/qinyu/project/MCIT'    # The root directory of the project.
        self.workspace_dir = '/mnt/hdd1/qinyu/checkpoints/vit_tracker'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.project_dir + '/pretrained_networks/'
        self.pregenerated_masks = ''
        self.gtot_dir = '/home/qinyu/dataset/tracking/rgbt/GTOT/data/'
        self.rgbt234_dir = '/home/qinyu/dataset/tracking/rgbt/RGBT234/data/'
        self.lasher_dir = '/mnt/hdd1/qinyu/dataset/tracking/rgbt/LasHeR/data/'
        self.vtuav_dir = '/home/qinyu/dataset/tracking/rgbt/VTUAV/data/'
