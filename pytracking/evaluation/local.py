from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.gtot_path = '/home/qinyu/dataset/tracking/rgbt/GTOT/data/'
    settings.lasher_path = '/mnt/hdd1/qinyu/dataset/tracking/rgbt/LasHeR/data/'
    settings.rgbt210_path = '/home/qinyu/dataset/tracking/rgbt/RGBT210/data/'
    settings.rgbt234_path = '/home/qinyu/dataset/tracking/rgbt/RGBT234/data/'
    settings.vtuav_path = '/home/qinyu/dataset/tracking/rgbt/VTUAV/data/'

    settings.network_path = '/home/qinyu/project/MCIT/pytracking/networks/'    # Where tracking networks are stored.
    settings.result_plot_path = '/home/qinyu/project/MCIT/pytracking/result_plots/'
    settings.results_path = '/home/qinyu/project/MCIT/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/qinyu/project/MCIT/pytracking/segmentation_results/'

    return settings

