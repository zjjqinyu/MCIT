import numpy as np
import multiprocessing
import os
import sys
import csv
from itertools import product
from collections import OrderedDict
from pytracking.evaluation import Sequence, Tracker
from ltr.data.image_loader import imwrite_indexed


PREDICTION_FIELD_NAMES = ['video', 'object', 'frame_num', 'present', 'score', 'xmin', 'xmax', 'ymin', 'ymax']


def _save_tracker_output_oxuva(seq: Sequence, tracker: Tracker, output: dict):
    if not os.path.exists(tracker.results_dir):
        os.makedirs(tracker.results_dir)

    frame_names = [os.path.splitext(os.path.basename(f))[0] for f in seq.frames]

    img_h, img_w = output['image_shape']
    tracked_bb = np.array(output['target_bbox'])
    object_presence_scores = np.array(output['object_presence_score'])

    tracked_bb = np.vstack([
        tracked_bb[:, 0]/img_w,
        (tracked_bb[:, 0] + tracked_bb[:, 2])/img_w,
        tracked_bb[:, 1]/img_h,
        (tracked_bb[:, 1] + tracked_bb[:, 3])/img_h,
    ]).T
    tracked_bb = tracked_bb.clip(0., 1.)

    tracked_bb = tracked_bb[1:]
    object_presence_scores = object_presence_scores[1:]
    frame_numbers = np.array(list(map(int, frame_names[1:])))
    vid_id, obj_id = seq.name.split('_')[:2]

    pred_file = os.path.join(tracker.results_dir, '{}_{}.csv'.format(vid_id, obj_id))

    with open(pred_file, 'w') as fp:
        writer = csv.DictWriter(fp, fieldnames=PREDICTION_FIELD_NAMES)

        for i in range(0, len(frame_numbers)):
            row = {
                'video': vid_id,
                'object': obj_id,
                'frame_num': frame_numbers[i],
                'present': str(object_presence_scores[i] > output['object_presence_score_threshold']).lower(),  # True or False
                'score': object_presence_scores[i],
                'xmin': tracked_bb[i, 0],
                'xmax': tracked_bb[i, 1],
                'ymin': tracked_bb[i, 2],
                'ymax': tracked_bb[i, 3],
            }
            writer.writerow(row)


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        os.makedirs(tracker.results_dir)

    base_results_path = os.path.join(tracker.results_dir, seq.name)
    segmentation_path = os.path.join(tracker.segmentation_dir, seq.name)

    frame_names = [os.path.splitext(os.path.basename(f['rgb']))[0] for f in seq.frames]

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)

        elif key == 'segmentation':
            assert len(frame_names) == len(data)
            if not os.path.exists(segmentation_path):
                os.makedirs(segmentation_path)
            for frame_name, frame_seg in zip(frame_names, data):
                imwrite_indexed(os.path.join(segmentation_path, '{}.png'.format(frame_name)), frame_seg)


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, visdom_info=None):
    """Runs a tracker on a sequence."""

    def _results_exist():
        if seq.dataset == 'oxuva':
            vid_id, obj_id = seq.name.split('_')[:2]
            pred_file = os.path.join(tracker.results_dir, '{}_{}.csv'.format(vid_id, obj_id))
            return os.path.isfile(pred_file)
        elif seq.object_ids is None:
            bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    visdom_info = {} if visdom_info is None else visdom_info

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)
    else:
        try:
            output = tracker.run_sequence(seq, debug=debug, visdom_info=visdom_info)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        if seq.dataset == 'oxuva':
            _save_tracker_output_oxuva(seq, tracker, output)
        else:
            _save_tracker_output(seq, tracker, output)


def run_dataset(dataset, trackers, debug=False, threads=0, visdom_info=None):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
        visdom_info: Dict containing information about the server for visdom
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))
    import time
    start_time = time.time()

    multiprocessing.set_start_method('spawn', force=True)

    visdom_info = {} if visdom_info is None else visdom_info

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug, visdom_info=visdom_info)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, visdom_info) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
