# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import Dict, List

from nuscenes import NuScenes


train_detect = ['scene-0-48', 'scene-0-101', 'scene-0-64', 'scene-0-154', 'scene-0-180', 'scene-0-162', 'scene-0-81', 'scene-0-187', 'scene-0-172', 'scene-0-89', 'scene-0-103', 'scene-0-102', 'scene-0-184', 'scene-0-148', 'scene-0-11', 'scene-0-3', 'scene-0-4', 'scene-0-129', 'scene-0-40', 'scene-0-166', 'scene-0-164', 'scene-0-22', 'scene-0-116', 'scene-0-6', 'scene-0-30', 'scene-0-163', 'scene-0-161', 'scene-0-123', 'scene-0-86', 'scene-0-95', 'scene-0-17', 'scene-0-195', 'scene-0-67', 'scene-0-24', 'scene-0-151', 'scene-0-38', 'scene-0-85', 'scene-0-167', 'scene-0-117', 'scene-0-58', 'scene-0-113', 'scene-0-200', 'scene-0-35', 'scene-0-108', 'scene-0-124', 'scene-0-147', 'scene-0-84', 'scene-0-94', 'scene-0-133', 'scene-0-120', 'scene-0-182', 'scene-0-181', 'scene-0-170', 'scene-0-153', 'scene-0-176', 'scene-0-37', 'scene-0-107', 'scene-0-53', 'scene-0-199', 'scene-0-146', 'scene-0-41', 'scene-0-96', 'scene-0-8', 'scene-0-75', 'scene-0-152', 'scene-0-72', 'scene-0-142', 'scene-0-169', 'scene-0-5', 'scene-0-2', 'scene-0-12', 'scene-0-76', 'scene-0-45', 'scene-0-36', 'scene-0-10', 'scene-0-57', 'scene-0-92', 'scene-0-91', 'scene-0-69', 'scene-0-18', 'scene-0-190', 'scene-0-173', 'scene-0-79', 'scene-0-175', 'scene-0-59', 'scene-0-118', 'scene-0-44', 'scene-0-158', 'scene-0-149', 'scene-0-33', 'scene-0-130', 'scene-0-171', 'scene-0-135', 'scene-0-14', 'scene-0-34', 'scene-0-126', 'scene-0-128', 'scene-0-105', 'scene-0-88', 'scene-0-179', 'scene-0-7', 'scene-0-114', 'scene-0-110', 'scene-0-83', 'scene-0-106', 'scene-0-77', 'scene-0-70', 'scene-0-159', 'scene-0-68', 'scene-0-51', 'scene-0-71', 'scene-0-65', 'scene-0-82', 'scene-0-63', 'scene-0-50', 'scene-0-23', 'scene-0-145', 'scene-0-97', 'scene-0-155', 'scene-0-1', 'scene-0-165', 'scene-0-99', 'scene-0-46', 'scene-0-29', 'scene-0-122', 'scene-0-60', 'scene-0-196', 'scene-0-198', 'scene-0-39', 'scene-0-13', 'scene-0-25', 'scene-0-27', 'scene-0-87', 'scene-0-134', 'scene-0-186', 'scene-0-177', 'scene-0-109', 'scene-0-80', 'scene-0-61', 'scene-0-138', 'scene-0-137', 'scene-0-62', 'scene-0-43', 'scene-0-121', 'scene-0-144', 'scene-0-143', 'scene-0-168', 'scene-0-188', 'scene-0-140', 'scene-0-16', 'scene-0-191', 'scene-0-20', 'scene-0-28', 'scene-0-54', 'scene-0-194', 'scene-0-52', 'scene-0-183', 'scene-0-139', 'scene-0-93', 'scene-0-73', 'scene-0-31', 'scene-0-157', 'scene-0-49', 'scene-0-15', 'scene-0-127', 'scene-0-150', 'scene-0-112', 'scene-0-21', 'scene-0-132', 'scene-0-104']

train_track = ['scene-0-48', 'scene-0-101', 'scene-0-64', 'scene-0-154', 'scene-0-180', 'scene-0-162', 'scene-0-81', 'scene-0-187', 'scene-0-172', 'scene-0-89', 'scene-0-103', 'scene-0-102', 'scene-0-184', 'scene-0-148', 'scene-0-11', 'scene-0-3', 'scene-0-4', 'scene-0-129', 'scene-0-40', 'scene-0-166', 'scene-0-164', 'scene-0-22', 'scene-0-116', 'scene-0-6', 'scene-0-30', 'scene-0-163', 'scene-0-161', 'scene-0-123', 'scene-0-86', 'scene-0-95', 'scene-0-17', 'scene-0-195', 'scene-0-67', 'scene-0-24', 'scene-0-151', 'scene-0-38', 'scene-0-85', 'scene-0-167', 'scene-0-117', 'scene-0-58', 'scene-0-113', 'scene-0-200', 'scene-0-35', 'scene-0-108', 'scene-0-124', 'scene-0-147', 'scene-0-84', 'scene-0-94', 'scene-0-133', 'scene-0-120', 'scene-0-182', 'scene-0-181', 'scene-0-170', 'scene-0-153', 'scene-0-176', 'scene-0-37', 'scene-0-107', 'scene-0-53', 'scene-0-199', 'scene-0-146', 'scene-0-41', 'scene-0-96', 'scene-0-8', 'scene-0-75', 'scene-0-152', 'scene-0-72', 'scene-0-142', 'scene-0-169', 'scene-0-5', 'scene-0-2', 'scene-0-12', 'scene-0-76', 'scene-0-45', 'scene-0-36', 'scene-0-10', 'scene-0-57', 'scene-0-92', 'scene-0-91', 'scene-0-69', 'scene-0-18', 'scene-0-190', 'scene-0-173', 'scene-0-79', 'scene-0-175', 'scene-0-59', 'scene-0-118', 'scene-0-44', 'scene-0-158', 'scene-0-149', 'scene-0-33', 'scene-0-130', 'scene-0-171', 'scene-0-135', 'scene-0-14', 'scene-0-34', 'scene-0-126', 'scene-0-128', 'scene-0-105', 'scene-0-88', 'scene-0-179', 'scene-0-7', 'scene-0-114', 'scene-0-110', 'scene-0-83', 'scene-0-106', 'scene-0-77', 'scene-0-70', 'scene-0-159', 'scene-0-68', 'scene-0-51', 'scene-0-71', 'scene-0-65', 'scene-0-82', 'scene-0-63', 'scene-0-50', 'scene-0-23', 'scene-0-145', 'scene-0-97', 'scene-0-155', 'scene-0-1', 'scene-0-165', 'scene-0-99', 'scene-0-46', 'scene-0-29', 'scene-0-122', 'scene-0-60', 'scene-0-196', 'scene-0-198', 'scene-0-39', 'scene-0-13', 'scene-0-25', 'scene-0-27', 'scene-0-87', 'scene-0-134', 'scene-0-186', 'scene-0-177', 'scene-0-109', 'scene-0-80', 'scene-0-61', 'scene-0-138', 'scene-0-137', 'scene-0-62', 'scene-0-43', 'scene-0-121', 'scene-0-144', 'scene-0-143', 'scene-0-168', 'scene-0-188', 'scene-0-140', 'scene-0-16', 'scene-0-191', 'scene-0-20', 'scene-0-28', 'scene-0-54', 'scene-0-194', 'scene-0-52', 'scene-0-183', 'scene-0-139', 'scene-0-93', 'scene-0-73', 'scene-0-31', 'scene-0-157', 'scene-0-49', 'scene-0-15', 'scene-0-127', 'scene-0-150', 'scene-0-112', 'scene-0-21', 'scene-0-132', 'scene-0-104']

train = list(sorted(set(train_detect + train_track)))

val = ['scene-0-32', 'scene-0-111', 'scene-0-156', 'scene-0-193', 'scene-0-119', 'scene-0-131', 'scene-0-160', 'scene-0-125', 'scene-0-19', 'scene-0-55', 'scene-0-42', 'scene-0-197', 'scene-0-66', 'scene-0-74', 'scene-0-100', 'scene-0-115', 'scene-0-189', 'scene-0-192', 'scene-0-185', 'scene-0-56', 'scene-0-174', 'scene-0-90', 'scene-0-47', 'scene-0-136', 'scene-0-141', 'scene-0-26', 'scene-0-78', 'scene-0-178', 'scene-0-9', 'scene-0-98']

test = ['scene-0-32', 'scene-0-111', 'scene-0-156', 'scene-0-193', 'scene-0-119', 'scene-0-131', 'scene-0-160', 'scene-0-125', 'scene-0-19', 'scene-0-55', 'scene-0-42', 'scene-0-197', 'scene-0-66', 'scene-0-74', 'scene-0-100', 'scene-0-115', 'scene-0-189', 'scene-0-192', 'scene-0-185', 'scene-0-56', 'scene-0-174', 'scene-0-90', 'scene-0-47', 'scene-0-136', 'scene-0-141', 'scene-0-26', 'scene-0-78', 'scene-0-178', 'scene-0-9', 'scene-0-98']

mini_train = ['scene-0-48', 'scene-0-101', 'scene-0-64', 'scene-0-154']

mini_val = ['scene-0-32', 'scene-0-111']



def create_splits_logs(split: str, nusc: 'NuScenes') -> List[str]:
    """
    Returns the logs in each dataset split of nuScenes.
    Note: Previously this script included the teaser dataset splits. Since new scenes from those logs were added and
          others removed in the full dataset, that code is incompatible and was removed.
    :param split: NuScenes split.
    :param nusc: NuScenes instance.
    :return: A list of logs in that split.
    """
    # Load splits on a scene-level.
    scene_splits = create_splits_scenes(verbose=False)

    assert split in scene_splits.keys(), 'Requested split {} which is not a known nuScenes split.'.format(split)

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split == 'test':
        assert version.endswith('test'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    else:
        raise ValueError('Requested split {} which this function cannot map to logs.'.format(split))

    # Get logs for this split.
    scene_to_log = {scene['name']: nusc.get('log', scene['log_token'])['logfile'] for scene in nusc.scene}
    logs = set()
    scenes = scene_splits[split]
    for scene in scenes:
        logs.add(scene_to_log[scene])

    return list(logs)


def create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping from split to scene names, rather than log names.
    The splits are as follows:
    - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
    - train_detect/train_track: Two halves of the train split used for separating the training sets of detector and
        tracker if required.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    all_scenes = train + val + test
    #assert len(all_scenes) == 1000 and len(set(all_scenes)) == 1000, 'Error: Splits incomplete!'
    scene_splits = {'train': train, 'val': val, 'test': test,
                    'mini_train': mini_train, 'mini_val': mini_val,
                    'train_detect': train_detect, 'train_track': train_track}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits


if __name__ == '__main__':
    # Print the scene-level stats.
    create_splits_scenes(verbose=True)
