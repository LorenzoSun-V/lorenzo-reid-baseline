# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import sys
import os
import os.path as osp
import re
import glob
from .bases import ImageDataset
from playreid.data.datasets import DATASET_REGISTRY
##### Log #####
# 22.01.2019
# - add v2
# - v1 and v2 differ in dir names
# - note that faces in v2 are blurred
TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'MSMT17_V1': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    }
}


@DATASET_REGISTRY.register()
class MSMT17(ImageDataset):
    """MSMT17.
    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_

    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    # dataset_dir = 'MSMT17_V2'
    dataset_dir = '/workspace/lorenzo/ReID/datasets/cmdm/msmt17'
    dataset_url = None
    dataset_name = 'msmt17'

    def __init__(self, root='datasets', **kwargs):
        # self.dataset_dir = root
        #
        # has_main_dir = False
        # for main_dir in VERSION_DICT:
        #     if osp.exists(osp.join(self.dataset_dir, main_dir)):
        #         train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
        #         test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
        #         has_main_dir = True
        #         break
        # assert has_main_dir, 'Dataset folder not found'
        #
        # self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        # self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        # self.list_train_path = osp.join(self.dataset_dir, main_dir, 'list_train.txt')
        # self.list_val_path = osp.join(self.dataset_dir, main_dir, 'list_val.txt')
        # self.list_query_path = osp.join(self.dataset_dir, main_dir, 'list_query.txt')
        # self.list_gallery_path = osp.join(self.dataset_dir, main_dir, 'list_gallery.txt')
        #
        # required_files = [
        #     self.dataset_dir,
        #     self.train_dir,
        #     self.test_dir
        # ]
        # self.check_before_run(required_files)
        #
        # train = self.process_dir(self.train_dir, self.list_train_path)
        # val = self.process_dir(self.train_dir, self.list_val_path)
        # query = self.process_dir(self.test_dir, self.list_query_path, is_train=False)
        # gallery = self.process_dir(self.test_dir, self.list_gallery_path, is_train=False)
        #
        # num_train_pids = self.get_num_pids(train)
        # query_tmp = []
        # for img_path, pid, camid in query:
        #     query_tmp.append((img_path, pid+num_train_pids, camid))
        # del query
        # query = query_tmp
        #
        # gallery_temp = []
        # for img_path, pid, camid in gallery:
        #     gallery_temp.append((img_path, pid+num_train_pids, camid))
        # del gallery
        # gallery = gallery_temp
        #
        # # Note: to fairly compare with published methods on the conventional ReID setting,
        # #       do not add val images to the training set.
        # if 'combineall' in kwargs and kwargs['combineall']:
        #     train += val

        # cmdm dataset:
        assert os.path.exists(self.dataset_dir), "Please check dataset MSMT17 path!"
        self.root = self.dataset_dir
        self.train_dir = osp.join(self.root, 'train')
        self.query_dir = osp.join(self.root, 'query')
        self.gallery_dir = osp.join(self.root, 'gallery')

        required_files = [
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        # cmdm dataset:
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        data = []
        pid_folders = os.listdir(dir_path)
        for pid_folder in pid_folders:
            img_paths = glob.glob(osp.join(dir_path, pid_folder, '*.jpg'))
            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                assert 1 <= camid <= 15
                camid -= 1  # index starts from 0
                if is_train:
                    pid = self.dataset_name + "_" + str(pid)
                    camid = self.dataset_name + "_" + str(camid)
                data.append((img_path, pid, camid))

        return data

        # this is fastreid original dataset
        # with open(list_path, 'r') as txt:
        #     lines = txt.readlines()
        #
        # data = []
        #
        # for img_idx, img_info in enumerate(lines):
        #     img_path, pid = img_info.split(' ')
        #     pid = int(pid)  # no need to relabel
        #     camid = int(img_path.split('_')[2]) - 1  # index starts from 0
        #     img_path = osp.join(dir_path, img_path)
        #     if is_train:
        #         pid = self.dataset_name + "_" + str(pid)
        #         camid = self.dataset_name + "_" + str(camid)
        #     data.append((img_path, pid, camid))

        # return data
