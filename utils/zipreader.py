import zipfile
import os

import cv2
import numpy as np

from mmcv.utils import is_str
from mmcv.opencv_info import USE_OPENCV2

if not USE_OPENCV2:
    from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
else:
    from cv2 import CV_LOAD_IMAGE_COLOR as IMREAD_COLOR
    from cv2 import CV_LOAD_IMAGE_GRAYSCALE as IMREAD_GRAYSCALE
    from cv2 import CV_LOAD_IMAGE_UNCHANGED as IMREAD_UNCHANGED

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED
}


class ZipReader(object):
    zip_bank = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        zip_bank = ZipReader.zip_bank
        if path in zip_bank:
            return zip_bank[path]
        else:
            # print("creating new zip_bank")
            zfile = zipfile.ZipFile(path, 'r')
            zip_bank[path] = zfile
            return zip_bank[path]

    @staticmethod
    def split_zip_style_path(path):
        pos_at = path.index('@')
        if pos_at == -1:
            print("character '@' is not found from the given path '%s'" %
                  (path))
            assert 0
        zip_path = path[0:pos_at]
        folder_path = path[pos_at + 1:]
        folder_path = str.strip(folder_path, '/')
        return zip_path, folder_path

    @staticmethod
    def list_folder(path):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        folder_list = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and \
               len(os.path.splitext(file_foler_name)[-1]) == 0 and \
               file_foler_name != folder_path:
                if len(folder_path) == 0:
                    folder_list.append(file_foler_name)
                else:
                    folder_list.append(file_foler_name[len(folder_path) + 1:])

        return folder_list

    @staticmethod
    def list_files(path, extension=['.*']):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        file_lists = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and str.lower(
                    os.path.splitext(file_foler_name)[-1]) in extension:
                if len(folder_path) == 0:
                    file_lists.append(file_foler_name)
                else:
                    file_lists.append(file_foler_name[len(folder_path) + 1:])

        return file_lists

    @staticmethod
    def imread(path, flag='color'):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        flag = imread_flags[flag] if is_str(flag) else flag
        im = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
        return im

    @staticmethod
    def read(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        return data
