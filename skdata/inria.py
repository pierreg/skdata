# -*- coding: utf-8 -*-
"""INRIA Person Dataset

http://pascal.inrialpes.fr/data/human/
"""

# Copyright (C) 2012
# Authors: Pierre Garrigues <pierre.garrigues@gmail.com>
#
# License: Simplified BSD

import os
from os import path
import shutil
from distutils import dir_util
from glob import glob
import hashlib

import numpy as np
import cv2

from data_home import get_data_home
from utils import download, extract, xml2dict


class INRIAPerson(object):
    """INRIA Person Dataset

    Attributes
    ----------
    meta: list of dict
        Metadata associated with the dataset. For each image with index i,
        meta[i] is a dict with keys:

            id: str
                Identifier of the image.

            filename: str
                Full path to the image.

            sha1: str
                SHA-1 hash of the image.

            split: str
                'train', 'test'.

            objects: list of dict [optional]
                Description of the objects present in the image. Note that this
                key may not be available if split is 'test'. If the key is
                present, then objects[i] is a dict with keys:

                    name: str
                        Name (label) of the object.

                    bounding_box: dict with int values
                        Bounding box coordinates (0-based index). dict with
                        keys 'x_min', 'x_max', 'y_min', 'y_max' and int values
                        such that:
                        +-----------------------------------------▶ x-axis
                        |
                        |   +-------+    .  .  .  y_min (top)
                        |   | bbox  |
                        |   +-------+    .  .  .  y_max (bottom)
                        |
                        |   .       .
                        |
                        |   .       .
                        |
                        |  x_min   x_max
                        |  (left)  (right)
                        |
                        ▼
                        y-axis


    Notes
    -----
    If joblib is available, then `meta` will be cached for faster processing.
    To install joblib use 'pip install -U joblib' or 'easy_install -U joblib'.
    """

    ARCHIVES = {
        'data': {
            'url': ('http://pascal.inrialpes.fr/data/human/'
                    'INRIAPerson.tar'),
            'sha1': 'c57941e3f359649342df9bf889f8a0bc6d159505',
        },
    }

    def __init__(self, meta=None):
        if meta is not None:
            self._meta = meta

        self.name = self.__class__.__name__

        try:
            from joblib import Memory
            mem = Memory(cachedir=self.home('cache'))
            self._get_meta = mem.cache(self._get_meta)
        except ImportError:
            pass

    def home(self, *suffix_paths):
        return path.join(get_data_home(), 'inria', self.name, *suffix_paths)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: fetch()
    # ------------------------------------------------------------------------

    def fetch(self):
        """Download and extract the dataset."""

        home = self.home()
        if not path.exists(home):
            os.makedirs(home)

        # download archives
        archive_filenames = []
        for key, archive in self.ARCHIVES.iteritems():
            url = archive['url']
            sha1 = archive['sha1']
            basename = path.basename(url)
            archive_filename = path.join(home, basename)
            if not path.exists(archive_filename):
                download(url, archive_filename, sha1=sha1)
            archive_filenames += [(archive_filename, sha1)]
            self.ARCHIVES[key]['archive_filename'] = archive_filename

        # extract them
        if not path.exists(path.join(home, 'INRIAPerson')):
            for archive in self.ARCHIVES.itervalues():
                url = archive['url']
                sha1 = archive['sha1']
                archive_filename = archive['archive_filename']
                extract(archive_filename, home, sha1=sha1, verbose=True)
                # move around stuff if needed
                if 'moves' in archive:
                    for move in archive['moves']:
                        src = self.home(move['source'])
                        dst = self.home(move['destination'])
                        # We can't use shutil here since the destination folder
                        # may already exist. Fortunately the distutils can help
                        # us here (see standard library).
                        dir_util.copy_tree(src, dst)
                        dir_util.remove_tree(src)

    # ------------------------------------------------------------------------
    # -- Dataset Interface: meta
    # ------------------------------------------------------------------------

    @property
    def meta(self):
        if hasattr(self, '_meta'):
            return self._meta
        else:
            self.fetch()
            self._meta = self._get_meta()
            return self._meta

    def _get_meta(self):

        base_dirname = path.join(self.home(), self.name)
        dirs = dict([(basename, path.join(base_dirname, basename))
                      for basename in os.listdir(base_dirname)
                      if path.isdir(path.join(base_dirname, basename))])


        img_pattern_trn = path.join(dirs['Train'], "*/*.png")
        img_pattern_tst = path.join(dirs['Test'], "*/*.png")
        img_filenames = sorted(glob(img_pattern_trn) + glob(img_pattern_tst))
        n_imgs = len(img_filenames)

        # --
        print "Parsing annotations..."
        meta = []
        n_objects = 0
        img_ids = []
        for i, img_filename in enumerate(img_filenames):

            data = {}

            data['filename'] = img_filename

            # sha1 hash
            sha1 = hashlib.sha1(open(img_filename).read()).hexdigest()
            data['sha1'] = sha1

            # image id
            img_basename = path.basename(path.split(img_filename)[1])
            img_id = path.splitext(img_basename)[0]
            img_ids += [img_id]

            data['id'] = img_id

            # -- get split
            split = img_filename.split("/")[-3]
            data['split'] = split.lower()

            # -- get bounding box and image size
            im = cv2.imread(img_filename)
            data['shape'] = {"depth": im.shape[2], "height": im.shape[0],
                             "width": im.shape[1]}

            # -- get annotation filename
            data['objects'] = []
            annot_filename = path.join(base_dirname, split, "annotations",
                                       "%s.txt" % img_id)
            if not path.exists(annot_filename):
                # annotation missing
                meta += [data]
                continue

            # -- parse txt file
            with open(annot_filename, "r") as fh:
                lines = fh.readlines()

            def box(s):
                x, y = s.split(", ")
                return int(y[:-1]), int(x[1:])

            for line in lines:

                # get bounding box
                if line[:8] != "Bounding":
                    continue
                b_min, b_max = line.split(":")[1].split(' - ')
                b_min = b_min.strip()
                b_max = b_max.strip()
                y_min, x_min = box(b_min)
                y_max, x_max = box(b_max)
                # t = (y2 - y1) / 4
                # bboxes.append((x1-t, y1-t, x2+t, y2+t))

                obj = dict(bounding_box = dict(x_min=x_min, x_max=x_max,
                                               y_min=y_min, y_max=y_max),
                           name="person")
                data['objects'] += [obj]
                n_objects += 1

            # -- print progress
            n_done = i + 1
            status = ("Progress: %d/%d [%.1f%%]"
                      % (n_done, len(img_filenames), 100. * n_done / n_imgs))
            status += chr(8) * (len(status) + 1)
            print status,

            # -- append to meta
            meta += [data]

        print

        print " Number of images: %d" % len(meta)
        print " Number of objects: %d" % n_objects


        splits = 'train', 'test'
        split_counts = dict([(split, 0) for split in splits])
        for data in meta:
            img_id = data['id']
            split_counts[data['split']] += 1

        for split in splits:
            count = split_counts[split]
            assert count > 0
            print(" Number of images in '%s': %d"
                  % (split, count))

        meta = np.array(meta)
        return meta

    # ------------------------------------------------------------------------
    # -- Dataset Interface: clean_up()
    # ------------------------------------------------------------------------

    def clean_up(self):
        if path.isdir(self.home()):
            shutil.rmtree(self.home())

    # ------------------------------------------------------------------------
    # -- Driver routines to be called by skdata.main
    # ------------------------------------------------------------------------

    @classmethod
    def main_fetch(cls):
        cls.fetch(download_if_missing=True)

    @classmethod
    def main_show(cls):
        raise NotImplementedError


def main_fetch():
    raise NotImplementedError


def main_show():
    raise NotImplementedError
