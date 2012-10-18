# -*- coding: utf-8 -*-
"""The Oxford-IIIT Pet Dataset

http://www.robots.ox.ac.uk/~vgg/data/pets/

OXFORD-IIIT PET Dataset 
-----------------------
Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar

We have created a 37 category pet dataset with roughly 200 images for each class.
The images have a large variations in scale, pose and lighting. All images have an
associated ground truth annotation of breed, head ROI, and pixel
level trimap segmentation.

Contents:
--------
trimaps/    Trimap annotations for every image in the dataset
        Pixel Annotations: 1: Foreground 2:Background 3: Not classified
xmls/       Head bounding box annotations in PASCAL VOC Format

list.txt    Combined list of all images in the dataset
        Each entry in the file is of following nature:
        Image CLASS-ID SPECIES BREED ID
        ID: 1:37 Class ids
        SPECIES: 1:Cat 2:Dog
        BREED ID: 1-25:Cat 1:12:Dog
        All images with 1st letter as captial are cat images while
        images with small first letter are dog images.
trainval.txt    Files describing splits used in the paper.However,
test.txt    you are encouraged to try random splits.



Support:
-------
For any queries contact,

Omkar Parkhi: omkar@robots.ox.ac.uk

References:
----------
[1] O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
   Cats and Dogs
   IEEE Conference on Computer Vision and Pattern Recognition, 2012

Note:
----
Dataset is made available for research purposes only. Use of these images must respect
the corresponding terms of use of original websites from which they are taken.
See [1] for list of websites.

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

from data_home import get_data_home
from utils import download, extract, xml2dict


class OxfordPet(object):
    """Oxford-IIIT Pet Dataset

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

            shape: dict with int values
                Shape of the image. dict with keys 'height', 'width', 'depth'
                and int values.

            split: str
                'train', 'val' or 'test'.

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

                    pose: str
                        'Left', 'Right', 'Frontal', 'Rear' or 'Unspecified'

                    truncated: boolean
                        True if the object is occluded / truncated.

                    difficult: boolean
                        True if the object has been tagged as difficult (should
                        be ignored during evaluation?).

            segmented: boolean
                True if segmentation information is available.

            owner: dict [optional]
                Owner of the image (self-explanatory).

            source: dict
                Source of the image (self-explanatory).


    Notes
    -----
    If joblib is available, then `meta` will be cached for faster processing.
    To install joblib use 'pip install -U joblib' or 'easy_install -U joblib'.
    """

    ARCHIVES = {
        'images': {
            'url': ('http://www.robots.ox.ac.uk/~vgg/data/pets/data/'
                    'images.tar.gz'),
            'sha1': 'b07d8d8bce6f2e4d5a1040b8dc23b66d12c1dc3a',
        },
        'annotations': {
            'url': ('http://www.robots.ox.ac.uk/~vgg/data/pets/data/'
                    'annotations.tar.gz'),
            'sha1': '96ff158bcae7ced76478f454d833d9fbc786bb36',
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
        return path.join(get_data_home(), 'oxford_pet', self.name, 
                         *suffix_paths)

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
        for name, archive in self.ARCHIVES.iteritems():
            archive_dir = path.join(home, name)
            if os.path.exists(archive_dir):
                continue
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

        img_pattern = path.join(self.home(), "images", "*.jpg")
        img_filenames = sorted(glob(img_pattern))
        n_imgs = len(img_filenames)

        # --
        print "Parsing annotations..."
        meta = []
        unique_object_names = []
        n_objects = 0
        img_ids = []
        for ii, img_filename in enumerate(img_filenames):

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

            # -- get xml filename
            xml_filename = path.join(self.home(), "annotations", "xmls", 
                                     img_id+".xml")
            if not path.exists(xml_filename):
                # annotation missing
                meta += [data]
                continue

            # -- parse xml
            xd = xml2dict(xml_filename)

            # image basename
            assert img_basename == xd['filename']

            # source
            data['source'] = xd['source']

            # owner (if available)
            if 'owner' in xd:
                data['owner'] = xd['owner']

            # size / shape
            size = xd['size']
            width = int(size['width'])
            height = int(size['height'])
            depth = int(size['depth'])
            data['shape'] = dict(height=height, width=width, depth=depth)

            # segmentation ?
            segmented = bool(xd['segmented'])
            data['segmented'] = segmented
            if segmented:
                # TODO: parse segmentation data (in 'SegmentationClass') or
                # lazy-evaluate it ?
                pass

            # objects with their bounding boxes
            objs = xd['object']
            if isinstance(objs, dict):  # case where there is only one bbox
                objs = [objs]
            objects = []
            for obj in objs:
                # parse bounding box coordinates and convert them to valid
                # 0-indexed coordinates
                bndbox = obj.pop('bndbox')
                x_min = max(0,
                            (int(np.round(float(bndbox['xmin']))) - 1))
                x_max = min(width - 1,
                            (int(np.round(float(bndbox['xmax']))) - 1))
                y_min = max(0,
                            (int(np.round(float(bndbox['ymin']))) - 1))
                y_max = min(height - 1,
                            (int(np.round(float(bndbox['ymax']))) - 1))
                bounding_box = dict(x_min=x_min, x_max=x_max,
                                    y_min=y_min, y_max=y_max)
                assert (np.array(bounding_box) >= 0).all()
                obj['bounding_box'] = bounding_box
                n_objects += 1
                if obj['name'] not in unique_object_names:
                    unique_object_names += [obj['name']]

                # convert 'difficult' to boolean
                if 'difficult' in obj:
                    obj['difficult'] = bool(int(obj['difficult']))
                else:
                    # assume difficult=False if key not present
                    obj['difficult'] = False

                # convert 'truncated' to boolean
                if 'truncated' in obj:
                    obj['truncated'] = bool(int(obj['truncated']))
                else:
                    # assume truncated=False if key not present
                    obj['truncated'] = False

                objects += [obj]

            data['objects'] = objects

            # -- print progress
            n_done = ii + 1
            status = ("Progress: %d/%d [%.1f%%]"
                      % (n_done, len(img_filenames), 100. * n_done / n_imgs))
            status += chr(8) * (len(status) + 1)
            print status,

            # -- append to meta
            meta += [data]

        print

        print " Number of images: %d" % len(meta)
        print " Number of unique object names: %d" % len(unique_object_names)
        print " Unique object names: %s" % unique_object_names

        # --
        print "Parsing splits..."
        main_dirname = path.join(self.home(), "annotations")

        # We use 'aeroplane_{train,trainval}.txt' to get the list of 'train'
        # and 'val' ids
        train_filename = path.join(main_dirname, 'list.txt')
        assert path.exists(train_filename)
        train_ids = np.loadtxt(train_filename, dtype=str)[:, 0]

        trainval_filename = path.join(main_dirname, 'trainval.txt')
        assert path.exists(trainval_filename)
        trainval_ids = np.loadtxt(trainval_filename, dtype=str)[:, 0]

        splits = 'train', 'val', 'test'
        split_counts = dict([(split, 0) for split in splits])
        for data in meta:
            img_id = data['id']
            if img_id in trainval_ids:
                if img_id in train_ids:
                    data['split'] = 'train'
                else:
                    data['split'] = 'val'
            else:
                data['split'] = 'test'
            split_counts[data['split']] += 1

        for split in splits:
            count = split_counts[split]
            # assert count > 0
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
