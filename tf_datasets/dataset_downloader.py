# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod


class DatasetDownloader:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def cleanup_temporary_files(self):
        pass

    @abstractmethod
    def maybe_download(self):
        pass

    @abstractmethod
    def maybe_extract(self):
        pass
