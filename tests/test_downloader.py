#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `python_boilerplate` package."""

import os
import tempfile
import pytest  # noqa

import tf_datasets as tfd
import tf_datasets.core.base_downloader  # noqa


def test_http_download():
    with tempfile.TemporaryDirectory() as tmpdirname:
        dl = tfd.core.base_downloader.BaseDownloader(
            'https://www.quora.com/', tmpdirname + '/quora_homepage.html')
        dl.download()
        assert os.path.exists(tmpdirname + '/quora_homepage.html')

        dl = tfd.core.base_downloader.BaseDownloader(
            'https://docs.google.com/uc?export=download0B6eKvaijfFUDbW4tdGpaYjgzZkU',
            tmpdirname + '/wider_face_test_file')
        dl.download()
        assert os.path.exists(tmpdirname + '/wider_face_test_file')


if __name__ == '__main__':
    test_http_download()
