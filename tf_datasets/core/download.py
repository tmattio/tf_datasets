import os
import sys
from tqdm import tqdm


class _TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_http(url, output_file, username=None, password=None):
    from urllib import request
    filename = url.replace('/', ' ').split()[-1]
    with _TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                   desc=filename) as t:  # all optional kwargs
        request.urlretrieve(url, filename=output_file,
                            reporthook=t.update_to, data=None)


def download_google_drive(file_id, output_file, base_url, total_size=None):
    import requests

    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    session = requests.Session()
    response = session.get(base_url, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(base_url, params=params, stream=True)

    if not total_size:
        try:
            total_size = response.headers['Content-length']
        except KeyError:
            total_size = None

    block_size = 32768
    filename = os.path.basename(output_file)
    with _TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                   desc=filename, total=total_size) as t:
        with open(output_file, "wb") as f:
            count = 0
            for chunk in response.iter_content(block_size):
                if chunk:  # filter out keep-alive new chunks
                    count += 1
                    sys.stdout.flush()
                    f.write(chunk)
                    t.update_to(b=count, bsize=block_size, tsize=total_size)


def extract_gzip(filepath, output_dir):
    import gzip
    output_filename, ext = os.path.splitext(filepath)
    output_filename = os.path.basename(output_filename)
    output_filepath = os.path.join(output_dir, output_filename)

    with gzip.open(filepath, 'rb') as input, open(output_filepath, 'wb') as output:
        output.write(input.read())


def extract_zip(filepath, output_dir):
    from zipfile import ZipFile
    with ZipFile(filepath, 'r') as z:
        z.extractall(output_dir)


def extract_tgz(filepath, output_dir):
    import tarfile

    tar = tarfile.open(filepath, 'r:*')
    for f in tar:
        output_file = os.path.join(output_dir, f.name)
        try:
            tar.extract(f, output_dir)
        except IOError as e:
            os.remove(output_file)
            tar.extract(f, output_dir)
        finally:
            os.chmod(output_file, f.mode)
