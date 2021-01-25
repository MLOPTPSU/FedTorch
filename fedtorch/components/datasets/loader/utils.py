# -*- coding: utf-8 -*-
import tarfile
import zipfile
import urllib.request
import shutil
import h5py
# import progressbar
from tqdm import tqdm
import collections
import os


_DATASET_MAP = {
    'shakespeare': 'https://storage.googleapis.com/tff-datasets-public/shakespeare.tar.bz2',
    'emnist'     : 'https://storage.googleapis.com/tff-datasets-public/fed_emnist'
}
def load_emnist(root_dir, only_digits=True):
    file_name_pre = 'fed_emnist'
    if only_digits:
        file_name_ext = '_digitsonly'
    else:
        file_name_ext = ''
    url = _DATASET_MAP['emnist'] + file_name_ext + '.tar.bz2'
    file_name = file_name_pre + file_name_ext + '.tar.bz2'
    file_path = os.path.join(root_dir, file_name)
    cache_dir = os.path.join(root_dir,'.cache')
    train_dir = os.path.join(cache_dir, file_name_pre + file_name_ext + '_train.h5')
    test_dir  = os.path.join(cache_dir, file_name_pre + file_name_ext + '_test.h5' )

    if not os.path.exists(cache_dir):
        download_and_extract(file_path, cache_dir, url)
    else:
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            shutil.rmtree(cache_dir)
            download_and_extract(file_path, cache_dir, url)
    train_client_data = HDF5ClientData(train_dir)
    test_client_data = HDF5ClientData(test_dir)
    return train_client_data, test_client_data

def load_shakespeare(root_dir):
    file_path = os.path.join(root_dir,'shakespeare.tar.bz2')
    cache_dir = os.path.join(root_dir,'.cache')
    train_dir = os.path.join(cache_dir,'shakespeare_train.h5')
    test_dir  = os.path.join(cache_dir,'shakespeare_test.h5')
    if not os.path.exists(cache_dir):
        download_and_extract(file_path, cache_dir, _DATASET_MAP['shakespeare'])
    else:
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            shutil.rmtree(cache_dir)
            download_and_extract(file_path, cache_dir, _DATASET_MAP['shakespeare'])
    train_client_data = HDF5ClientData(train_dir)
    test_client_data = HDF5ClientData(test_dir)
    return train_client_data, test_client_data


class HDF5ClientData():
    """A Class to contain federated data coming from tensorflow federated API
    """
    def __init__(self, hdf5_filepath):
        """
        Args:
            hdf5_filepath: String path to the hdf5 file.
        """
        self._filepath = hdf5_filepath

        self._h5_file = h5py.File(self._filepath, "r")
        self._client_ids = sorted(
            list(self._h5_file['examples'].keys()))

    def create_dataset_for_client(self, client_id):
        if client_id not in self.client_ids:
            raise ValueError(
                "ID [{i}] is not a client in this ClientData. See "
                "property `client_ids` for the list of valid ids.".format(
                    i=client_id))
        return collections.OrderedDict((name, ds[()]) for name, ds in sorted(
                self._h5_file['examples'][client_id].items()))

    @property
    def client_ids(self):
        return self._client_ids
    
    def close_file(self):
        self._h5_file.close()


def _extract_archive(file_path,extract_path='.', archive_format='auto'):
  """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.
  Arguments:
      file_path: path to the archive file
      extract_path: path to extract the archive file
      archive_format: Archive format to try for extracting the file.
          Options are 'auto', 'tar', 'zip', and None.
          'tar' includes tar, tar.gz, and tar.bz files.
          The default 'auto' is ['tar', 'zip'].
          None or an empty list will return no matches found.
  Returns:
      True if a match was found and an archive extraction was completed,
      False otherwise.
  """
  if archive_format is None:
    return False
  if archive_format == 'auto':
    archive_format = ['tar', 'zip']

  for archive_type in archive_format:
    if archive_type == 'tar':
      open_fn = tarfile.open
      is_match_fn = tarfile.is_tarfile
    if archive_type == 'zip':
      open_fn = zipfile.ZipFile
      is_match_fn = zipfile.is_zipfile

    if is_match_fn(file_path):
      with open_fn(file_path) as archive:
        try:
          archive.extractall(extract_path)
        except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
          if os.path.exists(extract_path):
            if os.path.isfile(extract_path):
              os.remove(extract_path)
            else:
              shutil.rmtree(extract_path)
          raise
      return True
  return False

def download_and_extract(file_path, extract_path, url):
    if not os.path.exists(file_path):
        # Download the files and extract it
        print("Downloading data from {}".format(url))
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=file_path) as t:
            urllib.request.urlretrieve(url,file_path, reporthook=t.update_to, data=None)
        _extract_archive(file_path,extract_path)
        os.remove(file_path)
    else:
        _extract_archive(file_path,extract_path)
    return


class TqdmUpTo(tqdm):
    # Provides `update_to(n)` which uses `tqdm.update(delta_n)`.

    last_block = 0
    def update_to(self, block_num=1, block_size=1, total_size=None):
        '''
        block_num  : int, optional
            Blocks transferred so far [default: 1].
        block_size : int, optional
                         The size of each block (in tqdm units) [default: 1].
        total_size : int, optional
                         Total file size (in tqdm units). If [default: None] remains unchanged.
        '''
        if total_size is not None:
            self.total = total_size
        self.update((block_num - self.last_block) * block_size)  
        self.last_block = block_num