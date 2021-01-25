# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch
import os
import glob
import shutil
import numpy as np
from tqdm import tqdm
import warnings

from scipy.special import softmax

from fedtorch.components.datasets.loader.utils import load_shakespeare, load_emnist

class EMNIST(Dataset):

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self,root, split='train', client_id=None, only_digits=True, download=False):
        self.root= root
        self.split=split
        self.client_id = client_id
        self.only_digits=only_digits

        if self.only_digits:
            self.num_clients = 3383
        else:
            self.num_clients = 3400 

        if self.split in ['train','val'] and self.client_id is None:
            raise ValueError("For train and val splits the client_id should be specified as number between [0,{}]".format(self.num_clients))
        
        if download:
            self.download()

        if self.split in ['train','val']:
            path = os.path.join(self.root, self.split, 'EMNIST_client_{}.pt'.format(self.client_id))
        else:
            path = os.path.join(self.root, self.split, 'EMNIST_test.pt')

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.data, self.targets = torch.load(path)
        self.targets = self.targets.long()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
        
    def _check_exists(self):
        if not os.path.exists(os.path.join(self.root, self.split)):
            return False
        num_data_files = len(glob.glob(os.path.join(self.root,self.split,  "EMNIST*.pt")))
        if self.split in ['train', 'val']:
            chk_num_data_files = num_data_files == self.num_clients
        else:
            chk_num_data_files = num_data_files == 1
        return chk_num_data_files

    def download(self):
        """Download the EMNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return
        if self.split in ['train','val']:
            os.makedirs(os.path.join(self.root, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.root, 'val'), exist_ok=True)
        else:
            os.makedirs(os.path.join(self.root, 'test'), exist_ok=True)

        dataset_train, dataset_test = load_emnist(self.root, only_digits=self.only_digits)
        print('Start generating datasets...')
        if self.split in ['train','val']:
            rand_client_ids = np.random.permutation(len(dataset_train.client_ids))
            for i in tqdm(rand_client_ids):
                train_path = os.path.join(self.root,'train', 'EMNIST_client_{}.pt'.format(i))
                val_path = os.path.join(self.root,'val', 'EMNIST_client_{}.pt'.format(i))
                client_id = dataset_train.client_ids[i]
                client_train_dataset = dataset_train.create_dataset_for_client(client_id)
                client_val_dataset  = dataset_test.create_dataset_for_client(client_id)
                train_data = torch.tensor(client_train_dataset['pixels'].astype(np.float32))
                train_labels = torch.tensor(client_train_dataset['label'])
                val_data = torch.tensor(client_val_dataset['pixels'].astype(np.float32))
                val_labels = torch.tensor(client_val_dataset['label'])
                Client_EMNIST_Dataset_train = (train_data,train_labels)
                Client_EMNIST_Dataset_val  = (val_data,val_labels)
                with open(train_path, 'wb') as f:
                    torch.save(Client_EMNIST_Dataset_train,f)
                with open(val_path, 'wb') as f:
                    torch.save(Client_EMNIST_Dataset_val,f)

                del train_data, train_labels, val_data, val_labels
        else:
            test_data = []
            test_labels = []
            test_path = os.path.join(self.root, 'test', 'EMNIST_test.pt')
            for i in tqdm(range(len(dataset_test.client_ids))):
                client_id = dataset_test.client_ids[i]
                client_test_dataset  = dataset_test.create_dataset_for_client(client_id)
                test_data_client = torch.tensor(client_test_dataset['pixels'].astype(np.float32))
                test_labels_client = torch.tensor(client_test_dataset['label'])
                test_data.append(test_data_client)
                test_labels.append(test_labels_client)
            test_data = torch.cat(test_data,dim=0)
            test_labels = torch.cat(test_labels)
            Client_EMNIST_Dataset_test  = (test_data,test_labels)
            with open(test_path, 'wb') as f:
                torch.save(Client_EMNIST_Dataset_test,test_path)

        print('Done!')
        # Removing cache files
        dataset_train.close_file()
        dataset_test.close_file()
        shutil.rmtree(os.path.join(self.root,'.cache'))
        return




class Synthetic(Dataset):

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self,root, split='train', client_id=None, num_tasks=50, alpha=0.0, beta=0.0,
                 generate=True, regression=False, num_dim=60, dim_max=60):
        self.root= root
        self.split=split
        self.client_id = client_id
        self.num_tasks=num_tasks
        self.alpha = alpha
        self.beta = beta
        self.regression=regression
        self.num_dim = num_dim
        
        if generate:
            if self.num_dim == 0:
                self.num_dim = torch.randint(10,60,[1]).item()
            self.generate_synthetic(num_dim=self.num_dim)

        if self.regression:
            base_path = os.path.join(self.root, 'synthetic_reg{}-{}'.format(self.alpha,self.beta))
        else:
            base_path = os.path.join(self.root, 'synthetic{}-{}'.format(self.alpha,self.beta))
        if not self._check_exists(base_path):
            raise RuntimeError('Dataset not found in {}.'.format(base_path) +
                               ' You can use generate=True to generate new dataset')
        if self.split == 'train':
            path = os.path.join(base_path, 'Client_{}.pt'.format(client_id))
        else:
            path = os.path.join(base_path, 'Test.pt')
        self.data, self.targets = torch.load(path)

        self.num_dim = self.data.shape[1]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


    def generate_synthetic(self, 
                           num_classes=2,
                           seed=931231,
                           num_dim=60,
                           min_num_samples=500,
                           max_num_samples=1000,
                           test_ratio=0.2):
        if self.regression:
            base_path = os.path.join(self.root, 'synthetic_reg{}-{}'.format(self.alpha,self.beta))
            num_classes=1
        else:
            base_path = os.path.join(self.root, 'synthetic{}-{}'.format(self.alpha,self.beta))
        if self._check_exists(base_path):
            return
        os.makedirs(base_path, exist_ok=True)

        Sigma = np.zeros((num_dim, num_dim))
        for i in range(num_dim):
            Sigma[i, i] = (i + 1)**(-1.2)
        
        num_samples = self.get_num_samples(min_num=min_num_samples, max_num=max_num_samples)
        tasks = self.get_all_tasks(num_samples, num_dim, num_classes, Sigma)
        test_data = {'x':[],'y':[]}
        for i,t in enumerate(tasks):
            train_data, test_portion = self.split_task(t, test_ratio)
            test_data['x'].append(test_portion['x'])
            test_data['y'].append(test_portion['y'])
            train_path = os.path.join(base_path, 'Client_{}.pt'.format(i))
            with open(train_path, 'wb') as f:
                    torch.save(train_data,f)
        
        test_data['x'] = torch.tensor(np.concatenate(test_data['x']))
        test_data['y'] = torch.tensor(np.concatenate(test_data['y']))
        test_dataset = (test_data['x'], test_data['y'])
        test_path = os.path.join(base_path, 'Test.pt')
        with open(test_path, 'wb') as f:
            torch.save(test_dataset,f)
        
    def get_num_samples(self, min_num=500, max_num=1000):
        num_samples = np.random.lognormal(3, 2, (self.num_tasks)).astype(int)
        num_samples = [min(s + min_num, max_num) for s in num_samples]
        return num_samples
    
    def get_all_tasks(self,num_samples, num_dim, num_classes, Sigma):
        tasks = [self.get_task(s, num_dim, num_classes, Sigma) for s in num_samples]
        return tasks

    def get_task(self, num_samples, num_dim, num_classes, Sigma):
        new_task = self._generate_task(num_samples, num_dim, num_classes, Sigma)
        return new_task

    def _generate_x(self, num_samples,num_dim,Sigma):
        B = np.random.normal(loc=0.0, scale=self.beta, size=None)
        loc = np.random.normal(loc=B, scale=1.0, size=num_dim)

        samples = np.ones((num_samples, num_dim + 1))
        samples[:, 1:] = np.random.multivariate_normal(
            mean=loc, cov=Sigma, size=num_samples)

        return samples

    def _generate_y(self, x, num_classes):
        num_samples, num_dim = x.shape
        loc = np.random.normal(loc=0, scale=self.alpha, size=None)
        w = np.random.normal(loc=loc, scale=1, size=(num_dim, num_classes))
        out = np.matmul(x, w) + np.random.normal(loc=loc, scale=0.1, size=(num_samples, num_classes))
        if not self.regression:
            prob = softmax(out, axis=1)
            y = np.argmax(prob, axis=1)
        else:
            y = np.squeeze(out)
        return y, w

    def _generate_task(self, num_samples, num_dim, num_classes, Sigma):
        x = self._generate_x(num_samples, num_dim, Sigma)
        y, w = self._generate_y(x, num_classes)

        # now that we have y, we can remove the bias coeff
        x = x[:, 1:]

        return {'x': x, 'y': y, 'w': w}

    def split_task(self, task, test_ratio):
        num_samples = task['x'].shape[0]
        shuffle_inds = np.random.permutation(num_samples)
        train_inds = shuffle_inds[:int(num_samples * (1-test_ratio))]
        test_inds = shuffle_inds[int(num_samples * (1-test_ratio)):]
        test_portion = {}
        y_type = 'float32' if self.regression else 'long'
        train_data = (torch.tensor(task['x'][train_inds,:].astype('float32')), torch.tensor(task['y'][train_inds].astype(y_type)))
        test_portion['x'] = task['x'][test_inds,:].astype('float32')
        test_portion['y'] = task['y'][test_inds].astype(y_type)
        return train_data, test_portion
        
    def _check_exists(self,base_path):
        if not os.path.exists(base_path):
            return False
        num_data_files = len(glob.glob(os.path.join(base_path,  "*.pt")))
        if num_data_files < self.num_tasks + 1:
            return False
        return True




class Shakespeare(Dataset):

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self,root, split='train', client_id=None, download=False, batch_size=2, seq_len=50):
        self.root= root
        self.split=split
        self.client_id = client_id
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.num_clients = 446 # Number of clients who have more than 2*51 chars in test data
        self.vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r\{\}')

        # mappings
        self.char2idx = {u:i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

        if self.split in ['train','val'] and self.client_id is None:
            raise ValueError("For train and val splits the client_id should be specified as number between [0,{}]".format(self.num_clients))
        
        if download:
            self.download()

        if self.split in ['train','val']:
            path = os.path.join(self.root, self.split, 'Shakespeare_client_{}.pt'.format(self.client_id))
        else:
            path = os.path.join(self.root, self.split, 'Shakespeare_test.pt')

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.split in ['train','val']:
            _raw_data, self.client_name, self.num_clients = torch.load(path)
        else:
            _raw_data, self.num_clients = torch.load(path)
        len_data = _raw_data.shape[0]
        _rem_ind = len_data % (self.seq_len+1)
        _data_mat = _raw_data[:len_data - _rem_ind].reshape(-1,self.seq_len + 1)
        self.data = _data_mat[:,:-1]
        self.targets = _data_mat[:,1:]
        del _raw_data, _data_mat


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
        
    def _check_exists(self):
        if not os.path.exists(os.path.join(self.root, self.split)):
            return False
        num_data_files = len(glob.glob(os.path.join(self.root,self.split,  "Shakespeare*.pt")))
        if self.split in ['train', 'val']:
            path = os.path.join(self.root, self.split, 'Shakespeare_client_0.pt')
            _,_, num_clients = torch.load(path)
            chk_num_data_files = num_data_files == num_clients
        else:
            chk_num_data_files = num_data_files == 1
        return chk_num_data_files

    def download(self):
        """Download the EMNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return
        if self.split in ['train','val']:
            os.makedirs(os.path.join(self.root, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.root, 'val'), exist_ok=True)
        else:
            os.makedirs(os.path.join(self.root, 'test'), exist_ok=True)


        dataset_train, dataset_test = load_shakespeare(self.root)
        print('Start generating datasets...')
        # Get clients with having enough char_len for the current batch size and seq_length
        client_ids = []
        cut_off_size = self.batch_size*(self.seq_len+1)
        for i,ci in enumerate(dataset_test.client_ids):
            raw_example_dataset = dataset_test.create_dataset_for_client(ci)
            example_char_len = 0 
            for exp in raw_example_dataset['snippets']:
                example_char_len += len(exp)
            if example_char_len >= cut_off_size:
                raw_example_dataset_train = dataset_train.create_dataset_for_client(ci)
                example_char_len_train = 0 
                for exp in raw_example_dataset_train['snippets']:
                    example_char_len_train += len(exp)
                if example_char_len_train >= cut_off_size:
                    client_ids.append(ci)
        self.num_clients = len(client_ids)

        # Save datasets
        if self.split in ['train','val']:
            rand_client_ids = np.random.permutation(len(client_ids))
            for i in tqdm(rand_client_ids):
                train_path = os.path.join(self.root,'train', 'Shakespeare_client_{}.pt'.format(i))
                val_path = os.path.join(self.root,'val', 'Shakespeare_client_{}.pt'.format(i))
                client_id = client_ids[i]
                client_train_dataset = dataset_train.create_dataset_for_client(client_id)
                client_val_dataset  = dataset_test.create_dataset_for_client(client_id)
                train_data = []
                val_data = []
                for train_sample in client_train_dataset['snippets']:
                    train_data.append(
                        self.to_inds(train_sample.decode('UTF-8'))
                        )
                train_data = torch.cat(train_data).long()
                for val_sample in client_val_dataset['snippets']:
                    val_data.append(
                        self.to_inds(val_sample.decode('UTF-8'))
                        )
                val_data = torch.cat(val_data).long()

                Client_Shakespeare_Dataset_train = (train_data, client_id, self.num_clients)
                Client_Shakespeare_Dataset_val   = (val_data, client_id, self.num_clients)
                with open(train_path, 'wb') as f:
                    torch.save(Client_Shakespeare_Dataset_train,f)
                with open(val_path, 'wb') as f:
                    torch.save(Client_Shakespeare_Dataset_val,f)

                del train_data, val_data
        else:
            test_data = []
            test_path = os.path.join(self.root, 'test', 'Shakespeare_test.pt')
            for i in tqdm(range(len(client_ids))):
                client_id = client_ids[i]
                test_data_client = []
                client_test_dataset  = dataset_test.create_dataset_for_client(client_id)
                for test_sample in client_test_dataset['snippets']:
                    test_data_client.append(
                        self.to_inds(test_sample.decode('UTF-8'))
                    )
                test_data.append(torch.cat(test_data_client).long())
            test_data = torch.cat(test_data).long()
            Client_Shakespeare_Dataset_test  = (test_data,self.num_clients)
            with open(test_path, 'wb') as f:
                torch.save(Client_Shakespeare_Dataset_test,test_path)

        print('Done!')
        dataset_train.close_file()
        dataset_test.close_file()
        shutil.rmtree(os.path.join(self.root,'.cache'))
        return
    
    def to_inds(self,string):
        return torch.tensor([self.char2idx.get(x) for x in string])
    def to_chars(self,inds):
        return self.idx2char[inds]
    def to_string(self,inds):
        return ''.join(self.idx2char[inds].tolist())