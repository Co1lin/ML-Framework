import os
import json
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

class MyDM(pl.LightningDataModule):

    def __init__(self, config = None):
        super().__init__()
        self.cfg = config
        self._load_data()

    def _get_dataset(self, mode: str):
        return MyDS(
            self.data,
            self.split_info[mode]
        )

    def _load_data(self):
        ds_path = os.path.join(self.cfg.data_dir)
        files = os.listdir(ds_path)
        
        # load data as a dict
        data = list(filter(lambda x: 'data.npz' in x, files))[0]
        data_path = os.path.join(ds_path, data)
        data = dict(np.load(data_path))
        
        # load spilt info
        split_file = list(filter(lambda x: 'split.npz' in x, files))
        split_info = {}
        if len(split_file) > 0:
            split_file = os.path.join(ds_path, split_file[0])
            split_info = dict(np.load(split_file, allow_pickle=True))
        else:
            num_entries = data['x'].shape[0]
            perm = np.random.permutation(num_entries)
            sp = [0.8, 0.1, 0.1]
            sp = [int(sp[0]*num_entries), int((sp[0] + sp[1])*num_entries)]
            split_info['train'] = perm[: sp[0]]
            split_info['val'] = perm[sp[0] : sp[1]]
            split_info['test'] = perm[sp[1] :]
            print('saving split info...')
            np.savez_compressed(
                os.path.join(f'{data_path[:-8]}split.npz'),
                train=split_info['train'],
                val=split_info['val'],
                test=split_info['test'],
            )
        
        '''for not random split'''
        '''
        tot_num = data['ret'].shape[0]
        self.split_info = {
            'train': np.arange(0, tot_num*0.8, dtype=np.integer),
            'val': np.arange(tot_num*0.8, tot_num*0.9, dtype=np.integer),
            'test': np.arange(tot_num*0.9, tot_num, dtype=np.integer),
        }
        '''
        # save x, y_label, info as torch tensors
        self.data = data
        self.split_info = split_info

    def train_dataloader(self):
        return self._get_dataloader('train')

    def val_dataloader(self):
        return self._get_dataloader('val')

    def test_dataloader(self):
        return self._get_dataloader('test')
    
    def _get_dataloader(self, mode: str):
        default_collate = torch.utils.data.dataloader.default_collate

        def collate_fn(batch):
            '''custom collate func for dict format batch of data'''
            collated_batch = {}
            for key in batch[0]:
                if key in ['1']:
                    collated_batch[key] = [elem[key] for elem in batch]
                else:
                    collated_batch[key] = default_collate([elem[key] for elem in batch])
        
            return collated_batch
        
        # Ref: https://pytorch.org/docs/master/notes/randomness.html#dataloader
        g = torch.Generator()
        g.manual_seed(self.cfg.seed)

        dataset = self._get_dataset(mode)

        dataloader = DataLoader(
            dataset=dataset,
            # sampler=sampler,
            num_workers=self.cfg.num_workers,
            batch_size=self.cfg.task.batch_size,
            shuffle=(mode == 'train'),
            collate_fn=default_collate,
            # worker_init_fn=seed_worker,
            generator=g,
        )
        
        return dataloader


class MyDS(Dataset):

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices
        
    def __len__(self):
        return self.data['x'].shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(self.data['x'][self.indices[index]])
        y_label = torch.from_numpy(self.data['y_label'][self.indices[index]])
        return {
            'x': x,
            'y_label': y_label,
        }
