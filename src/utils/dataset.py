import torch.utils.data as data
import PIL
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, df, transforms, tr_channels, dataset_path) -> None:
        self.experiments = df['experiment'].tolist()
        self.images = dataset_path + '/' + df['experiment'] + '/' + df['row'] + df['column'] + df['fov'].tolist()
        self.rows = df['row'].tolist()
        self.cols = df['column'].tolist()
        self.fovs = df['fov'].tolist()
        self.labels = df['label'].tolist()

        # convert channel names to list
        df['channel_names'] = df['channel_names'].apply(lambda x: eval(x))
        self.channels = df['channel_names'].tolist()
        self.train_channels = tr_channels

        # get transforms
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> dict:
        data = {}
        data['label'] = self.labels[index]
        data['row'] = self.rows[index]
        data['column'] = self.cols[index]
        data['fov'] = self.fovs[index]
        data['experiment'] = self.experiments[index]

        # find the indeces of the channels to be used
        chs = []
        for ch in self.train_channels:
            if ch in self.channels[index]:
                ch_idx = self.channels[index].index(ch) + 1
                ch_im = PIL.Image.open(self.images[index] + f'-ch{ch_idx}.png')
                # check if the image is 8-bit or 16-bit
                if ch_im.mode == 'L':
                    raise ValueError(f"Image {self.images[index]} is 8-bit, it should be 16-bit")
                chs.append(np.array(ch_im))
            else:
                chs.append(np.zeros((2160, 2160), dtype=np.uint16))
        
        image = np.stack(chs, axis=2)
        # cast to float
        image = image.astype(np.float32)

        image = self.transforms(image)

        data['image'] = image

        return data
