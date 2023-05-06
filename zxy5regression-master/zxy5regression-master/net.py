import torch.nn as nn
import torch

from tqdm import tqdm


def make_vgg19(batch_norm=False):
    layers = []
    in_channels = 1
    for v in [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn = make_vgg19(batch_norm=True)

        self.lin = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 64)
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        f = self.cnn(x)

        f = f.flatten(start_dim=1)
        g = self.lin(f)
        ##n = g.div(g.pow(2).sum(1, keepdim=True).pow(0.5))

        return g


def embed_trajectory(net, data_loader):
    embeddings = []
    for cart, _, _, _, _, _, _, _ in tqdm(data_loader):
        n = net(cart.cuda())
        embeddings.extend(n.tolist())
    return embeddings
