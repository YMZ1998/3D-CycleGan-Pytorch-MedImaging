import matplotlib.pyplot as plt
from utils.NiftiDataset import *
from torch.utils.data import DataLoader
import utils.NiftiDataset as NiftiDataset
from utils.random_crop import RandomCrop
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=r'./data/brain/train')
parser.add_argument("--resample", action='store_true', default=False, help='Decide or not to resample the images to a new resolution')
parser.add_argument("--new_resolution", type=float, default=(0.5, 0.5, 0.5), help='New resolution')
parser.add_argument("--patch_size", type=int, nargs=3, default=[192, 192, 32], help="Input dimension for the generator")
parser.add_argument("--batch_size", type=int, nargs=1, default=1, help="Batch size to feed the network (currently supports 1)")
parser.add_argument("--drop_ratio", type=float, nargs=1, default=0.1, help="Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1")
parser.add_argument("--min_pixel", type=int, nargs=1, default=0.1, help="Percentage of minimum non-zero pixels in the cropped label")

args = parser.parse_args()

min_pixel = int(args.min_pixel*((args.patch_size[0]*args.patch_size[1]*args.patch_size[2])/100))
print('min_pixel:', min_pixel)

trainTransforms = [
    # NiftiDataset.Resample(args.new_resolution, args.resample),
    # NiftiDataset.Registration(),
    # NiftiDataset.Align(),
    # NiftiDataset.Augmentation(),
    # NiftiDataset.Padding((300, 300, 300)),
    NiftiDataset.Padding((args.patch_size[0], args.patch_size[1], args.patch_size[2])),
    RandomCrop((args.patch_size[0], args.patch_size[1], args.patch_size[2]),
                            args.drop_ratio, min_pixel)
]

train_gen = NifitDataSet(args.data_path, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, train=True)
print('lenght train list:',len(train_gen))
train_loader = DataLoader(train_gen, batch_size=args.batch_size, shuffle=True)


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind],cmap= 'gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def plot3d(image):
    original=image
    original = np.rot90(original, k=-1)
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, original)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

if __name__ == '__main__':
    for i, batch in enumerate(train_loader):
        print(batch[0].shape)
        print(batch[1].shape)
        vol = batch[0].numpy()
        mask = batch[1].numpy()
        # print(vol.shape)
        vol = np.squeeze(vol)
        mask = np.squeeze(mask)
        print(vol.shape)
        plot3d(vol)
        # plot3d(mask)
# batch1 = train_loader.dataset[random.randint(0, len(train_gen) - 1)]
#
# vol = batch1[0].numpy()
# mask = batch1[1].numpy()
# print(vol.shape)
#
# vol = np.squeeze(vol, axis=0)
# mask = np.squeeze(mask, axis=0)
#
# plot3d(vol)
# plot3d(mask)
