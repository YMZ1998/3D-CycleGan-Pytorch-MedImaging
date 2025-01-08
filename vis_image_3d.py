import numpy as np
from matplotlib import pyplot as plt


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('Use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices // 2  # Start from the middle slice

        # Display the initial slice
        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices  # Move to the next slice
        else:
            self.ind = (self.ind - 1) % self.slices  # Move to the previous slice
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])  # Update the image data
        self.ax.set_ylabel(f'Slice {self.ind}')  # Update the slice label
        self.im.axes.figure.canvas.draw()  # Redraw the figure


def plot3d(image):
    # Rotate the image if needed
    image = np.rot90(image, k=-1)

    # Create a figure and axis
    fig, ax = plt.subplots(1, 1)

    # Adjust the figure size based on the image dimensions
    fig.set_size_inches(8, 8 * (image.shape[0] / image.shape[1]))

    # Create the IndexTracker object
    tracker = IndexTracker(ax, image)

    # Connect the scroll event to the tracker
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

    # Display the plot
    plt.tight_layout()  # Ensure the layout is tight to avoid clipping
    plt.show()


if __name__ == '__main__':
    import SimpleITK as sitk

    # Load the image
    image = sitk.ReadImage(r"./result/predict.nii.gz")
    image = sitk.GetArrayFromImage(image)  # Convert to NumPy array
    image = image.transpose(1, 2, 0)  # Transpose to (rows, cols, slices)
    image = np.rot90(image, k=1)  # Rotate if needed

    print("Image shape:", image.shape)  # Print the shape of the image

    # Plot the 3D image
    plot3d(image)