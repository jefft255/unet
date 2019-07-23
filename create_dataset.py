from pathlib import Path
from libtiff import TIFF
import numpy as np

from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    raw_data_path = Path("data_borneo")

    X = []
    y = []
    for file_data in raw_data_path.glob("*.tif"):
        img = TIFF.open(file_data).read_image()
        print(f"{file_data}: {img.shape}")
        M = 256
        N = 256
        tiles = [img[x:x+M,y:y+N] for x in list(range(0,img.shape[0],M))[:-1]
                                  for y in list(range(0,img.shape[1],N))[:-1]]
        X = X + [tile[:,:,:-1] for tile in tiles]
        y = y + [tile[:,:,-1, np.newaxis] for tile in tiles]

print(X[-1].shape)
X = np.stack(X, axis=0)
y = np.stack(y, axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
np.save("X_train", X_train)
np.save("y_train", y_train)
np.save("X_test", X_test)
np.save("y_test", y_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
