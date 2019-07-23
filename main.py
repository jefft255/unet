from model import *
from data import *
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


X_train = np.load("X_train.npy")
X_train = np.nan_to_num(X_train, X_train.mean())
y_train = np.load("y_train.npy")
y_train = np.nan_to_num(y_train, 7)
X_test = np.load("X_test.npy")
X_test = np.nan_to_num(X_test, X_train.mean())
y_test = np.load("y_test.npy")
y_test = np.nan_to_num(y_test, 7)

"""
for i in range(y_test.shape[0]):
    print(y_test.shape)
    plt.subplot(1,2,1)
    plt.imshow(y_test[i,:,:,0])
    img = X_test[i,:,:,5:]
    img[:,:,0] -= img[:,:,0].min()
    img[:,:,1] -= img[:,:,1].min()
    img[:,:,2] -= img[:,:,2].min()
    img[:,:,0] /= img[:,:,0].max()
    img[:,:,1] /= img[:,:,1].max()
    img[:,:,2] /= img[:,:,2].max()
    print(X_test.min())
    print(X_test.max())
    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.show()
"""

batch_size = 8

y_train = to_categorical(y_train, num_classes=8)
y_test = to_categorical(y_test, num_classes=8)

print(np.sum(y_train, axis=(0,1,2)))

myGene = trainGenerator(batch_size,  # Batch size
                        X_train,
                        y_train)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.001, cooldown=9000000, min_lr=0)
model.fit_generator(myGene,steps_per_epoch=X_train.shape[0] // batch_size,epochs=100,callbacks=[model_checkpoint, lr], validation_data=(X_test, y_test))

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test",results)
