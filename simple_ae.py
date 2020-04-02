from netCDF4 import Dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    R1 =  Dataset('data/T2m_R1_ym_1stMill.nc', 'r')
    temperature = R1.variables['T2m'][:]
    lat = R1.variables['lat'][:]
    lon = R1.variables['lon'][:]

    return temperature

def classifier():

    latent_dimension = 32

    data = get_data()
    data = data.reshape(999,96,192,1)

    train_data = data[:900]
    test_data = data[900:]

    print(train_data.shape)
    print(test_data.shape)

    encoder = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = (4,4),
            strides = (4,4),
            activation = 'relu'),
        tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = (4,4),
            strides = (4,4),
            activation = 'relu'),
        tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = (3,4),
            strides = (3,4),
            activation = 'relu'),
        tf.keras.layers.Conv2D(
            filters = 128,
            kernel_size = (2,3),
            strides = (2,3),
            activation = 'relu'),
        tf.keras.layers.Dense(latent_dimension)])

    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(128),
        tf.keras.layers.Conv2DTranspose(
            filters = 64,
            kernel_size = (2,3),
            strides = (2,3),
            activation = 'relu'),
        tf.keras.layers.Conv2DTranspose(
            filters = 64,
            kernel_size = (3,4),
            strides = (3,4),
            activation = 'relu'),
        tf.keras.layers.Conv2DTranspose(
            filters = 64,
            kernel_size = (4,4),
            strides = (4,4),
            activation = 'relu'),
        tf.keras.layers.Conv2DTranspose(
            filters = 1,
            kernel_size = (4,4),
            strides = (4,4),
            activation = 'relu')])


    model = tf.keras.Sequential([encoder, decoder])

    model.compile(loss = 'mse', optimizer = 'Adam', metrics = ['mse'])

    model.fit(train_data, train_data, batch_size = 8, epochs = 100) 
    model.save('first_model')
    #model.evaluate(train_data)

    outputs = encoder.predict(train_data)
    np.save('outputs.npy', outputs)

    image = test_data[np.random.randint(99)].reshape(1,96,192,1)

    im2  = model.predict(image)

    plt.figure()
    plt.subplot(1,2,1) 
    plt.imshow(image[0,:,:,0], cmap = 'coolwarm')
    plt.subplot(1,2,2)
    plt.imshow(im2[0,:,:,0], cmap = 'coolwarm')
    plt.show()
    plt.close()

    

if __name__ == '__main__':
    print(get_data().shape)
    classifier()


