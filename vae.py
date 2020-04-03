from netCDF4 import Dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

tf.compat.v1.disable_eager_execution()
def get_data():
    R1 =  Dataset('data/T2m_R1_ym_1stMill.nc', 'r')
    temperature = R1.variables['T2m'][:]
    lat = R1.variables['lat'][:]
    lon = R1.variables['lon'][:]

    return temperature



def classifier(latent_dimension=32):
    def sample_z(args):
        mu, log_sigma = args


    data = get_data()
    data = data.reshape(999,96,192,1)

    train_data = data[:900]
    test_data = data[900:]

    print(train_data.shape)
    print(test_data.shape)


    inputs =  tf.keras.layers.Input(shape=(96,192,1))
    encoder  =tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = (4,4),
            strides = (4,4),
            activation = 'relu')(inputs)
    layer1 = tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = (4,4),
            strides = (4,4),
            activation = 'relu')(encoder)

    layer2 = tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = (3,4),
            strides = (3,4),
            activation = 'relu')(layer1)
    q = tf.keras.layers.Conv2D(
            filters = 128,
            kernel_size = (2,3),
            strides = (2,3),
            activation = 'relu')(layer2)

    mu = tf.keras.layers.Dense(latent_dimension, activation=tf.keras.activations.linear)(q)
    log_sigma = tf.keras.layers.Dense(latent_dimension, activation=tf.keras.activations.linear)(q)
    eps = tf.keras.backend.random_normal(shape=(1, latent_dimension), mean=0, stddev=1)
    # TODO something strange
    z = tf.keras.layers.Add()([mu, K.exp(log_sigma / 2)])
    z = tf.keras.layers.Multiply()([z, eps])


    decoder = tf.keras.layers.Dense(128)(z)
    decoder =tf.keras.layers.Conv2DTranspose(
            filters = 64,
            kernel_size = (2,3),
            strides = (2,3),
            activation = 'relu')(decoder)
    decoder = tf.keras.layers.Conv2DTranspose(
            filters = 64,
            kernel_size = (3,4),
            strides = (3,4),
            activation = 'relu')(decoder)
    decoder = tf.keras.layers.Conv2DTranspose(
            filters = 64,
            kernel_size = (4,4),
            strides = (4,4),
            activation = 'relu')(decoder)
    decoder = tf.keras.layers.Conv2DTranspose(
            filters = 1,
            kernel_size = (4,4),
            strides = (4,4),
            activation = 'relu')(decoder)


    vae = tf.keras.Model(inputs, decoder)
    encoder_mu = tf.keras.Model(inputs, mu)
    encoder_sigma = tf.keras.Model(inputs, log_sigma)
    def vae_loss(y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        # E[log P(X|z)]
        recon = K.sqrt(K.mean(K.square(y_pred - y_true), axis=1))
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = - 0.5 * K.sum(- K.exp(log_sigma) - K.square(mu) + 1. + log_sigma, axis=1)

        return recon +0.01 * kl

    vae.compile(loss = vae_loss, optimizer = 'Adam', metrics = ['mse'])

    vae.fit(train_data, train_data, batch_size = 8, epochs = 100)
    #vae.save('vae')
    #encoder.save("vae_encoder")
    #model.evaluate(train_data)

    outputs = encoder_mu.predict(data)
    np.save(f'vae_mu_{latent_dimension}.npy', outputs)
    outputs = encoder_sigma.predict(data)
    np.save(f'vae_sigma_{latent_dimension}.npy', outputs)

    image = test_data[np.random.randint(99)].reshape(1,96,192,1)

    im2 = vae.predict(image)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image[0,:,:,0], cmap = 'coolwarm')
    plt.subplot(1,2,2)
    plt.imshow(im2[0,:,:,0], cmap = 'coolwarm')
    plt.savefig(f"plot_{latent_dimension}")
    plt.close()



if __name__ == '__main__':
    print(get_data().shape)
    for dimension in [4,8,16,64,128,32]:
        classifier(latent_dimension=dimension)


