from keras.layers import *
from keras.models import Sequential, Model

class gan():
    ''' A GAN for Kepler lightcurves - minimum Possible
    working example.

    Should accept Kepler lc's of 48 data points (~1 day)
    '''
    def __init__(self):
        self.lc_size = 48
        self.gen_shape = {'d1': 16,
                          'd2': 32,
                          'd3': 64
                         }
        self.d_shape = {'d1': 32,
                        'd2': 16
                        }

    def make_generator(self):
        self.gen_model = Sequential()
        self.gen_model.add(Dense(self.gen_shape['d1'],
                           input_shape=(None, self.lc_size)))
        self.gen_model.add(LeakyReLU(alpha=0.2))
        self.gen_model.add(BatchNormalization(momentum=0.8))
        self.gen_model.add(Dense(self.gen_shape['d2']))
        self.gen_model.add(LeakyReLU(alpha=0.2))
        self.gen_model.add(BatchNormalization(momentum=0.8))
        self.gen_model.add(Dense(self.gen_shape['d3']))
        self.gen_model.add(LeakyReLU(alpha=0.2))
        self.gen_model.add(BatchNormalization(momentum=0.8))
        self.gen_model.add(Dense(self.lc_size, activation='tanh'))
        print(self.gen_model.summary())

    def make_discriminator(self):
        self.d_model = Sequential()
        self.d_model.add(Dense(self.gen_shape['d1'],
                           input_shape=(None, self.lc_size)))
        self.d_model.add(LeakyReLU(alpha=0.2))
        self.d_model.add(Dense(self.gen_shape['d2']))
        self.d_model.add(LeakyReLU(alpha=0.2))
        self.d_model.add(Dense(1, activation='sigmoid'))
        print(self.d_model.summary())

    def model(self):
        z = Input(shape=(self.lc_size, ))
        img = self.gen_model(z)
        self.d_model.trainable = False
        real = self.d_model(img)
        self.combined = Model(z, real)
        gen_optimizer = Adam(lr=0.0002, beta_1=0.5)
        disc_optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.d_model.compile(loss='binary_crossentropy',
                             optimizer=disc_optimizer,
                             metrics=['accuracy'])
        self.gen_model.compile(loss='binary_crossentropy',
                               optimizer=gen_optimizer)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=gen_optimizer)

    def training(self):
        epochs = 10
        batch_size = 100
        num_examples = self.X_train.shape[0]
        num_batches = int(num_examples / float(batch_size))
        half_batch = int(batch_size / 2)
        for epoch in range(epochs + 1):
            for batch in range(num_batches):
                # Noise images
                noise = np.random.normal(0, 1, (half_batch, self.lc_size))
                fake_images = self.gen_model.predict(noise)
                fake_labels = np.zeros((half_batch, 1)) # label is 0
                # Real images ...
                idx = np.random.randint(0, self.X_train.shape[0], half_batch)
                real_images = X_train[idx]
                real_labels = np.ones((half_batch, 1)) # label is 1
                # Train the discriminator (real classified as ones and generated as zeros)
                d_loss_real = self.d_model.train_on_batch(real_images, real_labels)
                d_loss_fake = self.d_model.train_on_batch(fake_images, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                noise = np.random.normal(0, 1, (batch_size, 100))
                # Train the generator
                g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
                # Plot the progress
                print(f"Epoch: {epoch} D loss: {d_loss[0]} G loss: {g_loss}")

    def get_data(self):
        self.X_train = [] # TODO

if __name__ == "__main__":
    starwars = gan()
    starwars.make_generator()
    starwars.make_discriminator()
    starwars.model()
