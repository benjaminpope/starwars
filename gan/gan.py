from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class gan():
    ''' A GAN for Kepler lightcurves - minimum Possible
    working example.

    Should accept Kepler lc's of 48 data points (~1 day)
    '''
    def __init__(self):
        self.lc_size = 48
        self.gen_shape = {'d1': 32,
                          'd2': 64,
                          'd3': 128
                         }
        self.d_shape = {'d1': 64,
                        'd2': 32
                        }

    def make_generator(self):
        self.gen_model = Sequential()
        self.gen_model.add(Dense(self.gen_shape['d1'],
                           input_dim=self.lc_size))
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
                           input_dim=self.lc_size))
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
        gen_optimizer = Adam(lr=0.0001, beta_1=0.1)
        disc_optimizer = Adam(lr=0.0001, beta_1=0.1)
        self.d_model.compile(loss='binary_crossentropy',
                             optimizer=disc_optimizer,
                             metrics=['accuracy'])
        self.gen_model.compile(loss='binary_crossentropy',
                               optimizer=gen_optimizer)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=gen_optimizer)

    def train(self, epochs=100, batch_size=100):
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
                real_images = self.X_train[idx]
                real_labels = np.ones((half_batch, 1)) # label is 1
                # Train the discriminator (real classified as ones and generated as zeros)
                d_loss_real = self.d_model.train_on_batch(real_images, real_labels)
                d_loss_fake = self.d_model.train_on_batch(fake_images, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                noise = np.random.normal(0, 1, (batch_size, self.lc_size))
                # Train the generator
                g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))
                # Plot the progress
                print(f"Epoch: {epoch} D loss: {d_loss[0]} G loss: {g_loss}")

    def get_data(self, nd=1000):
        x = np.linspace(0, 6.0*np.pi, self.lc_size)
        self.X_train = np.zeros([nd, self.lc_size])
        for i in range(nd):
            self.X_train[i, :] = np.sin(x)# + np.random.rand()*np.pi*2)

    def make_img(self):
        noise = np.random.normal(0, 1, (10, self.lc_size))
        fake_images = self.gen_model.predict(noise)
        fig, ax = plt.subplots()
        ax.plot(fake_images.T)

    def plot_some_data(self):
        self.get_data(nd=10)
        fig, ax = plt.subplots()
        ax.plot(self.X_train.T)

if __name__ == "__main__":
    starwars = gan()
    starwars.make_generator()
    starwars.make_discriminator()
    starwars.model()
    starwars.plot_some_data()
    plt.show()
    starwars.get_data()
    for i in range(10):
        starwars.train()
        starwars.make_img()
        plt.show()
    starwars.plot_some_data()
    plt.show()
