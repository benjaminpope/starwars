from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class gan():
    ''' A GAN for Kepler lightcurves - minimum Possible
    working example.

    Should accept Kepler lc's of 48*2 data points (~1 day)
    '''
    def __init__(self,ndays=1,LSTM=False):
        self.LSTM = LSTM
        self.lc_size = 48*ndays
        self.gen_shape = {'d1': 32*ndays,
                          'd2': 64*ndays,
                          'd3': 128*ndays
                         }
        self.d_shape = {'d1': 64*ndays,
                        'd2': 32*ndays
                        }
        self.X_train = []
        self.img_counter = 0

    def noise(self, nd):
        if self.LSTM:
            return np.random.uniform(-1, 1, (nd, self.lc_size,1))
        else:
            return np.random.uniform(-1, 1, (nd, self.lc_size))

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
        self.d_model.add(Dropout(0.3))
        self.d_model.add(Dense(self.gen_shape['d2']))
        self.d_model.add(LeakyReLU(alpha=0.2))
        self.d_model.add(Dropout(0.3))
        self.d_model.add(Dense(1, activation='sigmoid'))
        print(self.d_model.summary())

    def model(self):
        gen_optimizer = Adam(lr=0.0001, beta_1=0.5)
        disc_optimizer = Adam(lr=0.0001, beta_1=0.5)
        self.d_model.compile(loss='binary_crossentropy',
                             optimizer=disc_optimizer,
                             metrics=['accuracy'])
        self.gen_model.compile(loss='binary_crossentropy',
                               optimizer=gen_optimizer)
        self.d_model.trainable=False
        z = Input(shape=(self.lc_size, ))
        img = self.gen_model(z)
        real = self.d_model(img)
        self.combined = Model(inputs=z, outputs=real)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=gen_optimizer)

    def make_LSTM_generator(self):
        self.gen_model = Sequential()
        self.gen_model.add(CuDNNLSTM(units=512, return_sequences=True,
                             input_shape=(self.lc_size, 1)))
        self.gen_model.add(LeakyReLU(0.2))
        self.gen_model.add(BatchNormalization(momentum=0.8))
        self.gen_model.add(Bidirectional(CuDNNLSTM(units=128)))
        self.gen_model.add(BatchNormalization(momentum=0.8))
        self.gen_model.add(Dense(512))
        self.gen_model.add(LeakyReLU(0.2))
        self.gen_model.add(BatchNormalization(momentum=0.8))
        self.gen_model.add(Dense(self.lc_size, activation='tanh'))
        self.gen_model.add(Reshape((self.lc_size, 1)))
        print(self.gen_model.summary())

    def make_LSTM_discriminator(self):
        self.d_model = Sequential()
        self.d_model.add(CuDNNLSTM(units=512, return_sequences=True,
                             input_shape=(self.lc_size, 1)))
        self.d_model.add(Bidirectional(CuDNNLSTM(units=512)))
        self.d_model.add(Dense(512))
        self.d_model.add(LeakyReLU(0.2))
        self.d_model.add(Dense(256))
        self.d_model.add(LeakyReLU(0.2))
        self.d_model.add(Dense(1, activation='sigmoid'))
        print(self.d_model.summary())

    def LSTM_model(self):
        gen_optimizer = Adam(lr=0.005, beta_1=0.5)
        disc_optimizer = Adam(lr=0.005, beta_1=0.5)
        self.d_model.compile(loss='binary_crossentropy',
                             optimizer=disc_optimizer,
                             metrics=['accuracy'])
        self.gen_model.compile(loss='binary_crossentropy',
                               optimizer=gen_optimizer)
        self.d_model.trainable=False
        z = Input(shape=(self.lc_size, 1))
        img = self.gen_model(z)
        real = self.d_model(img)
        self.combined = Model(inputs=z, outputs=real)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=gen_optimizer)

    def train(self, epochs=100, batch_size=500, LSTM=False):
        num_examples = self.X_train.shape[0]
        num_batches = int(num_examples / float(batch_size))
        half_batch = int(batch_size / 2)
        self.losses = np.zeros([epochs, 2])
        for epoch in range(epochs):
            for batch in range(num_batches):
                # Noise images
                noise = self.noise(half_batch)
                fake_images = self.gen_model.predict(noise)
                fake_labels = np.random.uniform(0.0, 0.1, half_batch)
                # Real images ...
                idx = np.random.randint(0, self.X_train.shape[0], half_batch)
                real_images = self.X_train[idx]
                if LSTM:
                    real_images = real_images.reshape(half_batch, self.lc_size, 1)
                real_labels = np.random.uniform(0.9, 1.0, half_batch)
                # Train the discriminator (real classified as ones and generated as zeros)
                self.d_model.trainable=True
                d_loss_real = self.d_model.train_on_batch(real_images, real_labels)
                d_loss_fake = self.d_model.train_on_batch(fake_images, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                self.d_model.trainable=False

                # Train the combined model
                noise = self.noise(batch_size)
                g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))
            self.losses[epoch, 0] = d_loss[0]
            self.losses[epoch, 1] = g_loss
            # Plot the progress
            if epoch % 10 == 0:
                    print(f"Epoch: {epoch} D loss: {d_loss[0]} G loss: {g_loss}")
                    self.save_imgs(epoch, batch)

    def plot_losses(self):
        fig, ax = plt.subplots()
        self.dl = ax.plot(self.losses[:,0], label=['D loss'])
        self.gl = ax.plot(self.losses[:,1], label=['G loss'])
        plt.show()

    def get_data(self, nd=1000):
        x = np.linspace(0, self.lc_size, self.lc_size)
        self.X_train = np.zeros([nd, self.lc_size])
        p = 80
        fs, ps, ns = 0.001, 1.0, 0.001
        for i in range(nd):
            self.X_train[i, :] = np.sin(x/(p + np.random.rand()*fs) + np.random.rand()*ps) \
                                 + np.random.randn(self.lc_size)*ns
            self.X_train[i, :] /= np.max(np.abs(self.X_train[i, :]))
        print(f'Data shape : {self.X_train.shape}')

    def make_img(self, nd=1):
        noise = self.noise(nd)
        fake_images = self.gen_model.predict(noise)
        print(f'Fake images shape : {fake_images.shape}')
        fig, ax = plt.subplots()
        ax.plot(fake_images.T)

    def plot_some_data(self, nd=10):
        if self.X_train == []:
            self.get_data(nd=10)
        fig, ax = plt.subplots()
        ax.plot(self.X_train[:nd, :].T)

    def save_imgs(self, epoch, batch):
        r = 5
        noise = self.noise(r)
        gen_imgs = self.gen_model.predict(noise)

        fig, ax = plt.subplots()
        cnt = 0
        for i in range(r):
            ax.plot(gen_imgs[cnt, :], alpha=0.5)
            cnt += 1
        lbl = str(int(self.img_counter))
        fig.savefig(f"lc_{lbl}.png" )
        self.img_counter += 1
        plt.close()

if __name__ == "__main__":
    LSTM = False
    starwars = gan(ndays=16,LSTM=LSTM)
    starwars.plot_some_data()
    plt.show()
    if LSTM:
        starwars.make_LSTM_generator()
        starwars.make_LSTM_discriminator()
        starwars.LSTM_model()
    else:
        starwars.make_generator()
        starwars.make_discriminator()
        starwars.model()
    starwars.get_data(nd=20000)
    starwars.train(epochs=200)
    starwars.plot_losses()
