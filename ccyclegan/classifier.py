from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import Reshape
from keras.layers.merge import concatenate
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
import random 
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


class CCycleGAN():
    def __init__(self,img_rows = 48,img_cols = 48,channels = 1, num_classes=7, latent_dim=99):
        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        ## dict
        self.lab_dict = {0: "Angry", 1: "Disgust" , 2: "Fear" , 3: "Happy" , 4: "Sad" , 5: "Surprise" , 6: "Neutral"}

        # Configure data loader
        self.dataset_name = 'fer2013'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,img_res=self.img_shape)


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d = self.build_discriminator()
        print("******** Discriminator ********")
        self.d.summary()
        self.d.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------
        
    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)
        
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        
        flat_img = Flatten()(img)
        
        model_input = multiply([flat_img, label_embedding])
        d0 = Reshape(self.img_shape)(model_input)

        d1 = d_layer(d0, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        d5 = Dense(128)(validity)
        d6 = LeakyReLU(alpha=0.2)(d5)
        d7 = Dropout(0.4)(d6)
        d77 = Flatten()(d7)
        d8 = Dense(1, activation='sigmoid')(d77)

        return Model([label,img], d8)

    
    def generate_new_labels(self,labels0):
        labels1 = [] 
        for i in range(len(labels0)):
            allowed_values = list(range(0, self.num_classes))
            allowed_values.remove(labels0[i])
            labels1.append(random.choice(allowed_values))
        return np.array(labels1,'int32')

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        for epoch in range(epochs):
            for batch_i, (labels0 , imgs) in enumerate(self.data_loader.load_batch(batch_size=batch_size)):
                labels1 = self.generate_new_labels(labels0)
                labels01 = self.generate_new_labels(labels0)
                
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Train the discriminators (original images = real / translated = Fake)
                d_loss_real = self.d.train_on_batch([labels0,imgs], valid)
                d_loss_real_fake = self.d.train_on_batch([labels01,imgs], fake)
                d_loss = (1/2) * np.add(d_loss_real, d_loss_real_fake)

              
                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    
        

    def sample_images(self, epoch, batch_i):
        t1 = np.ones(self.data_loader.lab_vect_test.shape[0])
        t0 = np.zeros(self.data_loader.lab_vect_test.shape[0])
        t = np.concatenate((t1,t0))
        
        
        labels1_ = self.generate_new_labels(self.data_loader.lab_vect_test)
        
        pred_prob_fake = self.d.predict([labels1_,self.data_loader.img_vect_test])
        pred_prob_valid_ = self.d.predict([self.data_loader.lab_vect_test,self.data_loader.img_vect_test])
        
        pred_probs = np.concatenate((pred_prob_fake.squeeze(),pred_prob_valid_.squeeze()))
        
        preds = pred_probs > 0.5 
        
        acc = accuracy_score(t,preds)
        ll = log_loss(t,pred_probs)
        auc = roc_auc_score(t,pred_probs)
        
        print("Accuracy[test:"+str(self.data_loader.lab_vect_test.shape[0])+"]:",acc)
        print("LogLoss[test:"+str(self.data_loader.lab_vect_test.shape[0])+"]:",ll)
        print("AUC[test:"+str(self.data_loader.lab_vect_test.shape[0])+"]:",auc)
    

if __name__ == '__main__':
    gan = CCycleGAN()
    gan.train(epochs=200, batch_size=64, sample_interval=200)
