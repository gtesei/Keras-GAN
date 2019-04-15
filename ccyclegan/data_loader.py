import scipy
from glob import glob
import numpy as np
import pandas as pd 

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128,1),path_csv=None):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.img_vect_train = None 
        self.img_vect_test = None 
        self.lab_vect_train = None 
        self.lab_vect_test = None 
        self.path_csv = path_csv 
        self._load_internally()
    
    def _load_internally(self):
        print(">> loading "+str(self.dataset_name)+" ...") 
        
        if self.dataset_name == 'fer2013':
            if self.path_csv is None:
                raw_data = pd.read_csv('./datasets/fer2013.csv')
            else: 
                raw_data = pd.read_csv(self.path_csv)
        else:
            raise Exception("dataset not supported:"+str(self.dataset_name))
        
        n_train = np.sum(raw_data['Usage'] == 'Training')
        n_test = np.sum(raw_data['Usage'] != 'Training')
        assert n_train + n_test == len(raw_data)
        
        self.img_vect_train = np.zeros( (n_train,self.img_res[0],
                                         self.img_res[1],self.img_res[2]) , 'float32')
        self.img_vect_test = np.zeros( (n_test,self.img_res[0],
                                         self.img_res[1],self.img_res[2]) , 'float32')
        self.lab_vect_train = np.zeros( n_train , 'int32' )
        self.lab_vect_test = np.zeros( n_test , 'int32' )
        
        i_train , i_test = 0,0
        for i in range(len(raw_data)):
            img = raw_data["pixels"][i] 
            x_pixels = np.array(img.split(" "), 'float32')
            x_pixels = x_pixels/127.5 - 1.
            x_pixels = x_pixels.reshape(self.img_res)
            us = raw_data["Usage"][i] 
            if us == 'Training':
                self.img_vect_train[i_train] = x_pixels
                self.lab_vect_train[i_train] = int(raw_data["emotion"][i]) 
                i_train = i_train + 1
            else:
                self.img_vect_test[i_test] = x_pixels
                self.lab_vect_test[i_test] = int(raw_data["emotion"][i]) 
                i_test = i_test + 1
            
        assert i_train == len(self.img_vect_train) 
        assert i_train == len(self.lab_vect_train) 
        assert i_test == len(self.lab_vect_test) 
        assert i_test == len(self.img_vect_test) 
                
    def load_data(self, batch_size=1, is_testing=False):
        idx = np.random.choice(self.img_vect_test.shape[0],size=batch_size)
        if is_testing: 
            batch_images = self.img_vect_test[idx]
            labels = self.lab_vect_test[idx]
        else:
            batch_images = self.img_vect_train[idx]
            labels = self.lab_vect_train[idx]
        if is_testing:
            return labels , batch_images
        for i in range(batch_size):
            if np.random.random() > 0.5:
                batch_images[i] = np.fliplr(batch_images[i])
        ## TODO sampling of true false positives (e.g. false happy true faces)
        return labels , batch_images

    def load_batch(self, batch_size=1, is_testing=False):
        if is_testing:
            raise Exception("not supported")
        n_batches = int(len(self.img_vect_train) / batch_size)
        total_samples = n_batches * batch_size
        for i in range(n_batches):
            idx = np.random.choice(self.img_vect_test.shape[0],size=batch_size)
            if is_testing: 
                batch_images = self.img_vect_test[idx]
                labels = self.lab_vect_test[idx]
            else:
                batch_images = self.img_vect_train[idx]
                labels = self.lab_vect_train[idx]
            for i in range(batch_size):
                if np.random.random() > 0.5:
                    batch_images[i] = np.fliplr(batch_images[i])
            yield labels , batch_images


if __name__ == 'main':
    dl = DataLoader(dataset_name='fer2013',img_res=(48,48,1))
