import os
import pickle
import tensorflow as tf
from linora.sample import ImageDataset

class params(object):
    model = None
    preprocess_function = None
    feature_folder = None
    features = None
    image_paths = None
    
class ImageSearch:
    def __init__(self, model=None, preprocess_function=None, feature_folder='./image_feature'):
        self.params = params
        if model is None:
            base_model = tf.keras.applications.VGG16(weights='imagenet')
            self.params.model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        else:
            self.params.model = model
        self.params.preprocess_function = preprocess_function
        self.params.feature_folder = feature_folder
        if not tf.io.gfile.exists(self.params.feature_folder):
            tf.io.gfile.makedirs(self.params.feature_folder)
    
    def reset_feature_folder(self):
        if tf.io.gfile.exists(self.params.feature_folder):
            tf.io.gfile.rmtree(self.params.feature_folder)
        tf.io.gfile.makedirs(self.params.feature_folder)
    
    def extract(self, image_file):
        tf.debugging.assert_type(image_file, tf.dtypes.string, f'{image_file} must be tf.dtypes.string.')
        if self.params.preprocess_function is None:
            image = tf.io.read_file(image_file)
            image = tf.io.decode_image(image)
            image = tf.image.resize(image, size=(224,224))
            image = tf.expand_dims(image, axis=0)
            image = tf.keras.applications.vgg16.preprocess_input(image)
        else:
            image = self.params.preprocess_function(image_file)
        feature = self.params.model.predict(image)[0]
        feature = feature / tf.linalg.norm(feature)
        return feature

    def save(self, feature, image_file):
        feature_file = self._path_join(self.params.feature_folder+'/'+self._path_join(image_file).replace(':', 'aa__aa').replace('/', 'oo__oo')+'.pkl')
        with tf.io.gfile.GFile(feature_file, 'wb') as f:
            pickle.dump(feature, f)
        
    def save_feature(self, folder_name, image_format=['png', 'jpg', 'jpeg']):
        tf.debugging.assert_equal(True, tf.io.gfile.isdir(folder_name), f'{folder_name} must be folder')
        image_list = ImageDataset(folder_name, image_format).data.image
        p = tf.keras.utils.Progbar(len(image_list))
        for path in image_list:
            self.save(self.extract(path), path)
            p.add(1)
    
    def load_feature(self):
        self.params.features = []
        self.params.image_paths = []
        glob_path = tf.io.gfile.glob(self._path_join(self.params.feature_folder+'/*.pkl'))
        p = tf.keras.utils.Progbar(len(glob_path))
        for path in glob_path:
            with tf.io.gfile.GFile(path, 'rb') as f:
                self.params.features.append(pickle.load(f))
            self.params.image_paths.append(os.path.basename(path).replace('aa__aa', ':').replace('oo__oo', '/')[:-4])
            p.add(1)
    
    def search(self, image_file, top_n=30):
        query = self.extract(image_file)
        dists = tf.linalg.norm(self.params.features - query, axis=1)  # Do search
        ids = tf.argsort(dists)[:top_n]
        scores = [(dists[i], self.params.image_paths[i]) for i in ids]
        return scores
    
    def _path_join(self, path):
        return eval(repr(path).replace("\\", '/').replace("//", '/'))
    
