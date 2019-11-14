
import configparser

import cv2
import numpy as np
import tensorflow as tf

class InferenceManager(object):
    def __init__(self, detection_only=False):

        self.detection_only = detection_only

        # Config parser
        cfg = configparser.ConfigParser()
        cfg.read('config.ini')

        detection_model_path = cfg['MODELS']['DetectionPath']
        classification_model_path = cfg['MODELS']['ClassificationPath']

        # Loading frozen hand detection model
        self._dtc_graph, self._dtc_session = self._load_detection_graph(detection_model_path)

        if not detection_only:
            # Loading Keras gesture classification model
            self._cls_model, self._cls_graph, self._cls_session = self._load_classification_graph(classification_model_path)

    # Load frozen hand detection and Keras gesture classification model
    def _load_detection_graph(self, path):
        graph = tf.Graph()

        with graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            session = tf.Session(graph=graph)

        return graph, session

    def _load_classification_graph(self, path):

        import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
        import keras

        graph = tf.Graph()

        with graph.as_default():
            session = tf.Session()

            with session.as_default():
                model = keras.models.load_model(path)
                graph = tf.get_default_graph()

        return model, graph, session

    # Hand detector
    def detect(self, image_np):

        # Input Tensor
        image_tensor = self._dtc_graph.get_tensor_by_name('image_tensor:0')

        # Output Tensors
        dtc_boxes = self._dtc_graph.get_tensor_by_name('detection_boxes:0')
        dtc_scores = self._dtc_graph.get_tensor_by_name('detection_scores:0')

        # Perform inference
        image_np_expanded = np.expand_dims(image_np, axis=0)
        feed_dict = {image_tensor: image_np_expanded}
        boxes, scores = self._dtc_session.run([dtc_boxes, dtc_scores], feed_dict)

        return np.squeeze(boxes), np.squeeze(scores)

    def classify(self, image):

        if self.detection_only:
            raise Exception('Classification model not loaded for this instance.')

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.flip(image, 1)

        # Reshape
        res = cv2.resize(image, (28,28), interpolation=cv2.INTER_AREA)

        # Convert to float values between 0. and 1.
        res = res.astype(dtype="float64")
        res = res / 255
        res = np.reshape(res, (1, 28, 28, 1))

        # Perform Inference
        with self._cls_graph.as_default():
            with self._cls_session.as_default():
                return self._cls_model.predict(res)[0]