
import configparser
from multiprocessing import Pool, Queue
from multiprocessing.managers import BaseManager

import utils.image as imgutils

def _worker_func(shared_model, enable_rec_v, input_q, output_q, cropped_output_q, inferences_q, cap_params):
    while True:
        frame = input_q.get()
        if (frame is not None):
            if (type(enable_rec_v) is bool and enable_rec_v) or enable_rec_v.value:
                # Hand detection
                boxes, scores = shared_model.detect(frame)

                # Get region of interest
                dtc_result = imgutils.get_box_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
                                                    scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)

                # Draw bounding boxes
                imgutils.draw_box_on_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
                                           scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)

                # Hand gesture classification
                if dtc_result is not None:
                    cls_result = shared_model.classify(dtc_result)
                    inferences_q.put(cls_result)

                # Add frame annotated with bounding box to queue
                cropped_output_q.put(dtc_result)
            output_q.put(frame)
        else:
            output_q.put(frame)

class WorkersManager(object):
    def __init__(self, num_workers, queues_size, enable_rec_v, inference_manager_class):
        super(WorkersManager, self).__init__()

        self._rec_pool = None
        self._num_workers = num_workers
        self._enable_rec_v = enable_rec_v

        # Config parser
        cfg = configparser.ConfigParser()
        cfg.read('config.ini')
        self._score_threshold = float(cfg['MODELS']['DetectionScoreThreshold'])

        # Init queues
        self.input_q = Queue(maxsize=queues_size)
        self.output_q = Queue(maxsize=queues_size)
        self.cropped_output_q = Queue(maxsize=queues_size)
        self.inferences_q = Queue(maxsize=queues_size)

        # Init shared InferenceManager
        BaseManager.register('InferenceManager', inference_manager_class)
        self._manager = BaseManager()
        self._manager.start()
        self.shared_inference_manager = self._manager.InferenceManager()

    def start(self, cap_params):
        cap_params['score_thresh'] = self._score_threshold
        self._rec_pool = Pool(self._num_workers, _worker_func,
                              (self.shared_inference_manager, self._enable_rec_v, self.input_q, self.output_q, self.cropped_output_q, self.inferences_q, cap_params))

    def terminate(self):
        self._rec_pool.terminate()
        self._rec_pool = None