
import time
import json
import configparser

import numpy as np

class DecisionMaker(object):
    def __init__(self):

        self._last_brightness = 255
        self._inf_queue = []
        self._t_last_inference = 0

        # Config parser
        cfg = configparser.ConfigParser()
        cfg.read('config.ini')

        self._gestures_to_actions = cfg['DECISIONS']['GesturesToActions'].split(',')
        self._action_to_payload = json.loads(cfg['DECISIONS']['ActionAssociatedPayload'])
        self._gestures_payl_rules = cfg['DECISIONS']['GesturesPayloadRules'].split(',')

        self._min_inf_queue_len = int(cfg['DECISIONS']['MinInferenceQueueLength'])
        self._max_inf_queue_len = int(cfg['DECISIONS']['MaxInferenceQueueLength'])
        self._decision_thresh = float(cfg['DECISIONS']['DecisionThreshold'])
        self._time_thresh = float(cfg['DECISIONS']['EmptyQueueTimeThreshold'])
        self._brightness_delta = int(cfg['DECISIONS']['BrightnessDelta'])

        # Convert rules to int
        for i in range(len(self._gestures_payl_rules)):
            try:
                self._gestures_payl_rules[i] = int(self._gestures_payl_rules[i])
            except:
                pass

    def _perform_decision(self, inferences):

        # Check last inference time and empty queue if above time threshold
        t_now = time.time()
        if (t_now - self._t_last_inference) > self._time_thresh:
            self._inf_queue = []
        self._t_last_inference = t_now

        # Add inference to queue
        self._inf_queue.append(inferences.argmax())

        # check for enough inferences in queue to perform decision
        if len(self._inf_queue) >= self._min_inf_queue_len:
            num_classes = len(self._gestures_to_actions)
            len_queue = len(self._inf_queue)
            inf_percent = [self._inf_queue.count(i) / len_queue for i in range(num_classes)]
            inf_percent = np.array(inf_percent) * 100

            inf_max = inf_percent.max()
            inf_argmax = inf_percent.argmax()

            # If the queue contains x% of the same inference
            if inf_max > self._decision_thresh:
                self._inf_queue = []
                return self._gestures_to_actions[inf_argmax], inf_argmax
            # Else if the queue is contains more than x, empty it
            elif len(self._inf_queue) > self._max_inf_queue_len:
                self._inf_queue = []

        return None

    def inferences_to_action(self, inferences):

        decision = self._perform_decision(inferences)

        if decision is not None:
            payload = {'parameters': {}}

            action, gesture = decision
            if action == '':
                return None

            payload_key = self._action_to_payload[action]
            payload_rule = self._gestures_payl_rules[gesture]

            if payload_key == 'brightness':
                if type(payload_rule) is int:
                    payload['parameters'][payload_key] = payload_rule
                    self._last_brightness = payload_rule
                elif payload_rule == 's':
                    payload['parameters'][payload_key] = self._last_brightness
                elif payload_rule == 'i':
                    variation = min(255, self._last_brightness + self._brightness_delta)
                    payload['parameters'][payload_key] = variation
                    self._last_brightness = variation
                elif payload_rule == 'd':
                    variation = max(0, self._last_brightness - self._brightness_delta)
                    payload['parameters'][payload_key] = variation
                    self._last_brightness = variation

            return action, payload


        return None