
import configparser
from multiprocessing import Process, Value, Array
from ctypes import c_char_p

import speech_recognition as sr

_stopped = False

# Google Speech Recognition process
def _speech_rec_func(recognized_speech_a, activation_tokens, enable_rec_v):
    magic_word_detected = False
    r = sr.Recognizer()

    global _stopped
    while not _stopped:
        # obtain audio from the microphone
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            rec_speech = r.recognize_google(audio)
            print(f"Google Speech Recognition thinks you said \"{rec_speech}\"")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            continue
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            continue

        recognized_speech_a.value = rec_speech

        # Activation word recognition
        magic_words = set([m.lower() for m in activation_tokens])
        results = set([x.lower() for x in [rec_speech]])

        magic_word_detected = len((magic_words).intersection(results)) > 0
        if magic_word_detected:
            enable_rec_v.value = not enable_rec_v.value

class SpeechRecognizer(object):

    def __init__(self):
        super(SpeechRecognizer, self).__init__()

        # Config parser
        cfg = configparser.ConfigParser()
        cfg.read('config.ini')
        self._activation_tokens = cfg['SPEECH']['ActivationTokens'].split(',')

        # Init shared values
        self.enable_rec_v = Value('b', False)
        self.recognized_speech_a = Array(c_char_p, '')
        self._speech_rec_p = None

        global _stopped
        _stopped = False

    def start(self):
        if self._speech_rec_p is None:
            global _stopped
            _stopped = False
            self._speech_rec_p = Process(target=_speech_rec_func,
                                         args=(self.recognized_speech_a, self._activation_tokens, self.enable_rec_v))
            self._speech_rec_p.start()

    def terminate(self):
        if self._speech_rec_p is not None:
            global _stopped
            _stopped = True
            self._speech_rec_p.terminate()
            self._speech_rec_p = None

