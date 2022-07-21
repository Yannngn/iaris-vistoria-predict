import detect, classify
from utils import get_model_instance_detection, get_model_instance_classification, save_log

#   params
##      device
##      model_class

##      dectection only
###         num_class

##      classification only
###         uniques
NUM_CLASSES = 2
MIN_SCORE = .5
DEVICE = 'gpu'
COUNT = 0

class Predictor(object):
    def __init__(self, log) -> None:
        self.log = log
        self.params = {}
        self.results = []
        self.model, self.transform = self.get_model()
        pass
    
    def get_model(self):
        pass

    def save_results(self) -> None:
        save_log(self.results, self.params)
        pass   
    
class Detector(Predictor):
    def get_model(self):
        return get_model_instance_detection(self.params['num_class']), None
    
    def call(self, images) -> None:
        if not self.model:
            self.model, self.transform = self.get_model()
        self.results = detect.predict(self.model, images, self.params)
        pass

    def print_results(self) -> None:
        for result in self.results: print(f"{result['image']}: {result['scores']}")
        pass
        
class Classifier(Predictor):
    def get_model(self):
        return get_model_instance_classification(self.params)
            
    def call(self, images) -> None:
        if not self.model:
            self.model, self.transform = self.get_model()
        self.results = classify.predict(self.model, self.transform, images, self.params)
        pass
    
    def print_results(self) -> None:
        for result in self.results: print(f"{result['image']}: {result[self.params['model_class']]}")
        pass
    
class ClassifyCor(Classifier):
    def __init__(self) -> None:
        self.params = {"model_class": "cor",
                       "device": "gpu"}
        self.results = []
        self.model = None
        pass

class ClassifyMarca(Classifier):
    def __init__(self) -> None:
        self.params = {"model_class": "marca",
                       "device": "gpu"}
        self.results = []
        self.model = None
        pass

class ClassifyModelo(Classifier):
    def __init__(self) -> None:
        self.params = {"model_class": "modelo",
                       "device": "gpu"}
        self.results = []
        self.model = None
        pass

'''     Não está funcionando

class ClassifyFarol(Classifier):
    def __init__(self) -> None:
        self.params = {"model_class": "farol",
                       "uniques": [False, True], # confirmar
                       "device": "gpu"}
        self.results = []
        self.model = None
        pass
'''

class DetectEmblema(Detector):
    def __init__(self) -> None:
        self.params = {"model_class": "emblema",
                       "num_class": NUM_CLASSES,
                       "device": DEVICE,
                       "min_score": MIN_SCORE,
                       "count": COUNT}
        self.results = []
        self.model = None
        pass

class DetectFarol(Detector):
    def __init__(self) -> None:
        self.params = {"model_class": "farol",
                       "num_class": NUM_CLASSES,
                       "device": DEVICE,
                       "min_score": MIN_SCORE,
                       "count": COUNT}
        self.results = []
        self.model = None
        pass

class DetectRoda(Detector):
    def __init__(self) -> None:
        self.params = {"model_class": "roda",
                       "num_class": NUM_CLASSES,
                       "device": DEVICE,
                       "min_score": MIN_SCORE,
                       "count": COUNT}
        self.results = []
        self.model = None
        pass

class DetectMilha(Detector):
    def __init__(self) -> None:
        self.params = {"model_class": "milha",
                       "num_class": NUM_CLASSES,
                       "device": DEVICE,
                       "min_score": MIN_SCORE,
                       "count": COUNT}
        self.results = []
        self.model = None
        pass

class DetectParabrisa(Detector):
    def __init__(self) -> None:
        self.params = {"model_class": "parabrisa",
                       "num_class": NUM_CLASSES,
                       "device": DEVICE,
                       "min_score": MIN_SCORE,
                       "count": COUNT}
        self.results = []
        self.model = None
        pass
    
class DetectRetrovisor(Detector):
    def __init__(self) -> None:
        self.params = {"model_class": "retrovisor",
                       "num_class": NUM_CLASSES,
                       "device": DEVICE,
                       "min_score": MIN_SCORE,
                       "count": COUNT}
        self.results = []
        self.model = None
        pass
    
class DetectLampadaTras(Detector):
    def __init__(self) -> None:
        self.params = {"model_class": "lampada_tras",
                       "num_class": NUM_CLASSES,
                       "device": DEVICE,
                       "min_score": MIN_SCORE,
                       "count": COUNT}
        self.results = []
        self.model = None
        pass
