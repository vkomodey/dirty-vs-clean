from enum import Enum

classes = ['cleaned', 'dirty']
classes_vocab = {
    'cleaned': 0,
    'dirty': 1
}

class TrainPhase(Enum):
    train = 'train'
    validation = 'val'

class ImageClasses(Enum):
    cleaned = 'cleaned'
    dirty = 'dirty'
