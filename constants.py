from enum import Enum

classes = ['cleaned', 'dirty']
classes_vocab = {
    'cleaned': 0,
    'dirty': 1
}
num_to_class_vocab = {
    0: 'cleaned',
    1: 'dirty'
}

class TrainPhase(Enum):
    train = 'train'
    validation = 'val'

class ImageClasses(Enum):
    cleaned = 'cleaned'
    dirty = 'dirty'
