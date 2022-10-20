import os
from PIL import Image
import csv
import pickle as pickle
import sys
sys.path.insert(0, '.')
verb_classes = [
    'awaken', 'close', 'cook', 'dress', 'drink', 'eat', 'fix', 'grasp',
    'hold', 'laugh', 'lie', 'make', 'open', 'photograph', 'play',
    'pour', 'put', 'run', 'sit', 'smile', 'sneeze', 'snuggle', 'stand',
    'take', 'talk', 'throw', 'tidy', 'turn', 'undress', 'walk', 'wash',
    'watch', 'work'
]


#replace 'shoe' to 'shoes'
object_classes = [
'None', 'bag', 'bed', 'blanket', 'book', 'box', 'broom', 'chair',
'closet/cabinet', 'clothes', 'cup/glass/bottle', 'dish',
'door', 'doorknob', 'doorway', 'floor', 'food', 'groceries',
'hair', 'hands', 'laptop', 'light', 'medicine', 'mirror', 'paper/notebook', 'phone/camera', 'picture',
'pillow', 'refrigerator', 'sandwich', 'shelf', 'shoes', 'sofa/couch', 'table', 'television', 'towel', 'vacuum', 'window'

]

scene_convert = {'Basement (A room below the ground floor)': 'basement',
                          'Bathroom': 'bathroom',
                          'Bedroom': 'bedroom',
                          'Closet / Walk-in closet / Spear closet': 'closet',
                          'Dining room': 'dining room',
                          'Entryway (A hall that is generally located at the entrance of a house)': 'entryway',
                          'Garage': 'garage',
                          'Hallway': 'hallway',
                          'Home Office / Study (A room in a house used for work)': 'home office',
                          'Kitchen': 'kitchen',
                          'Laundry room': 'laundry room',
                          'Living room': 'living room',
                          'Other': 'other',
                          'Pantry': 'pantry',
                          'Recreation room / Man cave': 'recreation room',
                          'Stairs': 'stairs'}

AID2ACTION = './data/Charades_v1_classes.txt'

c2ov = {0: (9, 8),
        1: (9, 16),
        2: (9, 23),
        3: (9, 25),
        4: (9, 26),
        5: (9, 30),
        6: (12, 1),
        7: (12, 6),
        9: (33, 16),
        10: (33, 18),
        11: (33, 18),
        8: (12, 12),
        12: (33, 26),
        13: (33, 30),
        14: (33, 32),
        15: (25, 8),
        16: (25, 14),
        17: (25, 16),
        18: (25, 23),
        19: (25, 24),
        20: (1, 8),
        21: (1, 12),
        22: (1, 16),
        23: (1, 23),
        24: (1, 25),
        25: (4, 1),
        26: (4, 8),
        27: (4, 12),
        28: (4, 16),
        29: (4, 19),
        30: (4, 23),
        31: (4, 25),
        32: (4, 31),
        33: (35, 8),
        34: (35, 16),
        35: (35, 23),
        36: (35, 25),
        37: (35, 26),
        38: (35, 30),
        39: (5, 1),
        40: (5, 8),
        41: (5, 12),
        42: (5, 16),
        43: (5, 23),
        44: (5, 23),
        45: (5, 25),
        46: (20, 1),
        47: (20, 8),
        48: (20, 12),
        49: (20, 16),
        50: (20, 23),
        51: (20, 31),
        52: (20, 14),
        53: (31, 8),
        54: (31, 16),
        55: (31, 3),
        56: (31, 23),
        57: (31, 28),
        58: (31, 25),
        59: (7, 18),
        60: (7, 22),
        61: (16, 8),
        62: (16, 16),
        63: (16, 23),
        64: (16, 25),
        65: (29, 5),
        66: (29, 11),
        67: (29, 8),
        68: (29, 16),
        69: (29, 23),
        70: (3, 8),
        71: (3, 16),
        72: (3, 21),
        73: (3, 23),
        74: (3, 25),
        75: (3, 26),
        76: (27, 8),
        77: (27, 16),
        78: (27, 21),
        79: (27, 23),
        80: (27, 25),
        81: (30, 16),
        82: (30, 26),
        83: (26, 23),
        84: (26, 8),
        85: (26, 9),
        86: (26, 16),
        87: (25, 13),
        88: (26, 31),
        89: (37, 1),
        90: (37, 12),
        91: (37, 30),
        92: (37, 31),
        93: (23, 8),
        94: (23, 19),
        95: (23, 30),
        96: (23, 31),
        97: (14, 29),
        98: (6, 8),
        99: (6, 16),
        100: (6, 23),
        101: (6, 25),
        102: (6, 26),
        103: (21, 6),
        104: (21, 27),
        105: (21, 27),
        106: (10, 4),
        107: (10, 8),
        108: (10, 15),
        109: (10, 16),
        110: (10, 23),
        111: (10, 30),
        112: (8, 1),
        113: (8, 12),
        114: (8, 26),
        115: (24, 8),
        116: (24, 16),
        117: (24, 23),
        118: (11, 8),
        119: (11, 16),
        120: (11, 23),
        121: (11, 30),
        122: (32, 10),
        123: (32, 18),
        124: (15, 10),
        125: (15, 18),
        126: (15, 25),
        127: (15, 26),
        128: (22, 8),
        129: (22, 5),
        130: (17, 16),
        131: (34, 9),
        132: (34, 31),
        133: (2, 0),
        134: (2, 10),
        135: (2, 18),
        136: (36, 6),
        137: (36, 8),
        138: (36, 23),
        139: (19, 30),
        140: (13, 6),
        141: (13, 7),
        142: (28, 1),
        143: (28, 12),
        144: (18, 6),
        145: (24, 32),
        146: (0, 0),
        147: (16, 2),
        148: (9, 3),
        149: (0, 9),
        150: (0, 17),
        151: (0, 18),
        152: (0, 19),
        153: (0, 20),
        154: (0, 22),
        155: (9, 28),
        156: (16, 5)}

def make_vocab():
    with open(AID2ACTION, 'r') as act_f:
        action_cls_text = act_f.read().splitlines()
        # id2action:  id: action_text
        action_cls_text = [action[5:] for action in action_cls_text]
        action2id = dict(zip(action_cls_text, range(len(action_cls_text))))

    verb2id = dict(zip(verb_classes, range(len(verb_classes))))
    obj2id = dict(zip(object_classes, range(len(object_classes))))
    scene2id = dict(zip(list(scene_convert.values()), range(len(scene_convert))))
    act2ov = c2ov
    return obj2id, verb2id, scene2id, action2id, act2ov

def parse_charades_csv(filename, s_lab2int, act2id):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['action_flow_id']
            scene = scene_convert[row['scene']]
            ob_start_f = int(row['ob_start_frame'])
            ob_end_f = int(row['ob_end_frame'])

            ob_acts = [act2id[act] for act in row['ob_act_seq'].split(';')]

            # only one gt
            future_acts = [act2id[act] for act in
                            row['future_act_seq'].split(';')]  # answer: (choice_id, act_text# )

            intention = row['intention']

            actions = [a.split(' ') for a in actions.split(';')]
            actions = [{'class': x, 'start': float(
                y), 'end': float(z)} for x, y, z in actions]
            if intention:
                labels[vid] = {'actions': actions, 'ob_start_f':ob_start_f, 'ob_end_f':ob_end_f, 'intention':intention,
                               'ob_acts': ob_acts, 'future_acts':future_acts, 'scene':s_lab2int[scene]}
            else:
                labels[vid] = {'actions': actions, 'ob_start_f':ob_start_f, 'ob_end_f':ob_end_f,
                               'ob_acts': ob_acts, 'future_acts':future_acts, 'scene':s_lab2int[scene]}

    return labels

def cls2int(x, c2ov = None):
    a = x
    if isinstance(x, str):
        a = int(x[1:])
    o, v = c2ov[a]
    return o, v, a


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path, 'RGB')
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def cache(cachefile):
    """ Creates a decorator that caches the result to cachefile """
    def cachedecorator(fn):
        def newf(*args, **kwargs):
            print('cachefile {}'.format(cachefile))
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as f:
                    print("Loading cached result from '%s'" % cachefile)
                    return pickle.load(f)
            res = fn(*args, **kwargs)
            with open(cachefile, 'wb') as f:
                print("Saving result to cache '%s'" % cachefile)
                pickle.dump(res, f)
            return res
        return newf
    return cachedecorator