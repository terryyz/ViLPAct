"""
task1&2：视频取ground truth sequence的开始时间，到最晚结束的action截止时间 , 每个instance为初态的64帧或所有帧
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from glob import glob
from dataloader.utils import *


class ViLPAct(data.Dataset):
    def __init__(self, rgb_root, task_label_path, task, cachedir, rgb_transform=None, target_transform=None,
                 init_sample = False):
        """
        :param rgb_root: data dir of videos(rgb_frames)
        :param task_path: task data
        :param task: planning
        :param cachedir: cache save dir
        :param rgb_transform:
        :param target_transform:
        :param init_sample: if True, sample 64 frames, or keep all frames(batch size should be 1)
        """
        self.s_classes = 16
        self.o_classes = 38
        self.v_classes = 33
        self.frame_num = 64
        self.task = task
        self.init_sample = init_sample
        self.rgb_transform = rgb_transform
        self.target_transform = target_transform
        self.rgb_root = rgb_root
        self.cachedir = cachedir
        # action id : obj id, verb id
        self.obj2id, self.verb2id, self.s_lab2int, self.act2id, self.c2ov = make_vocab()
        self.id2act = {index: action for action, index in self.act2id.items()}

        self.task_labels = parse_charades_csv(task_label_path, self.s_lab2int, self.act2id)

        cachename = '{}/{}_{}_{}.pkl'.format(cachedir,
                                             self.__class__.__name__,  'act' , task)
        self.data = cache(cachename)(self.prepare)()


    def prepare(self):
        """
        act_planning
            Given：observed video, intent
        :return:
        """
        rgb_datadir = self.rgb_root
        too_short = 0
        no_predict = 0
        FPS = 24
        predict_instances = []
        task_labels = self.task_labels

        for i, (vid, label) in enumerate(task_labels.items()):
            print(i, label)
            # observed video clip from the total video
            ob_start_f = label['ob_start_f']
            ob_end_f = label['ob_end_f']

            # labels for action recognition on observed video
            o_observed_gt = torch.IntTensor(self.o_classes).zero_()  # 可观察的chunk的o_label 的ground_truth
            v_observed_gt = torch.IntTensor(self.v_classes).zero_()
            s_observed_gt = torch.IntTensor(1).zero_()
            act_observed_gt = torch.IntTensor(len(self.c2ov)).zero_()  # 可观察chunk的act_label

            # action seq label
            ob_acts = label['ob_acts']
            future_acts = label['future_acts']

            # ob_rgb_frames_lst
            rgb_iddir = rgb_datadir + '/' + vid
            rgb_lines = glob(rgb_iddir + '/*.jpg')
            rgb_n = len(rgb_lines)
            n = int(rgb_n)   #frames num of full video
            nf = ob_end_f - ob_start_f + 1   #frames num of observed video clip
            if self.init_sample:
                if n == 0 or ob_end_f == 0 or nf < self.frame_num:
                    print('filterd-short:{}，{}，{}'.format(vid, ob_start_f, ob_end_f))
                    too_short += 1
                    continue
                if nf > self.frame_num:
                    ob_start_f = ob_end_f - self.frame_num + 1
            else:
                if n == 0 or ob_end_f == 0 or nf == 0 or nf < 10:
                    print('filterd-short:{}，{}，{}'.format(vid, ob_start_f, ob_end_f))
                    too_short += 1
                    continue

            # start frame of observed video clip
            rgb_impath = '{}/{}-{:06d}.jpg'.format(
                rgb_iddir, vid, ob_start_f)
            frame_level_acts = np.zeros((nf, len(self.c2ov)), np.float32)
            for act in label['actions']:
                o, v, a = cls2int(act['class'], self.c2ov)
                o_observed_gt[o] = 1
                v_observed_gt[v] = 1
                act_observed_gt[a] = 1
                ## s is single label
                s_observed_gt = label['scene']
                for fr in range(0, nf, 1):
                    if (fr + ob_start_f) / FPS >= act['start'] and (fr + ob_start_f) / FPS < act['end']:
                        frame_level_acts[fr, int(act['class'][1:])] = 1  # binary classification


            # add the video id of each chunk
            id = vid
            # add the start image of each chunk
            time = ob_start_f

            if self.task == 'planning':
                predict_instances.append({
                    'observed_f': ob_end_f,  # observed end frame
                    'rgb_image_path': rgb_impath,  #  the first rgb_frame path
                    'o_observed_gt': o_observed_gt,  #  o_multilabel
                    'v_observed_gt': v_observed_gt,  #  v_multilabel
                    'act_observed_gt': act_observed_gt,  #  a_multilabel
                    's_observed_gt': s_observed_gt,  # int scene label
                    'frame_acts': frame_level_acts,
                    'id': id,  # vid
                    'time': time, #start frame,
                    'ob_gt':ob_acts,
                    'future_gt': future_acts,
                    'full_gt': np.array(ob_acts + future_acts)})  # predict acts list

            print("step:{}\t video:{} \t start_f:{}\t end_f:{} \t rgb_num:{}  \t ob_acts:{}, predict_acts:{}".format(i, rgb_iddir,ob_start_f,
                                                                                          ob_end_f, ob_end_f - ob_start_f + 1, ob_acts, future_acts))
            assert os.path.exists(rgb_impath)
            assert os.path.exists('{}/{}-{:06d}.jpg'.format(
                rgb_iddir, vid, ob_end_f))

        print('data num:{}, no predict:{}, too short:{}'.format(len(predict_instances),  no_predict,  too_short))
        return predict_instances

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
             ob_imgs_lst (list(path)): a list of  observed start imgs
             meta_lst (list(dict)): a list of {'id':video id, 'time': ob start frame}
             ob_olab_lst (list(int)): a list of ob obj target
             ob_slab_lst (list(scene)): a list of ob scene target
             ob_vlab_lst (list(verb)): a list of ob verb target
        """

        start_path = self.data[index]['rgb_image_path']  # start_path
        vid = self.data[index]['id']  # vid
        start_time = self.data[index]['time']  #start_frame
        observe_o = self.data[index]['o_observed_gt']  # [o_class，]
        observe_v = self.data[index]['v_observed_gt']  # [v_class,]
        observe_s = self.data[index]['s_observed_gt']  # [s]
        observe_a = self.data[index]['act_observed_gt']  # [a_class,]
        frame_acts = self.data[index]['frame_acts'] #chunk_num * [a_class]
        end_time = self.data[index]['observed_f'] #end_frame
        ob_gt = self.data[index]['ob_gt']
        future_gt = self.data[index]['future_gt']
        full_gt = self.data[index]['full_gt']


        rgb_base = start_path[:-5 - 5]
        rgb_framenr = int(start_path[-5 - 5:-4])
        rgb_chunk_imgs = []

        if self.init_sample:
            rgb_STACK = self.frame_num
        else:
            rgb_STACK = end_time - start_time + 1
        for j in range(rgb_STACK):
            _img = '{}{:06d}.jpg'.format(rgb_base, rgb_framenr + j)
            img = default_loader(_img)
            rgb_chunk_imgs.append(img)
        if self.rgb_transform is not None:
            _rgb_chunk_imgs = []
            for _per_img in rgb_chunk_imgs:
                tmp = self.rgb_transform(_per_img)
                _rgb_chunk_imgs.append(tmp)
            rgb_chunk_imgs = torch.stack(_rgb_chunk_imgs, dim=1)
            # init_sample == True: now the img is 3x64x224x224
            # init_sample == False: img is 3×T×224×224， assert batch_size == 1

        meta = {}
        meta['id'] = vid  # video id for segment
        meta['time'] = start_time  # start frame for segment
        return ob_gt, future_gt, full_gt, rgb_chunk_imgs, meta, observe_o, observe_s, observe_v, observe_a, frame_acts

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    RGB_Root Location: {}\n'.format(self.rgb_root)
        tmp = '    RGB_Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.rgb_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get(args):
    """ Entry point. Call this function to get all Charades dataloaders """
    #rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    
    rgb_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])


    if args.init_sample == False:
        assert args.batch_size == 1

    
    test_dataset = ViLPAct(
        args.rgb_data, args.test_file, args.task, args.cache_dir,
        rgb_transform=transforms.Compose([
            transforms.Resize(int(256./224*args.inputsize)),
            transforms.CenterCrop(args.inputsize),
            transforms.ToTensor(),
            rgb_normalize,
        ]),
        init_sample = args.init_sample)

    return test_dataset






