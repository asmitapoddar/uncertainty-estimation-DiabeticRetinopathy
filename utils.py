import numpy as np
import tempfile
import itertools as IT
import os
from torch.nn.parallel import DataParallel
import torch

from networks.stochbn import _MyBatchNorm
from networks.resnet import resnet
from tqdm import tqdm
from PIL import Image
import importlib
import sys
import pickle
import torchvision
from torchvision import models, transforms
import PIL
import scipy.ndimage as ndi
from torch.nn import Dropout
from torch.utils.data import Dataset
from torch.autograd import Variable

def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

class ImageFolderNoTarget(Dataset):
    def __init__(self,  data_dir, transform, eval_rotation):
        self.imgs = []
        self.eval_rotation = eval_rotation
        self.transform = transform
        for subdir, dirs, files in os.walk(data_dir):
            for f in files:
                file_path = subdir + os.sep + f
                if (is_image_file(f)):
                    if self.eval_rotation:
                        self.imgs.extend([file_path]*4)
                    else:
                        self.imgs.extend([file_path])

    def __getitem__(self, index):
        path = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f)
            if self.eval_rotation:
                img = img.rotate((index%4) * 90)
            return self.transform(img.convert('RGB')),0

    def __len__(self):
        return len(self.imgs)


def uniquify(path, sep=''):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s=sep, n=next(count))
    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir=dirname, prefix=filename, suffix=ext)
        tempfile._name_sequence = orig
    return filename


def make_description(args):
    return '{}'.format(vars(args))


class Ensemble:
    """
    Ensemble for classification. Take logits and average probabilities using softmax.
    """
    def __init__(self, save_logits=False):
        self.__n_estimators = 0
        self.cum_proba = 0
        self.logits = None
        if save_logits:
            self.logits = []

    def add_estimator(self, logits):
        """
        Add estimator to current ensemble. First call define number of objects (N) and number of classes (K).
        :param logits: ndarray of logits with shape (N, K)
        """
        if self.logits is not None:
            self.logits.append(np.copy(logits))
        l = np.exp(logits - logits.max(1)[:, np.newaxis])

        assert not np.isnan(l).any(), 'NaNs while computing softmax'
        self.cum_proba += l / l.sum(1)[:, np.newaxis]
        assert not np.isnan(self.cum_proba).any(), 'NaNs while computing softmax'

        self.__n_estimators += 1

    def get_proba(self):
        """
        :return: ndarray with probabilities of shape (N, K)
        """
        return self.cum_proba / self.__n_estimators

    def get_logits(self):
        return np.array(self.logits)


class AccCounter:
    """
    Class for count accuracy during pass through data with mini-batches.
    """
    def __init__(self):
        self.__n_objects = 0
        self.__sum = 0

    def add(self, outputs, targets):
        """
        Compute and save stats needed for overall accuracy.
        :param outputs: ndarray of predicted values (logits or probabilities)
        :param targets: ndarray of labels with the same length as first dimension of _outputs_
        """
        self.__sum += np.sum(outputs.argmax(axis=1) == targets)
        self.__n_objects += outputs.shape[0]

    def acc(self):
        """
        Compute current accuracy.
        :return: float accuracy.
        """
        return self.__sum * 1. / self.__n_objects

    def flush(self):
        """
        Flush stats.
        :return:
        """
        self.__n_objects = 0
        self.__sum = 0


def softmax(logits, temp=1.):
    assert not np.isnan(logits).any(), 'NaNs in logits for softmax'
    if len(logits.shape) == 2:
        l = np.exp((logits - logits.max(1)[:, np.newaxis]) / temp)
        try:
            assert not np.isnan(l).any(), 'NaNs while computing softmax'
            return l / l.sum(1)[:, np.newaxis]
        except Exception as e:
            raise e
    elif len(logits.shape) == 4:
        return de_hbn_ensemble(logits, temp=temp)
    else:
        l = np.exp((logits - logits.max(2)[:, :, np.newaxis]) / temp)
        assert not np.isnan(l).any(), 'NaNs while computing softmax with temp={}'.format(temp)
        l /= l.sum(2)[:, :, np.newaxis]
        return np.mean(l, axis=0)


def entropy_plot_xy(p):
    e = entropy(p)
    n = len(e)
    return sorted(e), np.arange(1, n + 1) / 1. / n


def entropy_plot_with_proba(p):
    e = entropy(p)
    n = len(e)
    return sorted(e), np.arange(1, n + 1) / 1. / n


def entropy_plot_with_logits(logits, adjust_t=False, k=0.2,
                             labels=None):
    if len(logits.shape) == 2:
        logits = logits[np.newaxis]

    temp = 1.
    if adjust_t:
        k = int(logits.shape[1] * k)
        val_logits = logits[:, :k]
        logits = logits[:, k:]
        temp = adjust_temp(val_logits, labels[:k])
    return entropy_plot_with_proba(softmax(logits, temp=temp))


def set_strategy(net, strategy):
    for m in net.modules():
        if isinstance(m, _MyBatchNorm):
            m.strategy = strategy


def set_do_to_train(net):
    have_do = False
    for m in net.modules():
        if isinstance(m, Dropout):
            m.train()
            have_do = True
    return have_do


def get_model(model='ResNet50', **kwargs):
    if model == 'ResNet50':
        return resnet(depth=50,num_classes=kwargs.get('num_output',2))
    else:
        raise NotImplementedError('unknown {} model'.format(model))


def get_dataloader(data_path,norm_mean,norm_std,
        eval_no_crop=True,
        eval_rot=False,
        batch_size=32):
    if eval_no_crop:
        transform_steps  =[ transforms.Resize(256), transforms.CenterCrop(224)]
    else: 
        transform_steps = [ transforms.Resize(224) ] 

    test_transform = transforms.Compose( 
            transform_steps + 
            [ transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std) ])

    '''
    test_transform_image = transforms.Compose(
        transform_steps +
        [transforms.ToTensor(),
         transforms.Normalize(norm_mean, norm_std)], transforms.ToPILImage())
         print('test_transform_image', type(test_transform_image))
    '''


    testsets = ImageFolderNoTarget(data_path, test_transform, eval_rot)
    testloader = torch.utils.data.DataLoader(
        testsets,
        batch_size = batch_size,
        shuffle = False,
        num_workers=8
    )
    return testloader


# def load_model(filename, num_output=2, print_info=False):
#     use_cuda = torch.cuda.is_available()
#     checkpoint = torch.load(filename)
#     net = get_model(num_output=num_output)
#     net = DataParallel(net, device_ids=range(torch.cuda.device_count()))
#     net.load_state_dict(checkpoint['model'].state_dict())
#     return net


def load_model(model_file, cuda):
    '''
    model_ft = resnet(False, 50)
    model_ft.fc = torch.nn.Linear(model_ft.fc.in_features, 4)
    #checkpoint = torch.load(model_file, map_location='cpu')
    checkpoint = torch.load(model_file)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] # remove module.
        print(name)
        new_state_dict[name] = v
    print(new_state_dict)
    # checkpoint pre_trained_model=torch.load("Path to the .pth file")
    #new=list(checkpoint['model'].state_dict()[0])
    #print("HIIII", checkpoint['model'].state_dict())
    #my_model_kvpair=model_ft.state_dict()
    #print(model_ft.state_dict().items())
    #count=0
    #for key,value in model_ft.state_dict().items():
    #    layer_name, weights = new[count]      
    #    my_model_kvpair[key]=weights
    #    count+=1


    #for (name, layer) in checkpoint.items():
        #iteration over outer layers
    #    print((name, layer))
    print(model_ft.state_dict())
    model_ft.load_state_dict(new_state_dict)
    # set_strategy(model_ft, 'running')
    set_strategy(model_ft, 'sample')
    # model_ft.bn1.strategy = 'sample'
    # for m in model_ft.layer1.modules():
    #     if isinstance(m, MyBatchNorm2d):
    #         m.strategy = 'sample'
    if cuda:
        model_ft = model_ft.cuda()
    #return model_ft
    return my_model_kvpair
    '''

    model_ft = models.resnet18()
    model_ft.fc = torch.nn.Linear(model_ft.fc.in_features, 5)

    model_ft = model_ft.cuda()
    #model_ft = torch.nn.DataParallel(
    #    model_ft, device_ids=range(torch.cuda.device_count()))

    checkpoint = torch.load(model_file)
    model_ft.load_state_dict(checkpoint.state_dict())
    
    #model_ft.load_state_dict(checkpoint['model'].state_dict())

    return model_ft

def pad(img, size, mode):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    return np.pad(img, [(size, size), (size, size), (0, 0)], mode)


class MyPad(object):
    def __init__(self, size, mode='reflect'):
        self.mode = mode
        self.size = size
        self.topil = transforms.ToPILImage()

    def __call__(self, img):
        return self.topil(pad(img, self.size, self.mode))


def to_np(x):
    return x.data.cpu().numpy()


def entropy(p):
    eps = 1e-8
    assert np.all(p >= 0)
    return np.apply_along_axis(lambda x: -np.sum(x[x > eps] * np.log(x[x > eps])), 1, p)


def ensemble(net, data, bs, n_infer=50, return_logits=False):
    """ Ensemble for net training with Vanilla BN """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    ens = Ensemble(save_logits=return_logits)
    acc_data = np.array(list(map(lambda x: transform_test(x).numpy(), data)))
    logits = []
    for _ in range(n_infer):
        logits = np.zeros([acc_data.shape[0], 5])
        perm = np.random.permutation(np.arange(acc_data.shape[0]))

        for i in range(0, len(perm), bs):
            idxs = perm[i: i + bs]
            inputs = Variable(torch.Tensor(acc_data[idxs]).cuda())
            outputs = net(inputs)
            assert np.allclose(logits[idxs], 0.)
            logits[idxs] = outputs.cpu().data.numpy()

        ens.add_estimator(logits)
    return ens.get_proba(), ens.get_logits()


def predict_proba(dataloader, net, ensembles=1, n_classes=10, return_logits=False, cuda=True):
    proba = np.zeros((len(dataloader.dataset), n_classes))
    labels = []
    logits = []
    p = 0
    for img, label in tqdm(dataloader):
        ens = Ensemble(save_logits=return_logits)
        if cuda:
            img = img.cuda()
        #for _ in range(ensembles):
        #    pred = net(img).data.cpu().numpy()
        #    ens.add_estimator(pred)
        pred = net(img).data.cpu().numpy()

        ens.add_estimator(pred)
        proba[p: p + pred.shape[0]] = ens.get_proba()
        #print('pred.shape[0]',pred.shape[0])    print('INTERMEDIATE proba', proba)
        p += pred.shape[0]
        labels += label.tolist()
        if return_logits:
            logits.append(ens.get_logits())

    if return_logits:
        logits = np.stack(logits)
        logits = logits.transpose(0, 2, 1, 3)
        logits = np.concatenate(logits, axis=0)
        logits = logits.transpose(1, 0, 2)
        return proba, np.array(labels), logits, np.array(dataloader.dataset.imgs)
    return proba, np.array(labels), np.array(dataloader.dataset.imgs)
