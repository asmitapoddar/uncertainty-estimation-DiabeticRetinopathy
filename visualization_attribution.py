from __future__ import print_function
import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from grad_cam import (BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation, IntegratedGradient)
from networks.resnet import resnet
# if model has LSTM
# torch.backends.cudnn.enabled = False
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

def normalize(attrs, ptile=99):
    '''Normalize the provided attributions so that they fall betweenb-1.0 and 1.0.
    '''
    h = np.percentile(attrs, ptile)
    l = np.percentile(attrs, 100-ptile)
    return np.clip(attrs/max(abs(h), abs(l)), -1.0, 1.0)

def normalize_one_side(attrs, ptile=99):
    '''Normalize the provided attributions so that they fall between -1.0 and 1.0.
    '''
    h = np.percentile(attrs, ptile)
    return np.clip(attrs/h, 0.0, 1.0)

def preprocess_img(image_path):
    raw_image = cv2.imread(image_path)[..., ::-1]
    raw_image = cv2.resize(raw_image, (224, ) * 2)
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.6000, 0.3946, 0.6041],
            std = [0.2124, 0.2335, 0.2360]
        )
    ])(raw_image).unsqueeze(0)
    return image

def gray_scale(img):
    '''Converts the provided RGB image to gray scale.
    '''
    img = np.average(img, axis=2)
    return np.transpose([img, img, img], axes=[1,2,0])

def pil_img(a):
    '''Returns a PIL image created from the provided RGB array.
    '''
    a = np.uint8(a)
    return Image.fromarray(a)

####CHANGES TO MODEL LOADING HERE
def load_resnet_stochbn(model_file):
    model_ft = models.resnet18()
    model_ft.fc = torch.nn.Linear(model_ft.fc.in_features, 5)

    model_ft = model_ft.cuda()
    #model_ft = torch.nn.DataParallel(
    #    model_ft, device_ids=range(torch.cuda.device_count()))

    checkpoint = torch.load(model_file)
    #model_ft.load_state_dict(checkpoint['model'].state_dict())
    model_ft.load_state_dict(checkpoint.state_dict())
    return model_ft

def load_resnet(model_file):
    # discard last layer
    model_ft = models.resnet18()
    model_ft.fc = torch.nn.Linear(model_ft.fc.in_features, 5)

    model_ft = model_ft.cuda()
    #model_ft = torch.nn.DataParallel(
    #    model_ft, device_ids=range(torch.cuda.device_count()))

    checkpoint = torch.load(model_file)
    #model_ft.load_state_dict(checkpoint['model'].state_dict())
    model_ft.load_state_dict(checkpoint.state_dict())
    return model_ft

def visdom_img(img, title):
    vis.image(np.asarray(img, dtype=np.uint8).transpose((2, 0, 1)), opts={'title': title})

def create_vis():
    image_folder = os.path.join(os.getcwd(), 'Result')
    ###MODEL FILE CHANGE HERE
    model_file = 'models/resnet-18_trained.t7'
    #model_file = 'networks/resnet-18_trained.t7'
    stoch_bn = True
    name = ''
    num_tries = 1

    device = torch.device('cuda')
    current_device = torch.cuda.current_device()
    print('Running on the GPU:', torch.cuda.get_device_name(current_device))

    #CHANGES TO NN HERE
    if stoch_bn:
        if model_file == '':
            model_file = '/models/resnet-18_trained.t7'
        #model = load_resnet_stochbn(model_file).module
        model = load_resnet_stochbn(model_file)
    else:
        if model_file == '':
            model_file = '/models/resnet-18_trained.t7'
        model = load_resnet(model_file)

    '''
    model_file1 = 'models/resnet-50.t7'
    model1 = load_resnet_stochbn(model_file1).module
    model_file2 = 'models/resnet-50_best.t7'
    model2 = load_resnet(model_file2).module
    model = load_resnet_stochbn(model_file2).module
    '''

    model.to(device)
    model.eval()
    gcam = GradCAM(model=model)
    gbp = GuidedBackPropagation(model=model)
    intgrad = IntegratedGradient(model=model, steps=10)

    def integrated_gradient(image_path, num_tries):
        image = preprocess_img(image_path)

        features = []
        probs_arr = []
        for i in range(num_tries):
            probs = intgrad.forward(image.to(device))
            probs = probs.cpu().detach().numpy()
            label = np.around(probs)
            feature = intgrad.generate(probs * label)
            features.append(feature)
            probs_arr.append(probs)

        feature = np.stack(features, axis=0).mean(axis=0) + 14
        feature_std = np.stack(features, axis=0).std(axis=0)
        probs = np.stack(probs_arr, axis=0).mean(axis=0)

        return features
        # return feature_std

    def guided_gradcam(image_path, target_layer='layer4.1'):   #layer4.2
        image = preprocess_img(image_path)
        regions = []
        features = []
        probs_arr = []
        for i in range(num_tries):
            probs = gcam.forward(image.to(device))
            target = np.around(probs.cpu().detach().numpy())
            gcam.backward(idx=target)
            region = gcam.generate(target_layer=target_layer)
            probs = gbp.forward(image.to(device))
            target = np.around(probs.cpu().detach().numpy())
            gbp.backward(idx=target)
            feature = gbp.generate()
            regions.append(region)
            features.append(feature)
            probs_arr.append(probs.cpu().detach().numpy())

        region = np.stack(regions, axis=0).mean(axis=0)
        feature = np.stack(features, axis=0).mean(axis=0)
        feature_std = np.stack(features, axis=0).std(axis=0)
        probs = np.stack(probs_arr, axis=0).mean(axis=0)

        h, w, _ = feature.shape
        region = cv2.resize(region, (w, h))[..., np.newaxis]
        output = feature * region
        #print('output.shape', output.shape, '[{:.5f} {:.5f} {:.5f} {:.5f}]'.format(probs[0], probs[1], probs[2], probs[3]))
        #print('region.shape', region.shape)
        #return feature, region
        np.save('GC.png', output)
        np.save('GC_features.png', output)
        return(features)

    R = np.array([255, 0, 0])
    G = np.array([0, 255, 0])
    B = np.array([0, 0, 255])

    for dirpath, dirnames, filenames in os.walk(image_folder):
        for file in filenames:
            if file.endswith('.png') or file.endswith('.jpg'):
                parts = dirpath.split('/')
                # cls = int(parts[-1])
                filepath = os.path.join(dirpath, file)

                # feature, region = guided_gradcam(filepath, num_tries=1)
                print(filepath)
                feature_arr_ig = integrated_gradient(filepath, num_tries)
                feature_arr_gc = guided_gradcam(filepath)

                # out, mask = visualize_attrs(Image.open(filepath).resize((224,224)), feature)
                # visdom_img(out, name+' '+file)
                output_dir = Path(dirpath.replace('data', 'pred'))
                output_dir.mkdir(parents=True, exist_ok=True)
                # print(np.asarray((feature_arr)[0].shape))
                # out.save(os.path.join(output_dir.as_posix(), file))

                np.save(os.path.join(output_dir.as_posix(), 'Integrated_Gradient.png'), feature_arr_ig)
                np.save(os.path.join(output_dir.as_posix(), 'GradCAM.png'), feature_arr_gc)

                # scipy.misc.imsave(os.path.join(output_dir.as_posix(), file), feature_arr)
                print(os.path.join(output_dir.as_posix(), file))
                # output = np.delete(f, np.s_[:], axis = 0)
                # im = Image.fromarray(np.asarray(feature_arr)[0])
                # im.save(os.path.join(output_dir.as_posix(), file))
                # f = np.asarray(feature_arr)
                # scipy.misc.imsave(os.path.join(output_dir.as_posix(), file), f[0])
    print('Visualization created')

    return(file)
