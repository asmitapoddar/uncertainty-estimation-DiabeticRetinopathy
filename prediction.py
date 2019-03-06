from __future__ import print_function, division

import torch
import numpy as np
import os
import utils
import cv2

def model_prediction():

    model_path = 'networks/resnet-18_trained.t7'
    test_dir = 'Result/'
    cuda = True
    log_file = 'evalss_data'
    eval_rot = True
    eval_no_crop = True
    n_tries = 10
    seed = 42
    output_dir = 'Result'
    running = True

    print("loading model", model_path)

    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_file = utils.uniquify(os.path.join(output_dir, log_file), sep='-')

    eval_data = test_dir
    net = utils.load_model(model_path, cuda)

    image_array = cv2.imread(os.path.join(test_dir, 'data', 'image.png'))
    mean_of_image = np.mean(image_array, axis=(0, 1))/1000
    std_of_image = np.std(image_array, axis=(0, 1))/100

    ###  Load the data into PyTorch using dataloader
    dataloader = utils.get_dataloader(test_dir,
            [0.6000, 0.3946, 0.6041],
            [0.2124, 0.2335, 0.2360],
            eval_no_crop, eval_rot, batch_size=1)

    #dataloader = utils.get_dataloader(test_dir, mean_of_image, std_of_image, eval_no_crop, eval_rot, batch_size=1)
    print(type(dataloader))

    '''
    # A function to get probabililties using only one iteration
    net = net.eval()
    for img, label in dataloader:
        print(type(img))
        print(type(label))

        img = img.cuda()
        print('IMAGE TYPE:',img)
        pred = net(img).data.cpu().numpy()
    print('checking', pred)
    probs = nn.functional.softmax(pred)
    print('PROBABILITIES:', probs)
    '''

    if not running:
        net.eval()
        utils.set_strategy(net, 'sample')
        have_do = utils.set_do_to_train(net)

        res = utils.predict_proba(dataloader, net, n_classes=5, return_logits=True, ensembles=n_tries, cuda=cuda)
        print('Result', res)
        '''
        eval_data['test'] = {
            'ensemble/proba': res[0],
            'ensemble/logits': res[2],
            'eval/labels': res[1],
            'ensemble/filenames': res[3]
        }    '''
    else:
        net.eval()
        utils.set_strategy(net, 'running')
        have_do = utils.set_do_to_train(net)

        res = utils.predict_proba(dataloader, net, n_classes=5, return_logits=True, ensembles=n_tries if have_do else 3)
        print('type(eval_data):', type(eval_data))
        '''
        eval_data['test'].update({
            'eval/proba': res[0],
            'eval/logits': res[2],
            'eval/labels': res[1],
            'ensemble/filenames': res[3]
        }) '''

    # Get the mean of predictions for n_tries iterations for each class
    prob_means_en = np.mean(res[0], axis=0)

    output_file_name = 'res_norotate'
    torch.save(res, output_dir + '/' + output_file_name)
    #print(res[2].shape)
    print('Created output file \'', output_file_name, ' \' ')

    torch.cuda.empty_cache()
    return (prob_means_en)
