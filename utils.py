import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pprint
import argparse
import random
import numpy as np
_utils_pp = pprint.PrettyPrinter()

def pprint(x):
    _utils_pp.pprint(x)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def fs_finetune(args, model, loader, ft_iter=100, ft_lr=0.01, num_classes=10, log=True, add_module=False, add_num=0):
    classifier = model.finetune_classifier(num_classes=num_classes)
    #classifier = nn.Linear(model.out_attr_dim, num_classes).cuda()
    if args['model'] == 'SMN':
        if add_module and add_num > 0:
            model.routing_module.add_module = True
            model.routing_module.add_modules(add_num)
        params = model.get_finetune_params(classifier, add_modules=add_module)
    else:
        params = classifier.parameters()
    optimizer_f = torch.optim.Adam(params, lr=ft_lr)

    finetune_iter = ft_iter
    model.train()

    def test():
        # then test the finetuned model on remain data
        accuracy, num = 0, 0
        model.eval()
        with torch.no_grad():
            for i in range(0, loader.test_len()):
                test_x, test_y = loader.test_get(i)
                test_x = model.to_device(test_x)
                test_y = model.to_device(test_y).long()

                feature = model.forward_feature(test_x)
                probs = model.forward_classifier(feature, classifier=classifier, add_module=add_module)
                preds = torch.argmax(probs, dim=1)

                correct = preds == test_y
                accuracy += correct.sum().item()
                num += test_x.shape[0]
        accuracy /= num
        return accuracy * 100

    for iter in range(finetune_iter):
        tr_loss, correct, num = 0, 0, 0
        for i in range(loader.ft_len()):
            finetune_x, finetune_y = loader.ft_get(i)

            finetune_x = model.to_device(finetune_x)
            finetune_y = model.to_device(finetune_y).long()
            feature = model.forward_feature(finetune_x)
            output, loss = model.forward_classifier(feature.detach(), y=finetune_y, classifier=classifier, add_module=add_module)
            # output, loss = model.forward_classifier(feature, y=finetune_y)

            optimizer_f.zero_grad()
            loss.backward()
            optimizer_f.step()

            preds = torch.argmax(output, dim=1)
            correct += (preds == finetune_y.long()).sum()
            tr_loss += loss.item()
            num += finetune_x.shape[0]

    accuracy = test()
    # print("final acc:", accuracy)

    if args['model'] == 'SMN':
        model.routing_module.add_module = False
    return accuracy

def test_model_iid(model, loader):
    accuracy, num = 0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, loader.test_iid_len()):
            test_x, test_y = loader.test_iid_get(i)
            test_x = model.to_device(test_x)
            test_y = model.to_device(test_y).long()

            probs = model(test_x)

            preds = torch.argmax(probs, dim=1)
            correct = preds == test_y
            accuracy += correct.sum().item()
            num += test_x.shape[0]

    accuracy = accuracy / num * 100
    return accuracy

def test_model_ood(model, loader):
    accuracy, num = 0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, loader.test_len()):
            test_x, test_y = loader.test_get(i)
            test_x = model.to_device(test_x)
            test_y = model.to_device(test_y).long()

            feature = model.forward_feature(test_x)
            probs = model.forward_classifier(feature)
            preds = torch.argmax(probs, dim=1)

            correct = preds == test_y
            accuracy += correct.sum().item()
            num += test_x.shape[0]
    accuracy = accuracy / num * 100
    return accuracy