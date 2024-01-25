import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from cifar10_models import *
from autoattack import AutoAttack
from torch.autograd import Variable

from advertorch.attacks import CarliniWagnerL2Attack
import numpy as np
import torchattacks

def test_autoattack(model, testloader, norm='Linf', eps=8/255, version='standard', verbose=True):
    
    adversary = AutoAttack(model, norm=norm, eps=eps, version=version, verbose=verbose)

    if version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
        adversary.apgd.n_restarts = 1
        adversary.apgd_targeted.n_restarts = 1

    x_test = [x for (x,y) in testloader]
    x_test = torch.cat(x_test, 0)
    y_test = [y for (x,y) in testloader]
    y_test = torch.cat(y_test, 0)


    with torch.no_grad():
        x_adv, y_adv = adversary.run_standard_evaluation(x_test, y_test, bs=testloader.batch_size, return_labels=True)

    adv_correct = torch.sum(y_adv==y_test).data
    total = y_test.shape[0]

    rob_acc = adv_correct / total
    print('Attack Strength:%.4f \t  AutoAttack Acc:%.3f (%d/%d)'%(eps, rob_acc, adv_correct, total))


def cw2_eval(model, test_loader, c,num_classes,req_count):
    # Compute the probability of the label class versus the maximum other
    adversary = CarliniWagnerL2Attack(
                        model, confidence=0.01, max_iterations=1000, clip_min=0., clip_max=1., learning_rate=0.01,
                        targeted=False, num_classes=num_classes, binary_search_steps=1, initial_const=c)

    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if batch_idx < int(req_count/test_loader.batch_size):
            inputs, targets = inputs.cuda(), targets.cuda()
            total += targets.size(0)
            advs = adversary.perturb(inputs, targets)
            adv_outputs = adversary.predict(advs)
            adv_preds = adv_outputs.max(dim=1, keepdim=False)[1]
            adv_correct += adv_preds.eq(targets.data).cpu().sum()
            rob_acc = 100. * float(adv_correct) / total

    return rob_acc

def eval_clean(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_robust(model,test_loader,attack):
    model.eval()
    test_loss = 0
    correct = 0
   
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv = attack(data,target)
            output = model(x_adv)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy