import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
import torch
from resnetteacher import *
from cifar10_models import *
import torch.nn as nn
import torchvision
from torchvision import transforms
import Cifar10attack_generator as attack
import torchattacks
parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--net', type=str, default="mobilenet_v2", help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10,help='WRN width factor')
parser.add_argument('--drop_rate', type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--model_path', default="/data4/zhuqingying/exper/MobileTest12/Trade-off0.72445natural acc0.8169robust acc0.632.pth", help='model for white-box attack evaluation')
parser.add_argument('--coeff', default=0.1, type=float) 
parser.add_argument('--num_classes', default=10, type=int) 
args = parser.parse_args()

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
if args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

print('==> Load Model')
if args.net == "resnet50":
    model = resnet50() #目前num_classes = 10 cifar10  在resnet18中已经定义
    net = "resnet50"
if args.net == "resnet56":
    model = resnet56() #目前num_classes = 10 cifar10  在resnet18中已经定义
    net = "resnet56"
if args.net == "mobilenet_v2":
    model = mobilenet_v2()
    net = "mobilenet_v2"
if args.net == "resnet18":
    model = resnet18() #目前num_classes = 10 cifar10  在resnet18中已经定义
    net = "resnet18"
if args.net == "WRN":
    model = wideresnet().cuda()
    net = "WRN{}-{}-dropout{}".format(args.depth,args.width_factor,args.drop_rate)



print(net)
print(args.model_path)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(args.model_path)

model.load_state_dict(checkpoint)

loss, test_nat_acc = attack.eval_clean(model, test_loader)
print('Natural Test Accuracy: {:.2f}%'.format(100. * test_nat_acc))

# cw_wori_acc = attack.cw2_eval(model,test_loader, c=args.coeff, num_classes=args.num_classes,req_count=10000)
# print('CW2 Test Accuracy: {:.2f}%'.format(cw_wori_acc))
atk1 = torchattacks.FGSM(model, eps=8/255)
loss, fgsm_wori_acc = attack.eval_robust(model,test_loader,atk1)
print('FGSM without Random Start Test Accuracy: {:.2f}%'.format(100. * fgsm_wori_acc))
atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20, random_start=True)
loss, pgd20_acc = attack.eval_robust(model, test_loader,atk2)
print('PGD20 Madry Test Accuracy: {:.2f}%'.format(100. * pgd20_acc))
atk3 = torchattacks.PGD(model, eps=0.03, alpha=0.007, steps=20, random_start=True)
loss, pgd20_acc = attack.eval_robust(model, test_loader, atk3)
print('PGD20 trades Test Accuracy: {:.2f}%'.format(100. * pgd20_acc))
atk4 = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=args.num_classes, seed=None, verbose=False)
loss, AA_acc = attack.eval_robust(model, test_loader,atk4)
print('AA Test Accuracy: {:.2f}%'.format(100. * AA_acc))
# # print('Evaluate AA:')
# attack.test_autoattack(model, test_loader, norm='Linf', eps=8/255,
#                 version='standard', verbose=False)