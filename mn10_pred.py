import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import logging
import argparse
import torch
from rslad_loss import *
from cifar10_models import *
import torchvision
from torchvision import datasets, transforms

# we fix the random seed to 0, this method can keep the results consistent in the same conputer.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

epochs = 300
batch_size = 128
epsilon = 8/255.0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

student = mobilenet_v2()
# 先模型并行化
student = torch.nn.DataParallel(student)
# 再加载模型
student.load_state_dict(torch.load('./models/mobilenet_v2-CIFAR10_RSLAD0.5487.pth'))
student = student.cuda()
student.eval()

for epoch in range(1,epochs+1):
    if (epoch%20 == 0 and epoch <215) or (epoch%1 == 0 and epoch >= 215):
        test_accs = []
        student.eval()
        for step,(test_batch_data,test_batch_labels) in enumerate(testloader):
            test_ifgsm_data = attack_pgd(student,test_batch_data,test_batch_labels,attack_iters=20,step_size=0.003,epsilon=8.0/255.0)
            logits = student(test_ifgsm_data)
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs = test_accs + predictions.tolist()
        test_accs = np.array(test_accs)
        test_acc = np.sum(test_accs==0)/len(test_accs)
        print('robust acc',np.sum(test_accs==0)/len(test_accs))
    