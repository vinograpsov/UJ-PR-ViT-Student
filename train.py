import torch
from PIL import Image
from torchvision import transforms
from torchsummary import summary
import numpy as np
import timm
import os
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
from torchvision import datasets
import matplotlib.pyplot as plt

from vit_rollout import VITAttentionRollout
from torch.utils.data import DataLoader, Subset
import resnet
import imagenet
import newmodel
from utils import create_teacher, create_student

import os
if not os.path.exists('deitbase_max'):
    os.makedirs('deitbase_max')



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    print("hm", device)

    model_teacher = create_teacher()
    model_teacher.to(device)

    model_student = create_student()
    model_student = model_student.to(device)

    model = newmodel.NewModel(model_teacher, model_student)
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # on your device
    # data_folder = './data/ILSVRC2012_img_val'
    # imagenet_data = imagenet.ImageNet(data_folder, transform)

    # # on server
    data_folder = '/shared/sets/datasets/vision/ImageNet'

    imagenet_data = datasets.ImageNet(data_folder, split='train', transform=transform)
    indices = torch.randperm(len(imagenet_data))
    num_samples_to_use = int(len(imagenet_data) * 0.1) ## 10%
    imagenet_data = Subset(imagenet_data, indices[:num_samples_to_use])

    train_dataloader = DataLoader(imagenet_data, batch_size=1, shuffle=True, generator=torch.Generator(device=device))

    LR = 0.001
    optimizer = torch.optim.Adam(model_student.parameters(), lr=LR)
    print("LR : " , LR)
    criterion = torch.nn.L1Loss().to(device)

    losses= []
    steps = []

    # total_samples = 0
    # total_correct = 0
    # with torch.no_grad():
    #     for i, (image, target) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    #         with torch.cuda.amp.autocast():
    #             output = model_teacher(image)
    #         print(output.shape)
    #         _, predicted = torch.max(output, 1)
    #         total_samples += target.size(0)
    #         total_correct += (predicted == target).sum().item()
    #
    #         if (i == 50):
    #             print(f'accuracy after {i} : {total_correct/total_samples}')


    for epoch in range(10):
        print("EPOCH: ", epoch+1)
        for i, (image, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        # for i, image in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            image = image.to(device)
            optimizer.zero_grad()
            output, target = model(image)
            output = output.reshape(14,14)
            loss = criterion(target, output)
            loss.backward()
            optimizer.step()
            if (i+1) % 500 == 0:
                losses.append(loss.item())
                steps.append(epoch * len(train_dataloader) + i+1)
            if (i+1) % 30000 == 0 or ((i+1) % 5000 ==0 and epoch == 0 ):
                print(f"STEP: {i+1}, loss: {loss.item()}")
                fig, axes = plt.subplots(1, 3, figsize=(10, 5))
                axes[0].imshow(image[0].permute(1, 2, 0).cpu().detach())
                axes[1].imshow(target.cpu())
                axes[2].imshow(output.cpu().detach())
                plt.savefig(f"deitbase_max/train{epoch}_{i+1}.png")
                plt.close(fig)

            # if i == 10 or i == 40:
            #     fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            #     axes[0].imshow(image[0].permute(1, 2, 0).cpu().detach())
            #     axes[1].imshow(target.cpu())
            #     axes[2].imshow(output.cpu().detach())
            #     plt.savefig(f"train{epoch}_{i + 1}.png")
            #     plt.close(fig)

        plt.plot(steps, losses)
        plt.title(f'Loss over time (after {epoch+1} epoch)')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig(f'deitbase_max/Loss{epoch+1}.png')
        plt.close()
        torch.save(model_student.state_dict(), 'deitbase_max/model_state.pth')


        if epoch == 5:
            plt.plot(steps, losses)
            plt.title(f'Loss over time (after {epoch + 1} epoch)')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.savefig(f'deitbase_max/Loss{epoch + 1}.png')
            plt.close()
            torch.save(model_student.state_dict(), 'deitbase_max/model_state_after5epoch.pth')


    plt.plot(steps, losses)
    plt.title('deitbase_max/Loss over time')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig("deitbase_max/Loss.png")
    plt.close()


