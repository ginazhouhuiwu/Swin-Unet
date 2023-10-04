import argparse, logging, os, random, sys, time

from collections import defaultdict

from datasets.dataset_dlmd import crop_pad, split

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms

import wandb

from tqdm import tqdm


gpu = 2
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
torch.cuda.set_device(gpu)
device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")

def set_ref_images(valloader_images, valloader_labels):
    ref = defaultdict(list)

    for i, data in enumerate(zip(valloader_images, valloader_labels)):
        inputs, labels = data[0], data[1]
        inputs, labels = inputs.cuda(), labels.cuda()

        if i == 5:
            break

        # Split by channel
        inputs0 = inputs[:, 0, ...].unsqueeze(1)
        labels0 = labels[:, 0, ...].unsqueeze(1)

        ref['image_ch0'].append(inputs0)
        ref['label_ch0'].append(labels0)
    
    return ref

def trainer_dlmd(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    trainloader_images, trainloader_labels, valloader_images, valloader_labels = split()
    print("Training dataset length: {}".format(len(trainloader_images)))
    print("Validation dataset length: {}".format(len(valloader_images)))

    wandb.init(
        # set the wandb project where this run will be logged
        project="swin-unet",

        # track hyperparameters and run metadata
        config={
            "learning_rate": base_lr,
            'weight_decay': 0.0001,
            "architecture": "swin-unet",
            "dataset": "dlmd",
        }
    )

    # set reference images
    ref = set_ref_images(valloader_images, valloader_labels)
    ref_label = ref['label_ch0'][0].cpu().detach().numpy()

    wandb.log(
        {"ground_truth": {
            "val_label": wandb.Image(ref_label.squeeze(0).transpose(1, 2, 0)),
        }}
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    # MSE loss, crop out borders
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader_images)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader_images), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:

        for i, data in enumerate(zip(trainloader_images, trainloader_labels)):

            inputs, labels = data[0], data[1]
            inputs, labels = inputs.cuda(), labels.cuda()

            inputs = inputs[:, 0, ...].unsqueeze(1)
            labels = labels[:, 0, ...].unsqueeze(1)

            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            loss = criterion(crop_pad(outputs), crop_pad(labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            
            wandb.log({"loss": loss.item()})
            
            if iter_num % 50 == 0:
                ref_output = model(ref['image_ch0'][0]).cpu().detach().numpy()

                wandb.log(
                        {"predictions": {
                            "train_input": wandb.Image(inputs[0].cpu().detach().numpy().transpose(1, 2, 0)),
                            "train_output": wandb.Image(outputs[:][0].cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]),
                            "train_label": wandb.Image(labels[0].cpu().detach().numpy().transpose(1, 2, 0)),
                            "val_output": wandb.Image(ref_output.squeeze(0).transpose(1, 2, 0))
                        }}
                    )

            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    wandb.finish()
    return "Training Finished!"
