import argparse, logging, os, random, sys, time

from datasets.dataset_dlmd import crop_pad, split

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms

from tensorboardX import SummaryWriter

import wandb

from tqdm import tqdm


gpu = 2
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
torch.cuda.set_device(gpu)
device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")


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

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    # MSE loss, crop out borders
    ce_loss = CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.1)

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

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader_images)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader_images), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:

        for i, data in enumerate(zip(trainloader_images, trainloader_labels)):

            inputs, labels = data[0], data[1]
            inputs, labels = inputs.cuda(), labels.cuda()

            print("image shape", inputs.shape)
            print("label shape", labels.shape)

            inputs = inputs[:, 0, ...].unsqueeze(1)
            labels = labels[:, 0, ...].unsqueeze(1)

            print("image single channel shape", inputs.shape)
            print("label single channel shape", labels.shape)

            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            print("outputs shape", outputs.shape)

            # loss_ce = ce_loss(outputs, labels[:].long())
            loss = criterion(crop_pad(outputs), crop_pad(labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_

            iter_num = iter_num + 1
            
            writer.add_scalar('info/lr', base_lr, iter_num)
            writer.add_scalar('info/loss', loss, iter_num)

            wandb.log({"lr": base_lr})
            wandb.log({"loss": loss})
            
            wandb.log(
                    {"predictions": {
                        "input": wandb.Image(inputs[0].cpu().detach().numpy().transpose(1, 2, 0)),
                        "output": wandb.Image(outputs[:][0].cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]),
                        "label": wandb.Image(labels[0].unsqueeze(0).cpu().detach().numpy().transpose(1, 2, 0))
                    }}
                )

            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 20 == 0:
                image = inputs[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = labels[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

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

    writer.close()
    wandb.finish()
    return "Training Finished!"
