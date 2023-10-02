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


gpu = 0
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
torch.cuda.set_device(gpu)
device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")


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

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
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
            "dataset": "synapse",
        }
    )

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:

        for i_batch, sampled_batch in enumerate(trainloader):

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            print("image_batch shape", image_batch.shape)
            print("label_batch shape", label_batch.shape)

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            print("outputs shape", outputs.shape)

            # loss_ce = ce_loss(outputs, label_batch[:].long())
            loss = criterion(outputs, label_batch)
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
                        "input": wandb.Image(image_batch[0].cpu().detach().numpy().transpose(1, 2, 0)),
                        "output": wandb.Image(outputs[:][0].cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]),
                        "label": wandb.Image(label_batch[0].unsqueeze(0).cpu().detach().numpy().transpose(1, 2, 0))
                    }}
                )

            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
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
