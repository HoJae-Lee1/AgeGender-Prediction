import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torchvision.transforms as transforms

## My .py modules
from Dataloader import AgeGenderDataset
from model import *
from util import *

## Others
import argparse
from tqdm import tqdm

def train(config):
    train_set = AgeGenderDataset(config.train_csv_path, min=config.min, max=config.max, transform=image_transformer()['train'])
    trainloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_set = AgeGenderDataset(config.test_csv_path, min=config.min, max=config.max, transform=image_transformer()['test'])
    testloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)

    age_clss = (config.max - config.min)//5
    model = AgeGenModel(age_clss)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print('use multiple GPUs')
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model = model.to(device)

    ## Criterion & Optimizer
    age_criterion = nn.BCELoss()
    gender_criterion = nn.BCELoss()

    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

    for epoch in range(config.num_epochs):
        for mode in config.mode:
            if mode == 'train':
                model.train()
                train_loader_pbar = tqdm(trainloader)

                train_loss = 0.0
                total = 0

                for batch_idx, data in enumerate(train_loader_pbar):
                    img, age, gender = data

                    optimizer.zero_grad()
                    age_pred, gender_pred = model(img)

                    age_loss = age_criterion(age_pred, age.to(device))
                    age_loss.backward()
                    optimizer.step()
                    train_loss += age_loss.item()
                    total += 1

                    train_loader_pbar.set_description(
                        '[training] epoch: %d/%d, ' % (epoch, config.num_epochs),
                        'train_loss: %.3f, ' % (train_loss/total),
                        'predicted: %d, ' % (torch.argmax(age_pred[0]) * 5 + config.min),
                        'Answer: %d' % (torch.argmax(age[0]) * 5 + config.min)
                    )

                if config.model_save == True:
                        model_out_path = 'model_load/model_epoch_{}.pth'.format(epoch)
                        torch.save(model, model_out_path)
            elif mode == 'eval':
                model.eval()
                test_loss = 0.0
                total = 0

                test_loader_pbar = tqdm(testloader)
                for batch_idx, data in enumerate(test_loader_pbar):
                    img, age, gender = data

                    optimizer.zero_grad()
                    age_pred, gender_pred = model(img)

                    age_loss = age_criterion(age_pred, age.to(device))

                    test_loss += age_loss.item()
                    total += 1
                    test_loader_pbar.set_description(
                        '[testing] epoch: %d/%d, ' % (epoch, config.num_epochs),
                        'valid_loss: %.3f, ' % (test_loss/total),
                        'predicted: %d, ' % (torch.argmax(age_pred[0]) * 5 + config.min),
                        'Answer: %d ' % (torch.argmax(age[0]) * 5 + config.min)
                    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_path', default='./data/wiki_crop_test.csv')
    parser.add_argument('--test_csv_path', default='./data/wiki_crop_train.csv')

    parser.add_argument('--img_resize', type=int, defualt=48)
    parser.add_argument('--model_save', default=True)
    parser.add_argument('--mode', default=['train', 'test'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=30)

    parser.add_argument('--min', type=int, default=10, help='minimum age')
    parser.add_argument('--max', type=int, default=60, help='maximum age')

    config = parser.parse_args()

