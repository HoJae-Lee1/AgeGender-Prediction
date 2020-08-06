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
import matplotlib.pyplot as plt

def train(config):
    train_set = AgeGenderDataset(config.train_csv_path, min=config.min, max=config.max, interval=config.interval, transform=image_transformer()['train'])
    trainloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_set = AgeGenderDataset(config.test_csv_path, min=config.min, max=config.max, interval=config.interval, transform=image_transformer()['test'])
    testloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if config.checkpoint is None:
        start_epoch = 0
        age_clss = (config.max - config.min) // config.interval
        model = AgeGenModel(age_clss)
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    else:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    model = model.to(device)

    ## Criterion & Optimizer
    age_criterion = nn.BCELoss()
    gender_criterion = nn.BCELoss()

    train_loss_record, test_loss_record = [], []
    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        for mode in config.mode:
            if mode == 'train':
                model.train()
                train_loader_pbar = tqdm(trainloader)

                train_loss = 0.0
                total = 0
                total_acc = 0.0
                for batch_idx, data in enumerate(train_loader_pbar):
                    img, age, gender = data
                    img = img.to(device)
                    age = age.to(device)
                    gender = gender.to(device)

                    #forward
                    age_pred, gender_pred = model(img)
                    #Loss
                    age_loss = age_criterion(age_pred, age.to(device))
                    #Backward
                    optimizer.zero_grad()
                    age_loss.backward()
                    #Update Weights
                    optimizer.step()

                    train_loss += age_loss.item()
                    total += 1
                    #Calculate Accuracy
                    int_age_pred = torch.argmax(age_pred, dim=1)
                    int_age = torch.argmax(age, dim=1)
                    acc =(int_age == int_age_pred).sum().item()/config.batch_size
                    total_acc += acc

                    train_loader_pbar.set_description(
                        '[training] epoch: %d/%d, ' % (epoch+1, config.num_epochs) +
                        'train_loss: %.3f, ' % (train_loss/total) +
                        'Accuracy: %.3f, ' % (total_acc/total) +
                        'predicted: %d, ' % (torch.argmax(age_pred[0]) * 5 + config.min) +
                        'Answer: %d' % (torch.argmax(age[0]) * 5 + config.min)
                    )

                train_loss_record.append(train_loss / total)
                if config.model_save == True:
                        model_out_path = config.model_save_path + str(epoch) + '.pth'
                        state ={'epoch': epoch, 'model': model, 'optimizer': optimizer}
                        torch.save(state, model_out_path)

            elif mode == 'test':
                model.eval()
                test_loss = 0.0
                total = 0
                total_acc = 0.0

                test_loader_pbar = tqdm(testloader)
                for batch_idx, data in enumerate(test_loader_pbar):
                    img, age, gender = data
                    img = img.to(device)
                    age = age.to(device)
                    gender = gender.to(device)

                    # forward
                    age_pred, gender_pred = model(img)
                    # Loss
                    age_loss = age_criterion(age_pred, age.to(device))

                    test_loss += age_loss.item()
                    total += 1

                    # Calculate Accuracy
                    int_age_pred = torch.argmax(age_pred, dim=1)
                    int_age = torch.argmax(age, dim=1)
                    acc = (int_age == int_age_pred).sum().item() / config.batch_size
                    total_acc += acc

                    test_loader_pbar.set_description(
                        '[testing] epoch: %d/%d, ' % (epoch+1, config.num_epochs) +
                        'valid_loss: %.3f, ' % (test_loss / total) +
                        'Accuracy: %.3f, ' % (total_acc / total) +
                        'predicted: %d, ' % (torch.argmax(age_pred[0]) * 5 + config.min) +
                        'Answer: %d ' % (torch.argmax(age[0]) * 5 + config.min)
                    )

                test_loss_record.append(test_loss / total)

    ## Plot the history of train/test loss
    plt.plot(train_loss_record, label='Training loss')
    plt.plot(test_loss_record, label='Validation loss')
    plt.legend(frameon=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_path', default='./data/wiki_crop_test.csv')
    parser.add_argument('--test_csv_path', default='./data/wiki_crop_train.csv')
    # parser.add_argument('--train_csv_path', default='~/Intern2020-2/AgeGenderPrediction/data/train_meta.csv')
    # parser.add_argument('--test_csv_path', default='~/Intern2020-2/AgeGenderPrediction/data/test_meta.csv')

    parser.add_argument('--model_save', dafault=True)
    parser.add_argument('--model_save_path', type=str, default='model_load/interval10_age_model_epoch_')
    # parser.add_argument('--checkpoint', default='./model_load/interval10_age_model_epoch_45.pth')
    parser.add_argument('--checkpoint', default=None)

    parser.add_argument('--img_resize', type=int, default=48)
    parser.add_argument('--mode', default=['train', 'test'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=30)

    parser.add_argument('--min', type=int, default=10, help='minimum age')
    parser.add_argument('--max', type=int, default=60, help='maximum age')
    parser.add_argument('--interval', type=int, default=10)

    config = parser.parse_args()
    train(config)

