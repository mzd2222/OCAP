import random
import sys
import time
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, dataloader

from fvcore.nn import FlopCountAnalysis

class myDataset(Dataset):
    def __init__(self, img_list, label_list):
        """
        :argument transform img and label to dataset for fine-tuning
        """
        self.img_list = img_list
        self.label_list = label_list

    def __getitem__(self, idx):
        img, label = self.img_list[idx], self.label_list[idx]
        return img, label

    def __len__(self):
        return len(self.img_list)


class myDataset2(Dataset):
    def __init__(self, img_list):
        """
        :argument transform img and label to dataset for calculating masks
        """
        self.img_list = img_list

    def __getitem__(self, idx):
        img = self.img_list[idx]
        return img

    def __len__(self):
        return len(self.img_list)



def read_Img_by_class(target_class, pics_num, data_loader, batch_size, shuffle=True):
    """
    :argument record imgs and labels
    :param batch_size:
    :param target_class:  eg: [0, 1, 2, 3]
    :param pics_num:  N
    :param data_loader:  data_loader
    :return: data_loader
    """

    counts = []
    inputs = []
    labels = []

    for idx in range(len(target_class)):
        counts.append(0)
        inputs.append([])
        labels.append([])

    for data, label in data_loader:

        if sum(counts) == len(target_class) * pics_num:
            break

        for idx in range(len(label)):

            if label[idx] in target_class:
                list_idx = target_class.index(label[idx])
            else:
                continue

            if counts[target_class.index(label[idx])] < pics_num:
                inputs[list_idx].append(data[idx])
                labels[list_idx].append(label[idx])
                counts[list_idx] += 1

    imgs = []
    targets = []
    for i, j in zip(inputs, labels):
        imgs += i
        targets += j

    mydataset = myDataset(imgs, targets)
    record_dataloader = dataloader.DataLoader(mydataset, batch_size=batch_size, shuffle=shuffle)

    return record_dataloader


def choose_mask_imgs(target_class, pics_num, data_loader):
    """
    :argument choose imgs for calculating masks
    """

    counts = []
    inputs = []

    for idx in range(len(target_class)):
        counts.append(0)

    for data, label in data_loader:

        if sum(counts) == len(target_class) * pics_num:
            break

        for idx in range(len(label)):
            if counts[target_class.index(label[idx])] < pics_num:
                inputs.append(data[idx])
                counts[target_class.index(label[idx])] += 1

    imgs = torch.stack(inputs, dim=0)

    return imgs


def choose_mask_imgs2(target_class, pics_num, data_loader, batch_size, shuffle=True):
    """
    :argument choose imgs for calculating masks, return data loader form
    """

    counts = []
    inputs = []

    for idx in range(len(target_class)):
        counts.append(0)

    for data, label in data_loader:

        if sum(counts) == len(target_class) * pics_num:
            break

        for idx in range(len(label)):
            if counts[target_class.index(label[idx])] < pics_num:
                inputs.append(data[idx])
                counts[target_class.index(label[idx])] += 1

    mydataset = myDataset2(inputs)
    imgs_dataloader = dataloader.DataLoader(mydataset, batch_size=batch_size, shuffle=shuffle)

    return imgs_dataloader



def test_reserved_classes(model, reserved_classes, test_data_loader,
                          device, is_print=True, test_class=True, dataset_num_class=10,
                          change_class=True):

    model.to(device)
    model.eval()

    dataset_num_class = dataset_num_class

    if test_class:
        class_correct = []
        class_num = []
        for _ in range(dataset_num_class):
            class_correct.append(0)
            class_num.append(0)

    with torch.no_grad():
        correct = 0
        num_data_all = 0
        for data, label in test_data_loader:
            input = data.to(device)
            target = label.to(device)  # [b]

            masks = torch.full([len(target)], False, dtype=torch.bool).to(device)

            for idx, i in enumerate(target):
                if i in reserved_classes:
                    masks[idx] = True

            if torch.sum(masks) == 0:
                continue

            input = input[masks]
            target = target[masks]

            output = model(input)
            pred = torch.argmax(output, 1)

            if change_class:
                for idx, item in enumerate(pred):
                    # print(len(pred), idx, len(reserved_classes), int(item))
                    pred[idx] = reserved_classes[int(item)]

            if test_class:
                for index in range(len(target)):
                    if pred[index] == target[index]:
                        class_correct[target[index]] += 1
                    class_num[target[index]] += 1

            correct += (pred == target).sum()
            num_data_all += len(target)

        total_acc = float(correct.item() / num_data_all)

        if test_class:
            class_acc = []
            for correct, nums in zip(class_correct, class_num):
                if nums == 0:
                    nums = 1
                class_acc.append(correct / nums)

        if is_print:
            if test_class:
                print('\n',
                      'each class corrects: ', class_correct, '\n',
                      'each class accuracy: ', class_acc, '\n',
                      'total accuracy: ', total_acc)
            else:
                print('\n', 'total accuracy: ', total_acc)

        if test_class:
            return round(total_acc, 4), class_correct, class_acc
        else:
            return round(total_acc, 4), None, None



def fine_tuning(model, reserved_classes, EPOCH, lr, model_save_path,
                train_data_loader, test_data_loader, device,
                use_all_data=True, frozen=False, dataset_num_class=10):

    # frozen

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)
    # optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    optimizer.zero_grad()

    best_acc = 0

    acc_list = []
    loss_list = []

    for epoch in range(EPOCH):
        model.train()

        epoch_loss = 0
        item_times = 0

        for idx, (data, label) in enumerate(tqdm(train_data_loader, desc='fine_tuning: ', file=sys.stdout)):
            data = data.to(device)
            label = label.to(device)

            if use_all_data:
                masks = torch.full([len(label)], False)

                for idx0, i in enumerate(label):
                    if i in reserved_classes:
                        masks[idx0] = True

                data = data[masks, :, :, :]
                label = label[masks]

            for idx0, item in enumerate(label):
                label[idx0] = reserved_classes.index(int(item))

            output = model(data)
            loss = loss_func(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
            item_times += 1

        scheduler.step()

        epoch_acc, _, _ = test_reserved_classes(model, reserved_classes,
                                                test_data_loader,
                                                device,
                                                test_class=False,
                                                is_print=False,
                                                dataset_num_class=dataset_num_class)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # print('model save')
            torch.save(model, model_save_path)

        if epoch % 10 == 6690 and epoch != 0:
            test_reserved_classes(model, reserved_classes,
                                  test_data_loader,
                                  device,
                                  test_class=True,
                                  is_print=True,
                                  dataset_num_class=dataset_num_class)
        else:
            print("epoch:" + str(epoch) + "\tepoch_acc: "
                  + str(epoch_acc) + "\tepoch_loss: " + str(round(epoch_loss / item_times, 5)))

        acc_list.append(epoch_acc)
        loss_list.append(round(epoch_loss / item_times, 5))

    return best_acc, acc_list, loss_list



def get_fine_tuning_data_loader2(record_features, reserved_classes, pics_num, data_loader, batch_size,
                                 divide_radio=2, redundancy_num=50, use_max=True):
    """
    :argument choose dataset for fine-tuning using kl_div

    :param record_features: (list)features the order is same as that in data_loader
    :param reserved_classes: eg: [0, 1, 2, 3]
    :param pics_num:  N
    :param data_loader: saved images
    :param batch_size: data_loader batch_size
    :param divide_radio: d
    :param redundancy_num: N_R
    :param use_max: True save max, delete small   False save small, delete max

    :return: fine_tuning data_loader
    """

    counts = []
    img_list = []
    label_list = []
    redundancy_counts = []
    feature_list = []

    for idx in range(len(reserved_classes)):
        counts.append(0)
        redundancy_counts.append(0)
        img_list.append([])
        label_list.append([])
        feature_list.append([])

    image_Kc_list = np.zeros([len(reserved_classes), pics_num])

    for (data, label), features in tqdm(zip(data_loader, record_features), desc='choosing fine tuning data: ', file=sys.stdout):

        if sum(counts) == len(reserved_classes) * pics_num and sum(redundancy_counts) == len(reserved_classes) * redundancy_num:
            break

        for idx in range(len(label)):

            list_idx = reserved_classes.index(label[idx])

            if counts[list_idx] < pics_num:

                if counts[list_idx] == 0:
                    if use_max:
                        image_Kc_list[list_idx][0] = 1.0
                    else:
                        image_Kc_list[list_idx][0] = 0.0001

                else:

                    if counts[list_idx] < pics_num / divide_radio:

                        old_features = torch.stack(feature_list[list_idx])
                        Kc = F.kl_div(features[idx].softmax(dim=0).log(), old_features.softmax(dim=1),
                                      reduction='batchmean') / len(old_features)

                    else:
                        samples = random.sample([ig for ig in range(counts[list_idx])], int(pics_num / divide_radio))
                        feature_mask = torch.full([counts[list_idx]], False)
                        for i in range(counts[list_idx]):
                            if i in samples:
                                feature_mask[i] = True
                        old_features = torch.stack(feature_list[list_idx])[feature_mask]

                        Kc = F.kl_div(features[idx].softmax(dim=0).log(), old_features.softmax(dim=1),
                                      reduction='batchmean') / len(old_features)

                    image_Kc_list[list_idx][counts[list_idx]] = Kc

                img_list[list_idx].append(data[idx])
                label_list[list_idx].append(label[idx])
                feature_list[list_idx].append(features[idx])
                counts[list_idx] += 1

            # redundancy
            elif counts[list_idx] == pics_num and redundancy_counts[list_idx] < redundancy_num:

                if use_max:
                    Kc_min = min(image_Kc_list[list_idx])
                    Kc_min_idx = np.argmin(image_Kc_list[list_idx])
                else:
                    Kc_max = max(image_Kc_list[list_idx])
                    Kc_max_idx = np.argmax(image_Kc_list[list_idx])

                samples = random.sample([ig for ig in range(counts[list_idx])], int(pics_num / divide_radio))
                feature_mask = torch.full([counts[list_idx]], False, dtype=torch.bool)
                for i in range(counts[list_idx]):
                    if i in samples:
                        feature_mask[i] = True
                old_features = torch.stack(feature_list[list_idx])[feature_mask]

                Kc = F.kl_div(features[idx].softmax(dim=0).log(), old_features.softmax(dim=1),
                              reduction='batchmean') / len(old_features)

                if use_max:
                    if Kc > Kc_min:
                        image_Kc_list[list_idx][Kc_min_idx] = Kc
                        img_list[list_idx][Kc_min_idx] = data[idx]
                        label_list[list_idx][Kc_min_idx] = label[idx]
                        feature_list[list_idx][Kc_min_idx] = features[idx]
                        redundancy_counts[list_idx] += 1

                else:
                    if Kc < Kc_max:
                        image_Kc_list[list_idx][Kc_max_idx] = Kc
                        img_list[list_idx][Kc_max_idx] = data[idx]
                        label_list[list_idx][Kc_max_idx] = label[idx]
                        feature_list[list_idx][Kc_max_idx] = features[idx]
                        redundancy_counts[list_idx] += 1


    imgs = []
    labels = []
    for i, j in zip(img_list, label_list):
        imgs += i
        labels += j

    max_kc = []
    min_kc = []
    for kk in image_Kc_list:
        kk = list(kk)
        if use_max:
            kk.remove(max(kk))
        else:
            kk.remove(min(kk))
        max_kc.append(round(max(kk), 6))
        min_kc.append(round(min(kk), 6))

    mydataset = myDataset(imgs, labels)

    new_data_loader = dataloader.DataLoader(mydataset, batch_size=batch_size, shuffle=True)

    return new_data_loader, max_kc, min_kc



def cal_feature_mean(record_features, reserved_classes):

    means = []
    for i in record_features:
        means.append(torch.mean(i, dim=0))
    divide_num = int(len(means) / len(reserved_classes))

    mean_list = []
    for i in range(len(reserved_classes)):
        mean_list.append(torch.mean(torch.stack(means[i*divide_num:(i+1)*divide_num], dim=0), dim=0))

    return mean_list


def cal_FLOPs_and_Parameters(model, device):
    """
    :return: FLOPs:G parameters:M
    """

    tensor = torch.rand(1, 3, 32, 32).to(device)

    flops = FlopCountAnalysis(model, (tensor, ))

    flops_total = flops.total() / 10e9  # G
    parameters_total = sum([param.nelement() for param in model.parameters()]) / 10e6   # M

    return flops_total, parameters_total



def compute_latency_ms_pytorch(model, input_size, iterations=None, device=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model.eval()
    model = model.cuda()
    input = torch.randn(*input_size).cuda()

    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in tqdm(range(iterations)):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    # FPS = 1000 / latency (in ms)
    return latency
