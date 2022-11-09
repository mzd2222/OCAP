import ast
from models.MobileNetV2_cifar_Model import *
from utils.Data_loader import Data_Loader_CIFAR
from utils.Functions import *

import argparse

# ---------------------------------------
activations = []          # activations for calculating mask
record_activations = []   # activations for calculating kl-div


# activation_hook for calculating mask
def mask_activation_hook(module, input, output):
    global activations
    activations.append(output.clone().detach().cpu())
    return


# activation_hook for calculating kl-div
def forward_activation_hook(module, input, output):
    global record_activations
    record = input[0].clone().detach().cpu()
    record_activations.append(record)
    return


def Compute_layer_mask(imgs, model, percent, device, activation_func):
    """
    :argument compute masks
    :param percent: saving ratio
    :return: masks dim:[layer_num, c]
    """

    percent = 1 - percent

    global activations
    model.eval()

    with torch.no_grad():
        imgs_masks = []
        hooks = []
        activations.clear()

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = module.register_forward_hook(mask_activation_hook)
                hooks.append(hook)

        imgs = imgs.to(device)
        _ = model(imgs)

        for hook in hooks:
            hook.remove()

        # ------ LAYER-BY-LAYER
        masks = []
        score_num_list = []
        for layer_activations in activations:
            if activation_func is not None:
                layer_activations = activation_func(layer_activations)
            # [img_num, c, h, w] => [img_num, c] --- [800, 64, 32, 32] => [800, 64]
            layer_activations_score = layer_activations.norm(dim=(2, 3), p=2)
            # [img_num, c]  eg [800, 64]
            layer_masks = torch.empty_like(layer_activations_score, dtype=torch.bool)
            # [image_num, c]
            for idx, imgs_activations_score in enumerate(layer_activations_score):
                # [c]
                sorted_tensor, index = torch.sort(imgs_activations_score)
                threshold_index = int(len(sorted_tensor) * percent)
                threshold = sorted_tensor[threshold_index]
                one_img_mask = imgs_activations_score.gt(threshold)
                layer_masks[idx] = one_img_mask

            """
            1 OCAP-AB
            """
            one_layer_mask = layer_masks[0]
            # [img_num, c] => [c]  [800, 64] => [64]
            for img in layer_masks[1:]:
                for channel_id, channel_mask in enumerate(img):
                    one_layer_mask[channel_id] = one_layer_mask[channel_id] | channel_mask

            """
            2 OCAP-FR
            """
            # [c]  [64]
            # score_num = torch.sum(layer_masks, dim=0)
            # score_num_list.append(score_num)
            # sorted_tensor, _ = torch.sort(score_num)
            # score_threshold_index = int(len(sorted_tensor) * percent)
            # score_threshold = sorted_tensor[score_threshold_index]
            # one_layer_mask = score_num.gt(score_threshold)

            masks.append(one_layer_mask)

        return masks, score_num_list


def pre_processing_Pruning(model: nn.Module, masks, jump_layers):

    model.eval()
    cfg = []
    count = 0
    cfg_mask = []
    pruned = 0
    total = 0
    cfg_old = []
    cfg_old2 = []

    for ii in masks:
        cfg_old.append(len(ii))
        cfg_old2.append(int(torch.sum(ii)))

    for index, module in enumerate(model.modules()):

        if isinstance(module, nn.BatchNorm2d):


            mask = masks[count]

            if count in jump_layers:
                mask = mask | True

            if (count + 1) % 3 == 0:
                mask = mask | True

            if count % 3 == 0 and count+1 < len(masks):
                change_num = int(torch.sum(masks[count])) - int(torch.sum(masks[count + 1]))

                i = 0
                while change_num != 0:
                    if change_num > 0:
                        if masks[count+1][i].item() is False:
                            masks[count+1][i] = torch.tensor(True, dtype=torch.bool)
                            change_num -= 1
                    if change_num < 0:
                        if masks[count][i].item() is False:
                            masks[count][i] = torch.tensor(True, dtype=torch.bool)
                            change_num += 1
                    i += 1

            if torch.sum(mask) == 0:
                mask[0] = 1

            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())

            total += len(mask)

            pruned += len(mask) - torch.sum(mask)

            count += 1

    pruned_ratio = pruned.detach().item() / total
    # print(cfg)
    # print(cfg_old)
    # print(cfg_old2)

    return cfg, cfg_mask, pruned_ratio


def change_mask(residual_mask, now_mask):
    change_num = int(torch.sum(now_mask)) - int(torch.sum(residual_mask))

    i = 0
    while change_num != 0:
        if change_num > 0:

            if now_mask[i].item() is True:
                now_mask[i] = torch.tensor(False, dtype=torch.bool)
                change_num -= 1
        if change_num < 0:

            if now_mask[i].item() is False:
                now_mask[i] = torch.tensor(True, dtype=torch.bool)
                change_num += 1
        i += 1

    if int(torch.sum(now_mask)) != int(torch.sum(residual_mask)):
        raise ValueError("channel is not the same")

    return now_mask


def Real_Pruning(new_model: nn.Module, old_model: nn.Module, cfg_masks, reserved_class):
    """
    :argument pruning according to the masks
    :param model:
    :param cfg_masks:
    :param reserved_class: reserved classes
    :return:
    """

    new_model.eval()
    old_model.eval()

    conv_idx = 0  # conv count
    bn_idx = 0  # bn count and mask id cound
    use_connect_sign = False

    for idx, (new_name_module, old_name_module) in enumerate(zip(new_model.named_modules(), old_model.named_modules())):

        new_name = new_name_module[0]
        new_module = new_name_module[1]
        old_name = old_name_module[0]
        old_module = old_name_module[1]

        if isinstance(old_module, nn.Conv2d):
            if conv_idx == 0:
                new_module.weight.data = old_module.weight.data.clone()
                conv_idx += 1
        if isinstance(old_module, nn.BatchNorm2d):
            if bn_idx == 0:
                new_module.weight.data = old_module.weight.data.clone()
                new_module.bias.data = old_module.bias.data.clone()
                new_module.running_mean = old_module.running_mean.clone()
                new_module.running_var = old_module.running_var.clone()
                bn_idx += 1

        # 负责剪InvertedResidual
        if isinstance(old_module, InvertedResidual):
            if old_module.use_res_connect:

                # 0
                # 0.0 conv
                conv_weigh = old_module.conv[0][0].weight.data.clone()
                new_module.conv[0][0].weight.data = conv_weigh.clone()[cfg_masks[bn_idx]]
                conv_idx += 1
                # 0.1 bn
                new_module.conv[0][1].num_features = int(torch.sum(cfg_masks[bn_idx]).item())
                new_module.conv[0][1].weight.data = old_module.conv[0][1].weight.data.clone()[cfg_masks[bn_idx]]
                new_module.conv[0][1].bias.data = old_module.conv[0][1].bias.data.clone()[cfg_masks[bn_idx]]
                new_module.conv[0][1].running_mean = old_module.conv[0][1].running_mean.clone()[cfg_masks[bn_idx]]
                new_module.conv[0][1].running_var = old_module.conv[0][1].running_var.clone()[cfg_masks[bn_idx]]
                bn_idx += 1

                # 1
                # 1.0 conv
                new_module.conv[1][0].in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())
                new_module.conv[1][0].groups = new_module.conv[1][0].in_channels
                new_module.conv[1][0].weight.data = old_module.conv[1][0].weight.data.clone()[cfg_masks[bn_idx]]
                new_module.conv[1][0].out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
                conv_idx += 1
                # 1.1 bn
                new_module.conv[1][1].num_features = int(torch.sum(cfg_masks[bn_idx]).item())
                new_module.conv[1][1].weight.data = old_module.conv[1][1].weight.data.clone()[cfg_masks[bn_idx]]
                new_module.conv[1][1].bias.data = old_module.conv[1][1].bias.data.clone()[cfg_masks[bn_idx]]
                new_module.conv[1][1].running_mean = old_module.conv[1][1].running_mean.clone()[cfg_masks[bn_idx]]
                new_module.conv[1][1].running_var = old_module.conv[1][1].running_var.clone()[cfg_masks[bn_idx]]
                bn_idx += 1

                # 2
                # 2.0 conv
                conv_weigh = old_module.conv[2].weight.data.clone()[:, cfg_masks[bn_idx - 1], :, :]
                new_module.conv[2].weight.data = conv_weigh.clone()
                new_module.conv[2].in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())

                conv_idx += 1
                # 2.1 bn
                new_module.conv[3].weight.data = old_module.conv[3].weight.data.clone()
                new_module.conv[3].bias.data = old_module.conv[3].bias.data.clone()
                new_module.conv[3].running_mean = old_module.conv[3].running_mean.clone()
                new_module.conv[3].running_var = old_module.conv[3].running_var.clone()
                bn_idx += 1

            else:
                for new_residual_module, old_residual_module in zip(new_module.modules(), old_module.modules()):
                    if isinstance(old_residual_module, nn.Conv2d):

                        if new_residual_module.groups != 1:
                            conv_weigh = old_residual_module.weight.data.clone()
                        else:
                            conv_weigh = old_residual_module.weight.data.clone()[:, cfg_masks[bn_idx - 1], :, :]
                        new_residual_module.in_channels = int(torch.sum(cfg_masks[bn_idx - 1]).item())

                        new_residual_module.weight.data = conv_weigh.clone()[cfg_masks[bn_idx]]
                        new_residual_module.out_channels = int(torch.sum(cfg_masks[bn_idx]).item())
                        if new_residual_module.groups != 1:
                            new_residual_module.groups = new_residual_module.in_channels
                        conv_idx += 1

                    if isinstance(old_residual_module, nn.BatchNorm2d):
                        new_residual_module.num_features = int(torch.sum(cfg_masks[bn_idx]).item())
                        new_residual_module.weight.data = old_residual_module.weight.data.clone()[cfg_masks[bn_idx]]
                        new_residual_module.bias.data = old_residual_module.bias.data.clone()[cfg_masks[bn_idx]]
                        new_residual_module.running_mean = old_residual_module.running_mean.clone()[cfg_masks[bn_idx]]
                        new_residual_module.running_var = old_residual_module.running_var.clone()[cfg_masks[bn_idx]]
                        bn_idx += 1

        # only for the last conv layer
        if isinstance(old_module, nn.Conv2d):
            if old_name == 'features.18.0':
                conv_weigh = old_module.weight.data.clone()
                new_module.weight.data = conv_weigh.clone()[cfg_masks[bn_idx]]
                new_module.out_channels = int(torch.sum(cfg_masks[bn_idx]).item())

        if isinstance(old_module, nn.BatchNorm2d):
            if old_name == 'features.18.1':
                new_module.num_features = int(torch.sum(cfg_masks[bn_idx]).item())
                new_module.weight.data = old_module.weight.data.clone()[cfg_masks[bn_idx]]
                new_module.bias.data = old_module.bias.data.clone()[cfg_masks[bn_idx]]
                new_module.running_mean = old_module.running_mean.clone()[cfg_masks[bn_idx]]
                new_module.running_var = old_module.running_var.clone()[cfg_masks[bn_idx]]
                bn_idx += 1

        if isinstance(old_module, nn.Linear):

            out_mask = torch.full([old_module.weight.data.size(0)], False, dtype=torch.bool)
            for ii in reserved_class:
                out_mask[ii] = True

            fc_data = old_module.weight.data.clone()[:, cfg_masks[-1]]

            fc_data = fc_data.clone()[out_mask, :]

            new_module.weight.data = fc_data.clone()
            new_module.bias.data = old_module.bias.data.clone()[out_mask]
            new_module.in_features = int(torch.sum(cfg_masks[-1]).item())
            new_module.out_features = int(torch.sum(out_mask).item())

    return new_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-dataSet_name", "--dataSet_name", help="CIFAR10, CIFAR100", default='CIFAR10')

    parser.add_argument("-reserved_c_num", "--reserved_c_num", help="reserved class num", default=5)
    parser.add_argument("-manual_radio", "--manual_radio", help="manual_radio", default=0.01)

    parser.add_argument("-train_b", "--train_batch_size", help="train_batch_size", default=256)
    parser.add_argument("-test_b", "--test_batch_size", help="test_batch_size", default=256)

    parser.add_argument("-epoch1", "--fine_tuning_epoch1", help="fine_tuning_epoch1", default=1)
    parser.add_argument("-test_epoch", "--test_epoch", help="test_epoch", default=1)

    parser.add_argument("-fine_batch_size", "--fine_tuning_batch_size", help="fine_tuning_batch_size", default=128)
    parser.add_argument("-fine_pics_num", "--fine_tuning_pics_num", help="fine_tuning_pics_num", default=256)

    parser.add_argument("-test_time", "--test_time", help="if test_time", default=False, type=ast.literal_eval)
    parser.add_argument("-test_latency", "--test_latency", help="if test_latency", default=False, type=ast.literal_eval)

    parser.add_argument("-test_range_n", "--test_range_n", help="test range(n)", default=5)

    args = parser.parse_args()

    dataSet_name = args.dataSet_name

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = Data_Loader_CIFAR(train_batch_size=int(args.train_batch_size),
                                    test_batch_size=int(args.test_batch_size),
                                    dataSet=dataSet_name, use_data_Augmentation=False,
                                    download=False, train_shuffle=True)

    if dataSet_name == "CIFAR10":
        model_path = 'MovileNetV2_cifar_before.pkl'
    elif dataSet_name == "CIFAR100":
        model_path = 'MovileNetV2_cifar100_before.pkl'

    model = torch.load(f=model_path).to(device)
    model.eval()

    test_time = args.test_time
    test_latency = args.test_latency

    version_id = 0
    model_id = 0

    if test_time:
        fine_tuning_epoch1 = int(args.fine_tuning_epoch1)
    else:
        fine_tuning_epoch = 50

    manual_radio = float(args.manual_radio)

    jump_layers_list = [i for i in range(52)]
    for iii in range(21, 50):
        jump_layers_list.remove(iii)


    fine_tuning_lr = 0.001
    fine_tuning_batch_size = int(args.fine_tuning_batch_size)
    fine_tuning_pics_num = int(args.fine_tuning_pics_num)

    use_KL_divergence = True
    divide_radio = 4
    redundancy_num = 100

    record_imgs_num = 512
    record_batch = 128

    frozen = False

    model_save_path = 'models'

    max_kc = None
    min_kc = None
    FLOPs_radio = 0.00
    Parameters_radio = 0.00
    # ----------------------------------------------------------------------
    # --------------
    # ----------------------------------------------------------------------

    reserved_classes_list = []
    for _ in range(5):
        reserved_classes = []
        for _ in range(int(args.reserved_c_num)):
            reserved_classes.append(random.randint(0, 10))
        reserved_classes_list.append(reserved_classes)
        reserved_classes.clear()


    for reserved_classes in reserved_classes_list:

        activations.clear()
        record_activations.clear()

        # ----------Step 1: Do a certain amount of forward inference forward,
        # and record the picture and intermediate activation value
        record_dataloader = read_Img_by_class(pics_num=record_imgs_num,
                                              batch_size=record_batch,
                                              target_class=reserved_classes,
                                              data_loader=data_loader.train_data_loader,
                                              shuffle=False)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(forward_activation_hook)

        # Simulated actual forward reasoning (before pruning) and recorded data
        model.eval()
        with torch.no_grad():
            for forward_x, _ in record_dataloader:
                forward_x = forward_x.to(device)
                _ = model(forward_x)
        hook.remove()
        model.eval()

        old_FLOPs, old_parameters = cal_FLOPs_and_Parameters(model, device)

        """
        Test different activation functions
        """
        # [None, nn.ReLU(), nn.LeakyReLU(), F.relu6, nn.Sigmoid(), nn.Tanh(), nn.ELU(), nn.Hardswish()]
        # activations_list = [None, nn.ReLU(), nn.LeakyReLU()]
        # for i in range(0, 101, 5):
        #     # print(i / 100.0)
        #     activations_list.append(nn.LeakyReLU(negative_slope=i/100.0))
        # activations_list.extend([nn.Sigmoid(), nn.Tanh(), nn.ELU(), nn.Hardswish()])

        # activations_list = [nn.ReLU(), nn.ReLU()]
        # negative_slope_list = [0.00, 0.00, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.11, 0.12, 0.13, 0.14, 0.16, 0.18]
        # for negative_slope in negative_slope_list:
        #     activations_list.append(nn.LeakyReLU(negative_slope=negative_slope))


        for act_func in [nn.LeakyReLU(negative_slope=0.06)]:
            # ----------The second part: Obtain the reserved classes according to the forward inference picture,
            # calculate the mask, prune and obtain the new model after pruning

            for i in range(int(args.test_range_n)):

                if test_time:
                    if i == 0:
                        fine_tuning_epoch = int(args.test_epoch)
                    else:
                        fine_tuning_epoch = fine_tuning_epoch1

                    if i != 0:
                        torch.cuda.synchronize()
                        time_in_pr = time.time()

                # Select a portion of the recorded data to calculate the mask
                imgs_tensor = choose_mask_imgs(target_class=reserved_classes,
                                               data_loader=record_dataloader,
                                               pics_num=30)
                layer_masks, score_num_list = Compute_layer_mask(imgs=imgs_tensor, model=model, percent=manual_radio,
                                                                 device=device, activation_func=act_func)

                # Pre-prune and calculate the mask
                cfg, cfg_masks, filter_remove_radio = pre_processing_Pruning(model, layer_masks, jump_layers=jump_layers_list)
                filter_remain_radio = 1 - filter_remove_radio

                redundancy_num_list = [100]
                for redundancy_num in redundancy_num_list:

                    # Formal pruning, parameter copy
                    new_model = torch.load(f=model_path).to(device)
                    model_after_pruning = Real_Pruning(new_model=new_model, old_model=model, cfg_masks=cfg_masks, reserved_class=reserved_classes)

                    if test_time:
                        if i != 0:
                            torch.cuda.synchronize()
                            time_out_pr = time.time()
                            time_prune = time_out_pr - time_in_pr
                            print("pruning time：" + str(time_prune) + "s")

                    # Calculate a variety of compression standards
                    model_after_pruning.eval()
                    new_FLOPs, new_parameters = cal_FLOPs_and_Parameters(model_after_pruning, device)
                    FLOPs_radio = new_FLOPs / old_FLOPs
                    Parameters_radio = new_parameters / old_parameters

                    # multi GPUs
                    if torch.cuda.device_count() > 1:
                        print("Let's use", torch.cuda.device_count(), "GPUs!")
                        model_after_pruning = nn.DataParallel(model_after_pruning)
                    model_after_pruning.to(device)

                    print("model_id:" + str(model_id)
                          + " ---filter_remain_radio:" + str(filter_remain_radio)
                          + " ---FLOPs_radio:" + str(FLOPs_radio)
                          + " ---Parameters_radio:" + str(Parameters_radio)
                          + "")

                    # ----------Step 3: Fine-tune a portion of the picture from the forward
                    # inference record using the algorithm

                    if test_time:
                        if i != 0:
                            torch.cuda.synchronize()
                            time_in_pr = time.time()

                    # --------------------------------------------- fine-tuning
                    # choose data for fine-tuning
                    fine_tuning_loader, max_kc, min_kc = get_fine_tuning_data_loader2(record_activations,
                                                                                      reserved_classes,
                                                                                      pics_num=fine_tuning_pics_num,
                                                                                      batch_size=fine_tuning_batch_size,
                                                                                      data_loader=record_dataloader,
                                                                                      redundancy_num=redundancy_num,
                                                                                      divide_radio=divide_radio,
                                                                                      use_max=True)

                    if test_time:
                        if i != 0:
                            torch.cuda.synchronize()
                            time_out_pr = time.time()
                            time_choose = time_out_pr - time_in_pr
                            print("data selection time：" + str(time_choose) + "s")

                        if i != 0:
                            torch.cuda.synchronize()
                            time_in_pr = time.time()

                    # fine-tuning
                    best_acc, acc_list, loss_list = fine_tuning(model_after_pruning, reserved_classes,
                                                                EPOCH=fine_tuning_epoch, lr=fine_tuning_lr,
                                                                device=device,
                                                                train_data_loader=fine_tuning_loader,
                                                                test_data_loader=data_loader.test_data_loader,
                                                                model_save_path=model_save_path,
                                                                use_all_data=False,
                                                                frozen=frozen,
                                                                dataset_num_class=data_loader.dataset_num_class)

                    if test_time:
                        if i != 0:
                            torch.cuda.synchronize()
                            time_out_pr = time.time()
                            time_fine_tune = time_out_pr - time_in_pr
                            print("fine-tuning time：" + str(time_fine_tune) + "s")

                    # test latency
                    if args.test_latency:
                        model.eval()
                        model_after_pruning.eval()
                        input_size = (100, 3, 32, 32)
                        latency_new = compute_latency_ms_pytorch(model_after_pruning, input_size, iterations=10, device='cuda')
                        latency_old = compute_latency_ms_pytorch(model, input_size, iterations=10, device='cuda')
                        latency_old = compute_latency_ms_pytorch(model, input_size, iterations=100, device='cuda')
                        latency_new = compute_latency_ms_pytorch(model_after_pruning, input_size, iterations=100, device='cuda')
                        print('model:{}, | latency: {}'.format('new', latency_new))
                        print('model:{}, | latency: {}'.format('old', latency_old))

                    print("model_id:---" + str(model_id) +
                          " best_acc:----" + str(best_acc) +
                          " reserved_classes:---" + str(reserved_classes) +
                          " manual_radio:---" + str(manual_radio) +
                          " filter_remain_radio:---" + str(filter_remain_radio) +
                          '\n')

                    model_id += 1
