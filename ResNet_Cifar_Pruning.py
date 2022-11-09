import argparse
import ast
from models.ResNet_cifar_Model import *
from utils.Data_loader import Data_Loader_CIFAR
from utils.Functions import *

# ---------------------------------------
activations = []
record_activations = []


# activation_hook for mask
def mask_activation_hook(module, input, output):
    global activations
    activations.append(output.clone().detach().cpu())
    return


# activation_hook for kl-div
def forward_activation_hook(module, input, output):
    global record_activations
    # input_size [b, 256, 8, 8]
    # output_size [b, 256, 1, 1]
    record = output.clone().detach().view(output.size(0), -1).cpu()
    record_activations.append(record)
    return


def Compute_layer_mask(imgs_dataloader, model, percent, device, activation_func):
    """
    :argument Calculate masks based on the input image
    :param percent: Proportion of retention
    :return: masks dim [layer_num, c]
    """

    percent = 1 - percent  # Calculate the percentage to be cut by retaining the percentage

    global activations
    model.eval()

    with torch.no_grad():
        imgs_masks = []
        hooks = []
        activations.clear()
        new_activations = []

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = module.register_forward_hook(mask_activation_hook)
                hooks.append(hook)

        batch_times = 0
        for imgs in imgs_dataloader:
            imgs = imgs.to(device)
            _ = model(imgs)
            batch_times += 1

        for hook in hooks:
            hook.remove()

        num_layers = int(len(activations) / batch_times)

        for idx1 in range(num_layers):

            layer_list = []

            for idx2 in range(batch_times):
                layer_list.append(activations[idx2 * num_layers + idx1])

            combine_activation = torch.cat(layer_list, dim=0)

            new_activations.append(combine_activation)

        for hook in hooks:
            hook.remove()

        # ------layer-by-layer
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
                threshold_index = min(int(len(sorted_tensor) * percent), len(sorted_tensor) - 1)
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


def pre_processing_Pruning(model: nn.Module, masks, jump_layers=2):

    model.eval()
    cfg = []
    count = 0
    cfg_mask = []
    pruned = 0
    total = 0

    for index, module in enumerate(model.modules()):

        if isinstance(module, nn.BatchNorm2d):

            mask = masks[count]
            if count < jump_layers:
                mask = mask | True

            if torch.sum(mask) == 0:
                mask[0] = 1

            cfg.append(int(torch.sum(mask)))

            cfg_mask.append(mask.clone())

            total += len(mask)

            pruned += len(mask) - torch.sum(mask)

            count += 1

    pruned_ratio = pruned.detach().item() / total

    return cfg, cfg_mask, pruned_ratio


def Real_Pruning(old_model: nn.Module, new_model: nn.Module, cfg_masks, reserved_class):
    """
    :argument According to cfg_mask, which is the mask of each bn layer, the parameters of the original model are
              copied to the new model, while the cs layer and linear layer of the new model are adjusted
    """

    old_model.eval()
    new_model.eval()
    old_modules_list = list(old_model.named_modules())
    new_modules_list = list(new_model.named_modules())

    bn_idx = 0  # bn count
    conv_idx = 0  # conv count

    current_mask = torch.ones(16)  # bn mask
    next_mask = cfg_masks[bn_idx]

    for idx, (old, new) in enumerate(zip(old_modules_list, new_modules_list)):

        old_name = old[0]
        new_name = new[0]
        old_module = old[1]
        new_module = new[1]

        if isinstance(old_module, nn.BatchNorm2d):

            current_mask = next_mask
            next_mask = cfg_masks[bn_idx + 1 if bn_idx + 1 < len(cfg_masks) else bn_idx]

            if isinstance(old_modules_list[idx + 1][1], channel_selection):
                new_module.weight.data = old_module.weight.data.clone()
                new_module.bias.data = old_module.bias.data.clone()
                new_module.running_mean = old_module.running_mean.clone()
                new_module.running_var = old_module.running_var.clone()

                # adjust cs layer index
                new_modules_list[idx + 1][1].indexes.data = current_mask.clone()

            else:

                new_module.weight.data = old_module.weight.data.clone()[current_mask]
                new_module.bias.data = old_module.bias.data.clone()[current_mask]
                new_module.running_mean = old_module.running_mean.clone()[current_mask]
                new_module.running_var = old_module.running_var.clone()[current_mask]

            bn_idx += 1

        if isinstance(old_module, nn.Conv2d):

            if conv_idx == 0:
                new_module.weight.data = old_module.weight.data.clone()
                conv_idx += 1

            elif not isinstance(old_modules_list[idx - 2][1], channel_selection) and \
                    not isinstance(old_modules_list[idx - 2][1], nn.BatchNorm2d):
                # print(old_name, new_name)
                new_module.weight.data = old_module.weight.data.clone()

            else:
                conv_weight = old_module.weight.data.clone()[:, current_mask, :, :]

                if conv_idx % 3 != 0:
                    conv_weight = conv_weight[next_mask, :, :, :]

                new_module.weight.data = conv_weight.clone()

                conv_idx += 1

        if isinstance(old_module, nn.Linear):
            input_size = sum(current_mask)
            out_size = len(reserved_class)
            new_model.fc = nn.Linear(input_size, out_size)

            out_mask = torch.full([old_module.weight.data.size(0)], False, dtype=torch.bool)
            for i in reserved_class:
                out_mask[i] = True

            fc_data = old_module.weight.data.clone()[:, current_mask]
            fc_data = fc_data[out_mask, :]

            new_model.fc.weight.data = fc_data.clone()
            new_model.fc.bias.data = old_module.bias.data.clone()[out_mask]

    return new_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-reserved_c_num", "--reserved_c_num", help="reserved class num", default=5)
    parser.add_argument("-manual_radio", "--manual_radio", help="manual_radio", default=0.05)

    parser.add_argument("-dataSet_name", "--dataSet_name", help="CIFAR10, CIFAR100", default='CIFAR10')

    parser.add_argument("-train_b", "--train_batch_size", help="train_batch_size", default=256)
    parser.add_argument("-test_b", "--test_batch_size", help="test_batch_size", default=256)

    parser.add_argument("-epoch1", "--fine_tuning_epoch1", help="fine_tuning_epoch1", default=1)
    parser.add_argument("-test_epoch", "--test_epoch", help="test_epoch", default=1)

    parser.add_argument("-fine_pics_num", "--fine_tuning_pics_num", help="fine_tuning_pics_num", default=20)
    parser.add_argument("-fine_batch_size", "--fine_tuning_batch_size", help="fine_tuning_batch_size", default=512)

    parser.add_argument("-test_time", "--test_time", help="if test_time", default=False, type=ast.literal_eval)
    parser.add_argument("-test_latency", "--test_latency", help="if test_latency", default=False, type=ast.literal_eval)

    parser.add_argument("-test_range_n", "--test_range_n", help="test range(n)", default=5)
    parser.add_argument("-test_range_n2", "--test_range_n2", help="test range(n) for ", default=5)

    parser.add_argument("-record_batch", "--record_batch", help="record_batch", default=512)

    parser.add_argument("-mask_batch", "--mask_batch", help="mask_batch", default=512)

    args = parser.parse_args()

    dataSet_name = args.dataSet_name

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = Data_Loader_CIFAR(train_batch_size=int(args.train_batch_size),
                                    test_batch_size=int(args.test_batch_size),
                                    dataSet=dataSet_name,
                                    use_data_Augmentation=False,
                                    download=False, train_shuffle=True)


    if dataSet_name == "CIFAR10":
        model = torch.load(f='resnet56_before.pkl').to(device)
    elif dataSet_name == "CIFAR100":
        model = torch.load(f='resnet56_cifar100_before.pkl')

    model.eval()


    test_range_n = int(args.test_range_n)
    test_range_n2 = int(args.test_range_n2)

    test_time = args.test_time
    test_latency = args.test_latency

    model_id = 0  # 保存模型的id

    if test_time:
        fine_tuning_epoch1 = int(args.fine_tuning_epoch1)
    else:
        fine_tuning_epoch = 50

    manual_radio = float(args.manual_radio)
    jump_layers = 3

    fine_tuning_lr = 0.001
    fine_tuning_batch_size = int(args.fine_tuning_batch_size)
    fine_tuning_pics_num = int(args.fine_tuning_pics_num)

    use_KL_divergence = True
    divide_radio = 2
    redundancy_num = 100

    record_imgs_num = 512
    record_batch = int(args.record_batch)

    frozen = False

    msg_save_path = "msg.txt"
    model_save_path = 'model'

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
            if isinstance(module, nn.AvgPool2d):
                hook = module.register_forward_hook(forward_activation_hook)

        # Simulated actual forward reasoning (before pruning) and recorded data
        model.eval()
        with torch.no_grad():
            for forward_x, _ in record_dataloader:
                forward_x = forward_x.to(device)
                _ = model(forward_x)

        hook.remove()

        print("step 1 finished")

        """
        Test different activation functions
        """
        # activations_list = [None, nn.ReLU(), nn.LeakyReLU()]
        # for i in range(0, 101, 5):
        #     # print(i / 100.0)
        #     activations_list.append(nn.LeakyReLU(negative_slope=i/100.0))
        # activations_list.extend([nn.Sigmoid(), nn.Tanh(), nn.ELU(), nn.Hardswish()])
        #
        # activations_list = []
        # negative_slope_list = [0.23, 0.27, 0.29, 0.32, 0.34, 0.36, 0.38, 0.39, 0.41, 0.42, 0.43, 0.44, 0.46, 0.48]
        # for negative_slope in negative_slope_list:
        #     activations_list.append(nn.LeakyReLU(negative_slope=negative_slope))


        # ----------The second part: Obtain the reserved classes according to the forward inference picture,
        # calculate the mask, prune and obtain the new model after pruning
        for act_func in [nn.LeakyReLU(negative_slope=0.4)]:

            for i in range(test_range_n):

                if test_time:
                    if i == 0:
                        fine_tuning_epoch = int(args.test_epoch)
                    else:
                        fine_tuning_epoch = fine_tuning_epoch1

                    if i != 0:
                        torch.cuda.synchronize()
                        time_in_pr = time.time()

                # Select a portion of the recorded data to calculate the mask
                imgs_tensor = choose_mask_imgs2(target_class=reserved_classes,
                                                data_loader=record_dataloader,
                                                pics_num=20, batch_size=int(args.mask_batch),
                                                shuffle=True)

                layer_masks, score_num_list = Compute_layer_mask(imgs_dataloader=imgs_tensor, model=model,
                                                                 percent=manual_radio, device=device,
                                                                 activation_func=act_func)

                # Pre-prune and calculate the mask
                cfg, cfg_masks, filter_remove_radio = pre_processing_Pruning(model, layer_masks,
                                                                             jump_layers=jump_layers)
                filter_remain_radio = 1 - filter_remove_radio

                # Generate the model according to cfg
                new_model = resnet56(data_loader.dataset_num_class, cfg=cfg).to(device)

                # Formal pruning, parameter copy
                model_after_pruning = Real_Pruning(old_model=model, new_model=new_model,
                                                   cfg_masks=cfg_masks, reserved_class=reserved_classes)
                if test_time:
                    if i != 0:
                        torch.cuda.synchronize()
                        time_out_pr = time.time()
                        time_prune = time_out_pr - time_in_pr
                        print("pruning time：" + str(time_prune) + "s")

                # Multi GPUs
                if torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    model_after_pruning = nn.DataParallel(model_after_pruning)
                model_after_pruning.to(device)

                # Calculate a variety of compression standards
                model.eval()
                model_after_pruning.eval()
                old_FLOPs, old_parameters = cal_FLOPs_and_Parameters(model, device)
                new_FLOPs, new_parameters = cal_FLOPs_and_Parameters(model_after_pruning, device)
                FLOPs_radio = new_FLOPs / old_FLOPs
                Parameters_radio = new_parameters / old_parameters


                print("model_id:" + str(model_id)
                      + " ---filter_remain_radio:" + str(filter_remain_radio)
                      + " ---FLOPs_radio:" + str(FLOPs_radio)
                      + " ---Parameters_radio:" + str(Parameters_radio))

                if test_time:
                    if i != 0:
                        torch.cuda.synchronize()
                        time_in_pr = time.time()

                for iii in range(test_range_n2):
                    # ----------Step 3: Fine-tune a portion of the picture
                    # from the forward inference record using the algorithm
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
                    if test_latency:
                        model.eval()
                        model_after_pruning.eval()
                        input_size = (100, 3, 32, 32)
                        latency_new = compute_latency_ms_pytorch(model_after_pruning, input_size, iterations=10,
                                                                 device='cuda')
                        latency_old = compute_latency_ms_pytorch(model, input_size, iterations=10, device='cuda')
                        latency_old = compute_latency_ms_pytorch(model, input_size, iterations=100, device='cuda')
                        latency_new = compute_latency_ms_pytorch(model_after_pruning, input_size, iterations=100,
                                                                 device='cuda')
                        print('model:{}, | latency: {}'.format('new', latency_new))
                        print('model:{}, | latency: {}'.format('old', latency_old))

                    print("model_id:---" + str(model_id) +
                          " best_acc:----" + str(best_acc) +
                          " reserved_classes:---" + str(reserved_classes) +
                          " manual_radio:---" + str(manual_radio) +
                          " filter_remain_radio:---" + str(filter_remain_radio) +
                          '\n')

                    model_id += 1

