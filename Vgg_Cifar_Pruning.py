import argparse
import ast
from models.Vgg_cifal_Model import *
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
    record = input[0].clone().detach().cpu()
    record_activations.append(record)
    return


def Compute_activation_scores(activations_, activation_func=None):
    """
    :argument calculate scores
    :param activations_: [c,h,w]
    :param activation_func:
    :return: [c]
    """
    activations_scores = []
    for activation in activations_:

        if activation_func is not None:
            activation = activation_func(activation)

        activations_scores.append(activation.norm(dim=(1, 2), p=2))

    return activations_scores


def Compute_activation_thresholds(activations_scores, percent):
    """
    :argument Calculate the threshold value by the importance level of channel
    """

    thresholds = []
    for tensor in activations_scores:
        sorted_tensor, index = torch.sort(tensor)

        total = len(sorted_tensor)
        threshold_index = int(total * percent)
        threshold = sorted_tensor[threshold_index]

        thresholds.append(threshold)

    return thresholds


def Compute_layer_mask(imgs_dataloader, model, percent, device, activation_func):
    """
    :argument Calculate masks based on the input image
    :return: masks dim [layer_num, c]
    """

    percent = 1 - percent  # Calculate the percentage to be cut by retaining the percentage

    global activations
    #  change the model to eval here, otherwise the input data when calculating
    #  layer_mask will change the bn layer parameters, resulting in lower accuracy
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
                layer_list.append(activations[idx2*num_layers + idx1])

            combine_activation = torch.cat(layer_list, dim=0)

            new_activations.append(combine_activation)

        # ------ layer-by-layer
        masks = []
        score_num_list = []
        for layer_activations in new_activations:
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


def pre_processing_Pruning(model: nn.Module, masks, jump_layers=2):
    """
    :argument: Based on the mask entered, calculate the cfg needed to generate the new model,
               and the corresponding new layer_mask
    """
    model.eval()
    cfg = []  # New network structure parameters
    count = 0  # bn layer counting
    cfg_mask = []  # the new mask
    pruned = 0  # Count the number of channels cut
    total = 0  # Total number of channels

    for index, module in enumerate(model.modules()):

        if isinstance(module, nn.BatchNorm2d):

            mask = masks[count]
            if count < jump_layers:
                mask = mask | True

            # deal with the 0's left in the channel
            if torch.sum(mask) == 0:
                mask[0] = 1

            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())

            total += len(mask)

            pruned += len(mask) - torch.sum(mask)

            count += 1

        elif isinstance(module, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned.detach().item() / total

    return cfg, cfg_mask, pruned_ratio


def Real_Pruning(old_model: nn.Module, new_model: nn.Module, cfg_masks, reserved_class):
    """
    :argument According to cfg_mask, which is the mask of each bn layer, the parameters of the original model
              are copied to the new model, while the cs layer and linear layer of the new model are adjusted
    """

    old_model.eval()
    new_model.eval()
    old_modules_list = list(old_model.modules())
    new_modules_list = list(new_model.modules())

    mask_idx = 0  # mask id

    for idx, (old_module, new_module) in enumerate(zip(old_modules_list, new_modules_list)):

        if isinstance(old_module, nn.BatchNorm2d):
            new_module.weight.data = old_module.weight.data.clone()[cfg_masks[mask_idx]]
            new_module.bias.data = old_module.bias.data.clone()[cfg_masks[mask_idx]]
            new_module.running_mean = old_module.running_mean.clone()[cfg_masks[mask_idx]]
            new_module.running_var = old_module.running_var.clone()[cfg_masks[mask_idx]]

            mask_idx += 1

        if isinstance(old_module, nn.Conv2d):

            out_mask = cfg_masks[mask_idx]

            if mask_idx > 0:
                in_mask = cfg_masks[mask_idx - 1]
                new_weight = old_module.weight.data.clone()[:, in_mask, :, :]
                new_module.weight.data = new_weight.clone()[out_mask, :, :, :]
            else:
                new_module.weight.data = old_module.weight.data.clone()[out_mask, :, :, :]

        if isinstance(old_module, nn.Linear):

            out_mask = torch.full([old_module.weight.data.size(0)], False, dtype=torch.bool)
            for ii in reserved_class:
                out_mask[ii] = True

            fc_data = old_module.weight.data.clone()[:, cfg_masks[-1]]
            fc_data = fc_data.clone()[out_mask, :]

            new_module.weight.data = fc_data.clone()
            new_module.bias.data = old_module.bias.data.clone()[out_mask]

    return new_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-reserved_c_num", "--reserved_c_num", help="reserved class num", default=5)
    parser.add_argument("-manual_radio", "--manual_radio", help="manual_radio", default=0.05)

    parser.add_argument("-dataSet_name", "--dataSet_name", help="CIFAR10, CIFAR100", default='CIFAR10')

    parser.add_argument("-mask_pics_num", "--mask_pics_num", help="mask_pics_num", default=30)

    parser.add_argument("-train_b", "--train_batch_size", help="train_batch_size", default=512)
    parser.add_argument("-test_b", "--test_batch_size", help="test_batch_size", default=512)

    parser.add_argument("-epoch1", "--fine_tuning_epoch1", help="fine_tuning_epoch1", default=1)
    parser.add_argument("-test_epoch", "--test_epoch", help="test_epoch", default=1)

    parser.add_argument("-fine_pics_num", "--fine_tuning_pics_num", help="fine_tuning_pics_num", default=128)
    parser.add_argument("-fine_batch_size", "--fine_tuning_batch_size", help="fine_tuning_batch_size", default=512)

    parser.add_argument("-test_time", "--test_time", help="if test_time", default=False, type=ast.literal_eval)
    parser.add_argument("-test_latency", "--test_latency", help="if test_latency", default=False, type=ast.literal_eval)

    parser.add_argument("-test_range_n", "--test_range_n", help="test range(n)", default=5)
    parser.add_argument("-test_range_n2", "--test_range_n2", help="test range(n) for ", default=5)

    parser.add_argument("-record_batch", "--record_batch", help="record_batch", default=512)

    parser.add_argument("-mask_batch", "--mask_batch", help="mask_batch", default=1024)

    args = parser.parse_args()

    dataSet_name = args.dataSet_name

    mask_pics_num = int(args.mask_pics_num)

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = Data_Loader_CIFAR(train_batch_size=int(args.train_batch_size),
                                    test_batch_size=int(args.test_batch_size),
                                    dataSet=dataSet_name, use_data_Augmentation=False,
                                    download=False, train_shuffle=True)

    if dataSet_name == "CIFAR10":
        model = torch.load(f='vgg16_before.pkl').to(device)
    if dataSet_name == "CIFAR100":
        model = torch.load(f='vgg16_cifar100_before.pkl').to(device)
    model.eval()


    test_range_n = int(args.test_range_n)    #
    test_range_n2 = int(args.test_range_n2)  # randuncy num for

    test_time = args.test_time
    test_latency = args.test_latency
    redundancy_num = 100

    model_id = 0

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

    record_imgs_num = 512
    record_batch = int(args.record_batch)

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
                                              shuffle=True)

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

        print("step 1 finished")

        """
        Test different activation functions
        """
        # [None, nn.ReLU(), nn.LeakyReLU(), F.relu6, nn.Sigmoid(), nn.Tanh(), nn.ELU(), nn.Hardswish()]
        # activations_list = [None, nn.ReLU(), nn.LeakyReLU()]
        # for i in range(0, 101, 5):
        #     # print(i / 100.0)
        #     activations_list.append(nn.LeakyReLU(negative_slope=i/100.0))
        # activations_list.extend([nn.Sigmoid(), nn.Tanh(), nn.ELU(), nn.Hardswish()])
        #
        # activations_list = []
        # negative_slope_list = [0.07, 0.09, 0.12, 0.14, 0.16, 0.18, 0.21, 0.22, 0.23, 0.24, 0.26, 0.28]
        # for negative_slope in negative_slope_list:
        #     activations_list.append(nn.LeakyReLU(negative_slope=negative_slope))

        for act_func in [nn.LeakyReLU(negative_slope=0.14)]:
            # ----------The second part: Obtain the reserved classes according to the forward inference picture,
            #           calculate the mask, prune and obtain the new model after pruning
            # Select a portion of the recorded data to calculate the mask
            for i in range(int(args.test_range_n)):

                if test_time:
                    if i == 0:
                        fine_tuning_epoch = int(args.test_epoch)
                    else:
                        fine_tuning_epoch = fine_tuning_epoch1

                    if i != 0:
                        torch.cuda.synchronize()
                        time_in_pr = time.time()

                imgs_tensor_dataloader = choose_mask_imgs2(target_class=reserved_classes,
                                                           data_loader=record_dataloader,
                                                           pics_num=mask_pics_num,
                                                           batch_size=int(args.mask_batch), shuffle=True)
                layer_masks, score_num_list = Compute_layer_mask(imgs_dataloader=imgs_tensor_dataloader,
                                                                 model=model, percent=manual_radio,
                                                                 device=device,
                                                                 activation_func=act_func)

                # Pre-prune and calculate the mask
                cfg, cfg_masks, filter_remove_radio = pre_processing_Pruning(model, layer_masks, jump_layers=jump_layers)
                filter_remain_radio = 1 - filter_remove_radio

                # Generate the model according to cfg
                new_model = vgg16(data_loader.dataset_num_class, cfg=cfg).to(device)
                # Formal pruning, parameter copy
                model_after_pruning = Real_Pruning(old_model=model, new_model=new_model,
                                                   cfg_masks=cfg_masks, reserved_class=reserved_classes)

                if test_time:
                    if i != 0:
                        torch.cuda.synchronize()
                        time_out_pr = time.time()
                        time_prune = time_out_pr - time_in_pr
                        print("pruning time："+str(time_prune)+"s")

                # Calculate a variety of compression standards
                model.eval()
                model_after_pruning.eval()
                old_FLOPs, old_parameters = cal_FLOPs_and_Parameters(model, device)
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
                      + " ---Parameters_radio:" + str(Parameters_radio))

                if test_time:
                    if i != 0:
                        torch.cuda.synchronize()
                        time_in_pr = time.time()

                for iii in range(test_range_n2):

                    # ----------Step 3: Fine-tune a portion of the picture from the forward
                    # inference record using the algorithm

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
                            print("data selection time："+str(time_choose)+"s")

                        if i != 0:
                            torch.cuda.synchronize()
                            time_in_pr = time.time()

                    # FINE-TUNING
                    best_acc, acc_list, loss_list = fine_tuning(model_after_pruning, reserved_classes,
                                                                EPOCH=fine_tuning_epoch, lr=fine_tuning_lr,
                                                                device=device,
                                                                train_data_loader=fine_tuning_loader,
                                                                test_data_loader=data_loader.test_data_loader,
                                                                model_save_path=model_save_path,
                                                                use_all_data=False,
                                                                frozen=frozen)
                    if test_time:
                        if i != 0:
                            torch.cuda.synchronize()
                            time_out_pr = time.time()
                            time_fine_tune = time_out_pr - time_in_pr
                            print("fine-tuning time："+str(time_fine_tune)+"s")

                    # test latency
                    if test_latency:
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

