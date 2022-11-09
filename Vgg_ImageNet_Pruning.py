import argparse
import ast
from models.Vgg_Imgnet_Model import *
from utils.Data_loader import Data_Loader_ImgNet
from utils.Functions import *

# ---------------------------------------
activations = []
record_activations = []


def mask_activation_hook(module, input, output):
    global activations
    activations.append(output.clone().detach().cpu())
    return


def forward_activation_hook(module, input, output):
    global record_activations
    record = output.clone().detach().cpu()
    record_activations.append(record)
    return


def Compute_activation_scores(activations_, activation_func=None):
    activations_scores = []
    for activation in activations_:

        if activation_func is not None:
            activation = activation_func(activation)

        activations_scores.append(activation.norm(dim=(1, 2), p=2))

    return activations_scores


def Compute_activation_thresholds(activations_scores, percent):
    thresholds = []
    for tensor in activations_scores:
        sorted_tensor, index = torch.sort(tensor)

        total = len(sorted_tensor)
        threshold_index = int(total * percent)
        threshold = sorted_tensor[threshold_index]
        thresholds.append(threshold)

    return thresholds


def Compute_layer_mask(imgs_dataloader, model, percent, device, activation_func):
    percent = 1 - percent

    global activations

    model.eval()

    with torch.no_grad():
        imgs_masks = []
        hooks = []
        new_activations = []
        activations.clear()

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

        elif isinstance(module, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned.detach().item() / total

    return cfg, cfg_mask, pruned_ratio


def Real_Pruning(old_model: nn.Module, new_model: nn.Module, cfg_masks, reserved_class):
    old_model.eval()
    new_model.eval()
    old_modules_list = list(old_model.modules())
    new_modules_list = list(new_model.modules())

    mask_idx = 0
    linear_idx = 0

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

            # first linear
            if linear_idx == 0:
                input_mask = cfg_masks[-1].unsqueeze(1).expand(512, 49).flatten()
                new_module.weight.data = old_module.weight.data.clone()[:, input_mask]

            # last linear
            if linear_idx == 2:
                out_mask = torch.full([old_module.weight.data.size(0)], False, dtype=torch.bool)
                for ii in reserved_class:
                    out_mask[ii] = True
                new_module.weight.data = old_module.weight.data.clone()[out_mask, :]
                new_module.bias.data = old_module.bias.data.clone()[out_mask]

            linear_idx += 1

    return new_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-reserved_c_num", "--reserved_c_num", help="reserved class num", default=5)
    parser.add_argument("-manual_radio", "--manual_radio", help="manual_radio", default=0.10)

    parser.add_argument("-mask_pics_num", "--mask_pics_num", help="mask_pics_num", default=20)
    parser.add_argument("-mask_batch", "--mask_batch", help="mask_batch", default=10)

    parser.add_argument("-lr", "--learning_rate", help="learning_rate", default=0.003)

    parser.add_argument("-train_b", "--train_batch_size", help="train_batch_size", default=24)
    parser.add_argument("-test_b", "--test_batch_size", help="test_batch_size", default=8)

    parser.add_argument("-epoch", "--fine_tuning_epoch", help="fine_tuning_epoch", default=2)
    parser.add_argument("-epoch1", "--fine_tuning_epoch1", help="fine_tuning_epoch1", default=2)
    parser.add_argument("-test_epoch", "--test_epoch", help="test_epoch", default=1)

    parser.add_argument("-fine_pics_num", "--fine_tuning_pics_num", help="fine_tuning_pics_num", default=128)
    parser.add_argument("-fine_batch_size", "--fine_tuning_batch_size", help="fine_tuning_batch_size", default=8)

    parser.add_argument("-test_time", "--test_time", help="if test_time", default=False, type=ast.literal_eval)
    parser.add_argument("-test_latency", "--test_latency", help="if test_latency", default=False, type=ast.literal_eval)
    parser.add_argument("-test_range_n", "--test_range_n", help="test range(n)", default=5)

    parser.add_argument("-record_batch", "--record_batch", help="record_batch", default=64)
    parser.add_argument("-record_image", "--record_image", help="record_image", default=256)

    args = parser.parse_args()

    dataSet_name = "image_net"

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = Data_Loader_ImgNet(train_batch_size=int(args.train_batch_size),
                                     test_batch_size=int(args.test_batch_size),
                                     train_shuffle=True)

    model = vgg16_bn(pretrained=True).to(device)
    model.eval()

    test_time = args.test_time
    test_latency = args.test_latency

    model_id = 0

    if test_time:
        fine_tuning_epoch1 = int(args.fine_tuning_epoch1)
    else:
        fine_tuning_epoch = int(args.fine_tuning_epoch)

    manual_radio = float(args.manual_radio)
    jump_layers = 2

    fine_tuning_lr = float(args.learning_rate)
    fine_tuning_batch_size = int(args.fine_tuning_batch_size)
    fine_tuning_pics_num = int(args.fine_tuning_pics_num)

    use_KL_divergence = True
    divide_radio = 1
    redundancy_num = 100

    record_imgs_num = 256
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
                                              shuffle=False)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                if module.out_features == 4096:
                    hook = module.register_forward_hook(forward_activation_hook)

        # Simulated actual forward reasoning (before pruning) and recorded data
        model.eval()
        with torch.no_grad():
            for forward_x, _ in record_dataloader:
                forward_x = forward_x.to(device)
                _ = model(forward_x)
        hook.remove()

        print("step1 finished")

        for act_func in [nn.LeakyReLU(negative_slope=0.14)]:
            # ----------The second part: Obtain the reserved classes according to the forward inference picture,
            # calculate the mask, prune and obtain the new model after pruning
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
                                                           pics_num=int(args.mask_pics_num),
                                                           batch_size=int(args.mask_batch),
                                                           shuffle=True)

                layer_masks, score_num_list = Compute_layer_mask(imgs_dataloader=imgs_tensor_dataloader,
                                                                 model=model, percent=manual_radio,
                                                                 device=device,
                                                                 activation_func=act_func)

                # Pre-prune and calculate the mask
                cfg, cfg_masks, filter_remove_radio = pre_processing_Pruning(model, layer_masks,
                                                                             jump_layers=jump_layers)
                filter_remain_radio = 1 - filter_remove_radio
                print("pre-pruning finished")

                # Generate the model according to cfg
                new_model = vgg16_bn(cfg=cfg).to(device)
                # Formal pruning, parameter copy
                model_after_pruning = Real_Pruning(old_model=model, new_model=new_model,
                                                   cfg_masks=cfg_masks, reserved_class=reserved_classes)
                print("model-pruning finished")

                if test_time:
                    if i != 0:
                        torch.cuda.synchronize()
                        time_out_pr = time.time()
                        time_prune = time_out_pr - time_in_pr
                        print("pruning-time" + str(time_prune) + "s")

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

                # ----------Step 3: Fine-tune a portion of the picture from the forward
                #                   inference record using the algorithm
                # --------------------------------------------- fine-tuning
                if test_time:
                    if i != 0:
                        torch.cuda.synchronize()
                        time_in_pr = time.time()

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
                                                            frozen=frozen)
                if test_time:
                    if i != 0:
                        torch.cuda.synchronize()
                        time_out_pr = time.time()
                        time_fine_tune = time_out_pr - time_in_pr
                        print("fine-tuning time:" + str(time_fine_tune) + "s")

                # test latency
                if test_latency:
                    input_size = (16, 3, 224, 224)
                    latency_new = compute_latency_ms_pytorch(model_after_pruning, input_size, iterations=10,
                                                             device='cuda')
                    latency_old = compute_latency_ms_pytorch(model, input_size, iterations=10, device='cuda')
                    latency_new = compute_latency_ms_pytorch(model_after_pruning, input_size, iterations=100,
                                                             device='cuda')
                    latency_old = compute_latency_ms_pytorch(model, input_size, iterations=100, device='cuda')
                    print('model:{}, | latency: {}'.format('new', latency_new))
                    print('model:{}, | latency: {}'.format('old', latency_old))

                print("model_id:---" + str(model_id) +
                      " best_acc:----" + str(best_acc) +
                      " reserved_classes:---" + str(reserved_classes) +
                      " manual_radio:---" + str(manual_radio) +
                      " filter_remain_radio:---" + str(filter_remain_radio) +
                      '\n')

                model_id += 1
