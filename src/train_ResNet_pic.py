# coding=UTF-8
import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
import torch.utils.data as util_data
import lr_schedule
from data_list import ImageList
from torch.autograd import Variable
import random

optim_dict = {"SGD": optim.SGD}

try:
    from sklearn.manifold import TSNE; HAS_SK = True
except:
    HAS_SK = False; print('Please install sklearn for layer visualization')
import matplotlib.pyplot as plt
import random

optim_dict = {"SGD": optim.SGD}

def plot_with_labels(lowDWeights, labels, i):#三个绘图函数
    plt.cla()
    X, Y = lowDWeights[:, 0], labels[:, 0]
    # for x, y, s in zip(X, Y, labels):
        # c = cm.rainbow(int(255 * s / 30))
    plt.scatter(X, Y, marker="o", c="blue", label="target", s=20)
    # plt.scatter(X, Y, marker="x", c="red", label="source", s=30)
        # plt.text(x, y, int(s), color=plt.cm.bwr(int( s / 1.)), fontdict={'weight': 'bold', 'size': 9})#backgroundcolor=c) when use the bc,ignore c
    plt.xlim(X.min() - 5, X.max() + 5);
    plt.ylim(Y.min() - 5, Y.max() + 5);
    title = plt.title('iter: {:05d}'.format(i));
    plt.legend(loc='upper right')
    plt.show();
    plt.pause(0.01)
    tsne_path = './' + 'iter: {:05d}'.format(i) + '.png'
    plt.savefig(tsne_path, dpi=600)

def plot_with_labels1(lowDWeights, labels, i):
    plt.cla()
    X, Y = lowDWeights[:, 0], labels[:, 0]
    # for x, y, s in zip(X, Y, labels):
        # c = cm.rainbow(int(255 * s / 30))
    # if((labels==1)|(labels==3)|(labels==8)|(labels==9)|(labels==13)|(labels==15)|(labels==18)|(labels==20)|(labels==24)|(labels==26)):
    print(labels[0])
    for x,y,p in zip(X,Y,range(len(labels))):
        if(labels[p]==1):
            plt.scatter(x, y, marker="o", c="blue", s=20)
        else:
            plt.scatter(x, y, marker="o", c="red", s=20)
    # plt.scatter(X, Y, marker="x", c="red", label="source", s=30)
        # plt.text(x, y, int(s), color=plt.cm.bwr(int( s / 1.)), fontdict={'weight': 'bold', 'size': 9})#backgroundcolor=c) when use the bc,ignore c
    plt.xlim(X.min() - 5, X.max() + 5);
    plt.ylim(Y.min() - 5, Y.max() + 5);
    title = plt.title('iter: {:05d}'.format(i));
    plt.legend(loc='upper right')
    plt.show();
    plt.pause(0.01)
    tsne_path = './' + 'iter: {:05d}'.format(i) + '.png'
    plt.savefig(tsne_path, dpi=600)

def plot_with_labels2(lowDWeights, labels,lowDWeights1, labels1, i):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    X1, Y1 = lowDWeights1[:, 0], lowDWeights1[:, 1]
    # for x, y, s in zip(X, Y, labels):
        # c = cm.rainbow(int(255 * s / 30))
    plt.scatter(X, Y, marker="o", c="red", s=5)
    plt.scatter(X1, Y1, marker="o", c="blue", s=5)
        # plt.text(x, y, int(s), color=plt.cm.bwr(int( s / 1.)), fontdict={'weight': 'bold', 'size': 9})#backgroundcolor=c) when use the bc,ignore c
    plt.xlim(min(X.min(),X1.min()) - 5, max(X.max(),X1.max()) + 5);
    plt.ylim(min(Y.min(),Y1.min()) - 5, max(Y.max(),Y1.max()) + 5);
    title = plt.title('iter: {:05d}'.format(i));
    # plt.legend(loc='upper right')
    plt.show();
    plt.pause(0.01)
    tsne_path = './' + 'iter: {:05d}'.format(i) + '.png'
    plt.savefig(tsne_path, dpi=600)

def image_classification_predict(loader, model, test_10crop=True, gpu=True, softmax_param=1.0):#星载图像无监督分类
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test' + str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            for j in range(10):
                _, predict_out = model(inputs[j])
                outputs.append(nn.Softmax(dim=1)(softmax_param * predict_out))
            softmax_outputs = sum(outputs)
            if start_test:
                all_softmax_output = softmax_outputs.data.cpu().float()
                start_test = False
            else:
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
    else:
        iter_val = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_val.next()
            inputs = data[0]
            if gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            _, outputs = model(inputs)
            softmax_outputs = nn.Softmax(dim=1)(softmax_param * outputs)
            if start_test:
                all_softmax_output = softmax_outputs.data.cpu().float()
                start_test = False
            else:
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
    return all_softmax_output


def image_classification_test(loader, model, iter_num, test_10crop=True, gpu=True):#测试准确率并绘出聚类的图片
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test' + str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            outputs_1 = []
            for j in range(10):
                _, predict_out = model(inputs[j])
                outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs_1.append(_)
            outputs = sum(outputs)
            outputs_1 = sum(outputs_1)
            if start_test:
                all_output = outputs.data.float()
                all_output_1=outputs_1.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_output_1 = torch.cat((all_output_1, outputs_1.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data1 = iter_test.next()
            inputs = data1[0]
            labels = data1[1]
            if gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    start_test = True
    if test_10crop:
        iter_source = [iter(loader['source1' + str(i)]) for i in range(10)]
        for i in range(len(loader['source10'])):
            data1 = [iter_source[j].next() for j in range(10)]
            inputs1 = [data1[j][0] for j in range(10)]
            labels1 = data1[0][1]
            if gpu:
                for j in range(10):
                    inputs1[j] = Variable(inputs1[j].cuda())
                labels1 = Variable(labels1.cuda())
            else:
                for j in range(10):
                    inputs1[j] = Variable(inputs1[j])
                labels1 = Variable(labels1)
            outputs1 = []
            for j in range(10):
                _1, predict_out1 = model(inputs1[j])
                outputs1.append(_1)
            outputs1 = sum(outputs1)
            if start_test:
                all_output1 = outputs1.data.float()
                all_label1 = labels1.data.float()
                start_test = False
            else:
                all_output1 = torch.cat((all_output1, outputs1.data.float()), 0)
                all_label1 = torch.cat((all_label1, labels1.data.float()), 0)

    if HAS_SK:
        # Visualization of trained flatten layer (T-SNE)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
        plot_only = 500
        print(all_output_1.size(),all_output1.size())
        all_output_1 = torch.cat((all_output_1,all_output1),0)
        all_output2 = Variable(all_output_1.cpu()).data.numpy()
        # all_output3 = Variable(all_output1.cpu()).data.numpy()
        low_dim_embs = tsne.fit_transform(all_output2[:, :])
        low_dim_embs1 = low_dim_embs[295:,:]
        del all_output2
        low_dim_embs = low_dim_embs[:295,:]
        # del all_output3
        labels = all_label.cpu().numpy()[:]
        labels1 = all_label1.cpu().numpy()[:]
    # plot_with_labels(low_dim_embs, labels, iter_num)
    plot_with_labels2(low_dim_embs1, labels1,low_dim_embs,labels, iter_num)
    del low_dim_embs1, labels1,low_dim_embs,labels
    _, predict = torch.max(all_output, 1)
    accuracy = float(torch.sum(torch.squeeze(predict).float() == all_label)) / float(all_label.size()[0])
    return accuracy



def train(config):
    tie=1.0
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train( \
        resize_size=prep_config["resize_size"], \
        crop_size=prep_config["crop_size"])
    prep_dict["target"] = prep.image_train( \
        resize_size=prep_config["resize_size"], \
        crop_size=prep_config["crop_size"])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop( \
            resize_size=prep_config["resize_size"], \
            crop_size=prep_config["crop_size"])
    else:
        prep_dict["test"] = prep.image_test( \
            resize_size=prep_config["resize_size"], \
            crop_size=prep_config["crop_size"])

    ## set loss
    class_criterion = nn.CrossEntropyLoss()  # 分类损失
    transfer_criterion1 = loss.PADA  # 迁移学习损失(带权重）,BCEloss
    balance_criterion = loss.balance_loss()
    loss_params = config["loss"]

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = util_data.DataLoader(dsets["source"], \
                                                  batch_size=data_config["source"]["batch_size"], \
                                                  shuffle=True, num_workers=4)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = util_data.DataLoader(dsets["target"], \
                                                  batch_size=data_config["target"]["batch_size"], \
                                                  shuffle=True, num_workers=4)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test" + str(i)] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                               transform=prep_dict["test"]["val" + str(i)])
            dset_loaders["test" + str(i)] = util_data.DataLoader(dsets["test" + str(i)], \
                                                                 batch_size=data_config["test"]["batch_size"], \
                                                                 shuffle=False, num_workers=4)

            dsets["target" + str(i)] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                                 transform=prep_dict["test"]["val" + str(i)])
            dset_loaders["target" + str(i)] = util_data.DataLoader(dsets["target" + str(i)], \
                                                                   batch_size=data_config["test"]["batch_size"], \
                                                                   shuffle=False, num_workers=4)
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
        dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                                    batch_size=data_config["test"]["batch_size"], \
                                                    shuffle=False, num_workers=4)

        dsets["target_test"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                         transform=prep_dict["test"])
        dset_loaders["target_test"] = MyDataLoader(dsets["target_test"], \
                                                   batch_size=data_config["test"]["batch_size"], \
                                                   shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()

    ## collect parameters
    if net_config["params"]["new_cls"]:
        if net_config["params"]["use_bottleneck"]:
            parameter_list = [{"params": base_network.feature_layers.parameters(), "lr": 1}, \
                              {"params": base_network.bottleneck.parameters(), "lr": 10}, \
                              {"params": base_network.fc.parameters(), "lr": 10}]
        else:
            parameter_list = [{"params": base_network.feature_layers.parameters(), "lr": 1}, \
                              {"params": base_network.fc.parameters(), "lr": 10}]
    else:
        parameter_list = [{"params": base_network.parameters(), "lr": 1}]

    ## add additional network for some methods
    class_weight = torch.from_numpy(np.array([1.0] * class_num))  # 权重初始化为class_num维向量，初始值为1
    if use_gpu:
        class_weight = class_weight.cuda()
    ad_net = network.AdversarialNetwork(base_network.output_num())  # 鉴别器设置，输入为特征提取器的维数，输出为属于共域的可能性
    nad_net = network.NAdversarialNetwork(base_network.output_num())
    gradient_reverse_layer = network.AdversarialLayer(high_value=config["high"])
    if use_gpu:
        ad_net = ad_net.cuda()
        nad_net = nad_net.cuda()
    parameter_list.append({"params": ad_net.parameters(), "lr": 10})

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, \
                                                     **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ## train
    len_train_source = len(dset_loaders["source"]) - 1
    len_train_target = len(dset_loaders["target"]) - 1
    transfer_loss_value = classifier_loss_value = total_loss_value = attention_loss_value = 0.0
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == 0:  # 测试
            base_network.train(False)  # 先对特征提取器进行训练
            temp_acc = image_classification_test(dset_loaders, \
                                                 base_network, test_10crop=prep_config["test_10crop"], \
                                                 gpu=use_gpu)  # 对第一个有监督训练器进行分类训练，提取特征
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, alpha: {:03f}, tradeoff: {:03f} ,precision: {:.5f}".format(i, loss_params["para_alpha"],
                                                                                                loss_params[
                                                                                                    "trade_off"],
                                                                                                temp_acc)
            config["out_file"].write(log_str + '\n')
            config["out_file"].flush()
            print(log_str)
        if i % config["snapshot_interval"] == 0:  # 输出
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                                                             "iter_{:05d}_model.pth.tar".format(i)))

        if i % loss_params["update_iter"] == loss_params["update_iter"] - 1:
            base_network.train(False)
            target_fc8_out = image_classification_predict(dset_loaders, base_network, softmax_param=config[
                "softmax_param"])  # 在对有监督学习进行了训练后，喂进去目标域数据，判断权重
            class_weight = torch.mean(target_fc8_out, 0)  # predict模型输出的预测向量取均值，为每一个体权重，将个体权重转化为类别权重
            class_weight = (class_weight / torch.mean(class_weight)).cuda().view(-1)  # 归一化
            class_criterion = nn.CrossEntropyLoss(weight=class_weight)  # 这里的交叉熵损失函数就是带权重的

        ## train one iter
        base_network.train(True)
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        if use_gpu:
            inputs_source, inputs_target, labels_source = \
                Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), \
                Variable(labels_source).cuda()
        else:
            inputs_source, inputs_target, labels_source = Variable(inputs_source), \
                                                          Variable(inputs_target), Variable(labels_source)

        inputs = torch.cat((inputs_source, inputs_target), dim=0)  # 这里输入后一半目标域，前一半源域
        features, outputs = base_network(inputs)  # 从有分类网络获得特征与输出，到这里是特征提取器
        softmax_out = nn.Softmax(dim=1)(outputs).detach()
        ad_net.train(True)
        nad_net.train(True)
        weight_ad = torch.zeros(inputs.size(0))
        label_numpy = labels_source.data.cpu().numpy()
        for j in range(inputs.size(0) / 2):
            weight_ad[j] = class_weight[int(label_numpy[j])]# 计算实际样例权重
        # print(label_numpy)
        weight_ad = weight_ad / torch.max(weight_ad[0:inputs.size(0) / 2])  # 权重归一化
        for j in range(inputs.size(0) / 2, inputs.size(0)):  # 前一半源域，所以权重是计算的，后一半目标域，权重全为1
            weight_ad[j] = 1.0
        classifier_loss = class_criterion(outputs.narrow(0, 0, inputs.size(0) / 2), labels_source)  # 分类损失
        total_loss = classifier_loss
        total_loss.backward()
        optimizer.step()

    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--dset', type=str, default='NWPU-RESISC45', help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../data/NWPU-RESISC45/NWPU-45.txt',
                        help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../data/UCMercedLand-share/UCMerced-19.txt',
                        help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san1',
                        help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--tradeoff', type=float, default=0.1, help="interval of two continuous output model")
    parser.add_argument('--para_alpha', type=float, default=0.2, help="interval of two continuous output model")
    parser.add_argument('--para_lamda', type=float, default=0.1, help="interval of two continuous output model")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config["softmax_param"] = 1.0
    config["high"] = 1.0
    config["num_iterations"] = 16004
    config["para"] = 0.3
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "../snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "a")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop": True, "resize_size": 256, "crop_size": 224}
    config["loss"] = {"trade_off": args.tradeoff, "update_iter": 500, "para_alpha": args.para_alpha, "update_iter": 500, "para_lamda": args.para_alpha, "update_iter": 500}
    # config["para_alpha"] = {"para_alpha": args.para_alpha, "update_iter": 500}
    if "AlexNet" in args.net:
        config["network"] = {"name": network.AlexNetFc, \
                             "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    elif "VGG" in args.net:
        config["network"] = {"name": network.VGGFc, \
                             "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    config["optimizer"] = {"type": "SGD", "optim_params": {"lr": 1.0, "momentum": 0.9, \
                                                           "weight_decay": 0.0005, "nesterov": True}, "lr_type": "inv", \
                           "lr_param": {"init_lr": 0.001, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    if config["dataset"] == "NWPU-RESISC45":
        config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 36}, \
                          "target": {"list_path": args.t_dset_path, "batch_size": 36}, \
                          "test": {"list_path": args.t_dset_path, "batch_size": 4}}
        config["optimizer"]["lr_param"]["init_lr"] = 0.001
        config["loss"]["update_iter"] = 500
        config["network"]["params"]["class_num"] = 45
    train(config)
