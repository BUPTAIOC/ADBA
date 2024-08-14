# coding:utf-8
import DataTools
import argparse
import torch
import numpy as np
import copy
import csv
import random
import sys
from datetime import datetime
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import fsolve
from functools import partial
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
from dataset import load_mnist_test_data, load_cifar10_test_data, load_imagenet_test_data
from general_torch_model import GeneralTorchModel

from arch import mnist_model
from arch import cifar_model

import line_profiler as lp

import statistics


##################################################################################################
#represent blocks of a picture
class Block:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.width = x2 - x1 + 1

    def cut_block(self, subnum=2):
        bs = []
        for i in range(subnum):
            bs.append(copy.deepcopy(self))
 
        for i in range(subnum):
            line1 = self.x1 + (i * (self.x2 - self.x1)) // subnum #d1
            line2 = self.x1 + ((i + 1) * (self.x2 - self.x1)) // subnum #d2
            bs[i].x1 = line1
            if i > 0:
                bs[i].x1 = line1 + 1
            bs[i].x2 = line2
            bs[i].width = bs[i].x2 - bs[i].x1 + 1
        return bs

#represent a perturbation direction
class V:
    def __init__(self, size_channel, size_x, size_y, v):
        self.size_channel = size_channel
        self.size_x = size_x
        self.size_y = size_y
        self.pixnum = size_x * size_y * size_channel
        self.adv_v = [v for _ in range(self.pixnum)]
        self.score = 1.0
        self.Rmax = 1.0
        self.Rmin = 0.0

        list_temp = [-1, 1]
        if v == 0:
            for x in range(len(self.adv_v)):
                self.adv_v[x] = random.choice(list_temp)
    #reverse blocks of a perturbation direction to generate new directions
    def reverse_v(self, block):
        for x in range(block.x1, block.x2 + 1):
            self.adv_v[x] *= -1

    def advv_to_tensor(self):
        # initialize 
        three_d_list = [
            [[self.adv_v[channel * self.size_x * self.size_y + x * self.size_y + y]
              for y in range(self.size_y)]
             for x in range(self.size_x)]
            for channel in range(self.size_channel)]
        aim_np = np.array(three_d_list)
        perturbation = torch.tensor(aim_np)
        return perturbation

#represent Iterations
class Iter:
    def __init__(self, init_vbest, offspringN, iter_n=1):
        self.offspringN = offspringN
        self.iter_n = iter_n
        self.offspringVs = [] #d1 d2
        self.chosen_v = -1 #chose witch d
        self.old_vbest = copy.deepcopy(init_vbest) # dbest
        for i in range(offspringN):
            self.offspringVs.append(copy.deepcopy(init_vbest))
            self.offspringVs[i].Rmax, self.offspringVs[i].Rmin = 1.0, 0.0
    #create a new generation
    def mutation(self, model, original_image, label, aim_r, tolerance_binary_iters, blocks, binaryM):
        query = 0
        for vi in range(self.offspringN):
            self.offspringVs[vi].reverse_v(blocks[vi])
            self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = self.old_vbest.Rmax, 0.0

        query = query + self.compare_directions_usingADB(
            model, original_image, label, aim_r, tolerance_binary_iters, binaryM)

        for vi in range(self.offspringN): #initialize directions
            self.offspringVs[vi].reverse_v(blocks[vi])
        if self.chosen_v >= 0:
            self.old_vbest.reverse_v(blocks[self.chosen_v])
            self.old_vbest.Rmax, self.old_vbest.Rmin = (
                self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
            for vi in range(self.offspringN):
                self.offspringVs[vi].reverse_v(blocks[self.chosen_v])
                self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = (
                    self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
        self.iter_n = self.iter_n + 1
        return query
    #algorithm 2: Compare Directions Using ADB
    def compare_directions_usingADB(self, model, original_image, label, aim_r, maxIters, binaryM):
        perturbations = [] #d1 d2
        perturbed_images = [] #= x+ADB*d
        predicted = [] #= F(x+ADB*d)
        query = 0
        succV = []
        self.chosen_v = -1
        #initialize directions
        for i in range(len(self.offspringVs)):
            perturbations.append(self.offspringVs[i].advv_to_tensor().cuda())
            perturbed_images.append(torch.clamp(original_image.cuda() +
                                                self.old_vbest.Rmax * perturbations[i], 0.0, 1.0))
            predicted.append(model.predict_label(perturbed_images[i]).cpu())
            query = query + 1
            if predicted[i] != label:
                succV.append(i)
                self.offspringVs[i].Rmax = self.old_vbest.Rmax
                self.chosen_v = i
            else:
                self.offspringVs[i].Rmin = self.old_vbest.Rmax

        if len(succV) == 0:
            self.chosen_v = -1
            return query
        elif len(succV) == 1:
            self.chosen_v = succV[0]
            return query

        low, high = 0, self.old_vbest.Rmax
        for ite in range(0, maxIters): #conmaration loop
            ADB = DataTools.next_ADB(low, high, aim_r, self.old_vbest.Rmax, binaryM)#guess next ADB using rho(r)
            succVtemp = copy.deepcopy(succV)
            vi = 0
            while vi < len(succVtemp):
                perturbed_images[succVtemp[vi]] = torch.clamp(
                    original_image.cuda() + ADB * perturbations[succVtemp[vi]], 0.0, 1.0)
                predicted[succVtemp[vi]] = model.predict_label(perturbed_images[succVtemp[vi]]).cpu()
                query = query + 1
                if predicted[succVtemp[vi]] != label:
                    self.offspringVs[succVtemp[vi]].Rmax = ADB
                    self.chosen_v = succVtemp[vi]
                    if self.offspringVs[succVtemp[vi]].Rmax <= aim_r:
                        self.chosen_v = succVtemp[vi]
                        return query
                    vi = vi + 1
                else:
                    self.offspringVs[succVtemp[vi]].Rmin = ADB
                    succVtemp.pop(vi)

            if len(succVtemp) == 0:
                low = ADB
            elif len(succVtemp) == 1:
                self.chosen_v = succVtemp[0]
                return query
            elif len(succVtemp) >= 2:
                high = ADB
                succV = succVtemp

            if ite >= 4 and high - low <= 0.0002:
                break

        #d1 and d2 are close, just return d1
        self.chosen_v = succV[0]
        return query

#show the results dynamically
def progress_bar(imgi, query, iter, Rnow):
    sys.stdout.write(f'\rImg{imgi} Query{query :.0f}\t Iter{iter :.0f}\t Rinf{Rnow:.4f}')
    sys.stdout.flush()

#Algorithm 1: Approximate Decision Boundary Approach
def ATK_ADBA(model, original_image_x, img_number, label_y, sample_index, aim_r, tolerance_binary_iters, args):
    channels, size_x, size_y = original_image_x.shape[1], original_image_x.shape[2], original_image_x.shape[3]
    if args.channels == 1:
        channels = args.channels
    pix_num = channels * size_x * size_y
    v0 = V(channels, size_x, size_y, args.initDir)

    iter_num = 1
    block_iter = 0
    b0 = Block(0, pix_num - 1)
    bs1 = b0.cut_block(args.offspringN)
    blocks = [bs1]

    query = 0
    Rline = [[0, 1.0]]
    ITERATION = Iter(v0, args.offspringN, 1)
    query = query + ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters, blocks[0],
                                       args.binaryM)
    progress_bar(img_number, query, iter_num, ITERATION.old_vbest.Rmax)
    Rline.append([query, ITERATION.old_vbest.Rmax])
    """"""

    while (query < args.budget) and (ITERATION.old_vbest.Rmax > aim_r):  #main ATK loop
        block_iter = block_iter + 1
        blocks_i = []
        for i, bi in enumerate(blocks[block_iter - 1]):
            blocks_i.extend(bi.cut_block(args.offspringN))
            query_plus = ITERATION.mutation(model, original_image_x, label_y, aim_r, tolerance_binary_iters,
                                            blocks_i[args.offspringN * i:args.offspringN * (i + 1)], args.binaryM)
            query = query + query_plus
            progress_bar(img_number, query, iter_num, ITERATION.old_vbest.Rmax)
            Rline.append([query, ITERATION.old_vbest.Rmax])
            iter_num = iter_num + 1
            if (ITERATION.old_vbest.Rmax <= aim_r) or query >= args.budget:
                break
        blocks.append(copy.deepcopy(blocks_i))

    Rbest = ITERATION.old_vbest.Rmax
    adversarial_v = ITERATION.old_vbest.advv_to_tensor()
    adversarial_image = original_image_x + Rbest * adversarial_v
    adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)

    success = 1
    if Rbest > aim_r:
        success = -1
    # nparray = Rbest*np.array(iter_now.vbest.adv_v).flatten()
    adv_img = adversarial_image - original_image_x

    if query >= 1:
        #save and output images and atk images
        Filestring = ("Img"+str(sample_index)+
                     "_Que"+str(query)+
                      "_Time"+str(datetime.now().strftime("%H-%M-%S"))
                      )
        combined_file = DataTools.save_images(original_image_x,
                                              adversarial_image,
                                              0.5 * ITERATION.old_vbest.Rmax * (1 + adversarial_v),
                                              Filestring)
    nparray = np.array(adv_img.cpu()).flatten()
    return success, query, ITERATION.iter_n, Rbest, np.linalg.norm(nparray, ord=2), np.mean(
        nparray), Rline  # np.linalg.norm(nparray,ord=np.inf)


def RlineQ(Rline, radius_line, budget):
    start = 0
    for t in range(len(Rline) - 1):
        for q in range(start, min(Rline[t + 1][0], budget)):
            radius_line[q] = radius_line[q] + Rline[t][1]
            start = Rline[t + 1][0]
    return


def main_ADBA():
    profile = lp.LineProfiler()
    # ###################################################################################
    torch_model, test_loader = None, None
    parser = argparse.ArgumentParser(description='Hard Label Attacks')
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='Dataset')
    parser.add_argument('--epsilon', default=0.3, type=float,
                        help='attack strength')
    parser.add_argument('--imgnum', default=1000, type=int,
                        help='Number of samples to be attacked from test dataset.')
    parser.add_argument('--beginIMG', default=0, type=int,
                        help='begin test img number')
    parser.add_argument('--budget', default=10000, type=int,
                        help='Maximum queries for the attack')
    parser.add_argument('--binaryM', default=1, type=int,
                        help='binary search mod, mid 0 or median 1.')
    parser.add_argument('--initDir', default=1, type=int,
                        help='initial direction, 1,-1,and 0 for random')
    parser.add_argument('--channels', default=3, type=int,
                        help='output channels, 3 for max channels, 1 for 1 channel for all datas')
    parser.add_argument('--offspringN', default=2, type=int,
                        help='offspring diretion num in new iteration')
    parser.add_argument('--targeted', default=0, type=int,
                        help='targeted or untargeted')
    parser.add_argument('--norm', default='linf', type=str,
                        help='Norm for attack, linf only')
    parser.add_argument('--batch', default=1, type=int,
                        help='attack batch size.')
    parser.add_argument('--early', default='1', type=str,
                        help='early stopping (stop attack once the adversarial example is found)')

    args = parser.parse_args()
    targeted = True if args.targeted == '1' else False
    early_stopping = False if args.early == '0' else True
    order = 2 if args.norm == 'l2' else np.inf
    print(args)
    result_file_name = (args.dataset + "_budget" + str(args.budget) +
                        "_Early" + str(args.early) +
                        "_R" + str(args.epsilon) +
                        "_BinM" + str(args.binaryM) +
                        "_IMG" + str(args.imgnum) + ".csv")
    if args.dataset == 'mnist':
        model = mnist_model.MNIST().cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load('model/mnist_gpu.pt'))
        test_loader = load_mnist_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'cifar':
        model = cifar_model.CIFAR10().cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        model.load_state_dict(torch.load('model/cifar10_gpu.pt'))
        test_loader = load_cifar10_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=10, im_mean=None, im_std=None)
    elif args.dataset == 'vgg':
        model = models.__dict__["vgg19"]().cuda()
        #weight = models.vgg19(models.VGG19_Weights.DEFAULT)
        weight = torch.load("model/vgg19-dcbb9e9d.pth")
        model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif args.dataset == 'resnet50':
        model = models.__dict__["resnet50"]().cuda()
        weight = torch.load("model/resnet50-11ad3fa6.pth")
        model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif args.dataset == 'inception_v3':
        model = models.__dict__["inception_v3"]().cuda()
        # weight = models.inception_v3(models.Inception_V3_Weights.DEFAULT)
        weight = torch.load("model/inception_v3_google-0cc3c7bd.pth")
        model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif args.dataset == 'vit':
        model = models.__dict__["vit_b_32"]().cuda()
        # weight = models.vit_b_32(models.ViT_B_32_Weights.DEFAULT)
        weight = torch.load("model/vit_b_32-d86f8d99.pth")
        model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif args.dataset == 'efficient':
        model = models.__dict__["efficientnet_b0"]().cuda()
        # weight = torchvision.models.efficientnet_b0(models.EfficientNet_B0_Weights.DEFAULT)
        weight = torch.load("model/efficientnet_b0_rwightman-7f5810bc.pth")
        model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    elif args.dataset == 'densenet':
        model = models.__dict__["densenet161"](pretrained=True).cuda()
        # weight = torchvision.models.densenet161(models.DenseNet161_Weights.DEFAULT)
        # weight = torch.load("E:/PycharmCodes/BlackBoxImgAtk/code/model/densenet161-8d451a50.pth")
        # model.load_state_dict(weight)
        model = torch.nn.DataParallel(model, device_ids=[0])
        test_loader = load_imagenet_test_data(args.batch)
        torch_model = GeneralTorchModel(model, n_class=1000, im_mean=[0.485, 0.456, 0.406],
                                        im_std=[0.229, 0.224, 0.225])
    else:
        print("Invalid dataset")
        exit(1)
    # ###############################################################################
    orig_correct_picture_num = 0
    atk_success = 0  # total number of success attack samples
    atk_success_rate = 0
    tot_queries = 0
    avg_quer = 0
    mid_quer = 0
    avg_iter = 0
    QperI = 0
    tot_iters = 0
    stop_query = []
    radius_line = [0 for i in range(args.budget + 1)]
    with open("results_record/RES" + result_file_name, mode='w', encoding='utf-8', newline='') as file: #save result to .csv
        writer = csv.writer(file)
        writer.writerow(["img_i", "succ", "R", "que", "iter_num", "succ_r", "AVG_q", "MID_q", "AVG_iter", "QperI"])
    for i, (xi, yi) in enumerate(test_loader):
        picture_i = i
        original_image, label = xi, yi  # test_dataset[picture_i]
        xi, yi = xi.cuda(), yi.cuda()
        if orig_correct_picture_num >= args.imgnum:
            break
        if i < args.beginIMG:
            continue

        if torch_model.predict_label(xi) == yi:
            orig_correct_picture_num = orig_correct_picture_num + 1
            success, que, iter_num, R, R2, avgval, Rline = ATK_ADBA(torch_model, original_image, i,
                                                                   label, picture_i, args.epsilon, 8, args)
            RlineQ(Rline, radius_line, args.budget - 1)
            if success == 1 and que <= args.budget:
                atk_success = atk_success + 1
                tot_queries = tot_queries + que
                tot_iters = tot_iters + iter_num
            atk_success_rate = atk_success / orig_correct_picture_num
            if atk_success == 0:
                avg_quer, mid_quer, avg_iter = 0.0, 0.0, 0.0
            else:
                avg_quer = tot_queries / atk_success
                stop_query.append(que)
                mid_quer = statistics.median(stop_query)
                avg_iter = tot_iters / atk_success
                QperI = avg_quer / max(avg_iter, 1)


            with open("results_record/RES" + result_file_name, mode='a', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [picture_i, success, R, que, iter_num, atk_success_rate, avg_quer, mid_quer, avg_iter, QperI])
            print(  # f",\tIMG_{picture_i}"
                # f",\tSUCC={success}"
                # f",\tRinf={round(R, 3)}"
                # f",\tR2={round(R2,3)}"
                # f",\tQue={que}"
                # f",\tIter={iter_num}"
                f",\tACC_RATE:{round(atk_success_rate, 4)}"
                f",\tAVGquer={round(avg_quer, 3)}"
                f",\tMIDq={round(mid_quer, 1)}"
                f",\tAVGiter={round(avg_iter, 3)}"
                f",\tQperI={round(QperI, 3)}"
            )
        else:
            print(f"IMG{picture_i} originally classify wrongly")
    print(f"ORIGINAL_CLASSIFY_ACC={orig_correct_picture_num / i}")
    # save ATK acc rate line result to .csv
    with open("results_record/RES_ACCLINE" + result_file_name, encoding='utf-8', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Query_limit", "AVG_acc"])
        querys_sorted = sorted(stop_query)
        ATK_succ_num = 0
        que_sum = 0

        for budget_limit in range(1, args.budget):
            ATK_succ_num = 0
            for querys in querys_sorted:
                if querys <= budget_limit:
                    ATK_succ_num = ATK_succ_num + 1
            writer.writerow([budget_limit, ATK_succ_num / args.imgnum])

if __name__ == "__main__":
    main_ADBA()
