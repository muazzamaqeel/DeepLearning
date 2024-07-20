import json
import os
import random
import time
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
from math import sqrt
from itertools import product as product
from tqdm import tqdm
from pprint import PrettyPrinter
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as FT
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD

# Make sure the pretrained model is saved to models directory
os.environ['TORCH_HOME'] = 'models'

# The utils we will use (see utils.py - with minor modifications of the GitHub version)
from Muazzam_and_Wesley_utils import *

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the label_map and num_classes (Task 1.4)
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
              'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
              'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', 
                   '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', 
                   '#aa6e28', '#fffac8', '#800000', '#aaffc3', 
                   '#808000', '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_folder, split, keep_difficult=False):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}
        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])
        labels = torch.LongTensor(objects['labels'])
        difficulties = torch.ByteTensor(objects['difficulties'])

        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])
        images = torch.stack(images, dim=0)
        return images, boxes, labels, difficulties

class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.load_pretrained_layers()

    def forward(self, image):
        out = F.relu(self.conv1_1(image))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)
        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.pool2(out)
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.pool3(out)
        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        conv4_3_feats = out
        out = self.pool4(out)
        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.pool5(out)
        out = F.relu(self.conv6(out))
        conv7_feats = F.relu(self.conv7(out))
        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())
        pretrained_state_dict = torchvision.models.vgg16(weights='DEFAULT').state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        for i, param in enumerate(param_names[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])
        self.load_state_dict(state_dict)
        print("\nLoaded base model.\n")

class AuxiliaryConvolutions(nn.Module):
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out))
        conv8_2_feats = out
        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        conv9_2_feats = out
        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out))
        conv10_2_feats = out
        out = F.relu(self.conv11_1(out))
        conv11_2_feats = F.relu(self.conv11_2(out))
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats

class PredictionConvolutions(nn.Module):
    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()
        self.n_classes = n_classes
        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)
        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        batch_size = conv4_3_feats.size(0)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)
        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)
        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)
        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)
        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)
        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)
        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)
        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)
        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)
        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)
        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)
        return locs, classes_scores

class SSD300(nn.Module):
    def __init__(self, n_classes):
        super(SSD300, self).__init__()
        self.n_classes = n_classes
        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        conv4_3_feats, conv7_feats = self.base(image)
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)
        return locs, classes_scores

    def create_prior_boxes(self):
        fmap_dims = {'conv4_3': 38, 'conv7': 19, 'conv8_2': 10, 'conv9_2': 5, 'conv10_2': 3, 'conv11_2': 1}
        obj_scales = {'conv4_3': 0.1, 'conv7': 0.2, 'conv8_2': 0.375, 'conv9_2': 0.55, 'conv10_2': 0.725, 'conv11_2': 0.9}
        aspect_ratios = {'conv4_3': [1., 2., 0.5], 'conv7': [1., 2., 3., 0.5, .333], 'conv8_2': [1., 2., 3., 0.5, .333], 'conv9_2': [1., 2., 3., 0.5, .333], 'conv10_2': [1., 2., 0.5], 'conv11_2': [1., 2., 0.5]}
        fmaps = list(fmap_dims.keys())
        prior_boxes = []
        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]
                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0, 1)
        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))
            image_boxes = list()
            image_labels = list()
            image_scores = list()
            max_scores, best_label = predicted_scores[i].max(dim=1)
            for c in range(1, self.n_classes):
                class_scores = predicted_scores[i][:, c]
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = decoded_locs[score_above_min_score]
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)
                for box in range(class_decoded_locs.size(0)):
                    if suppress[box] == 1:
                        continue
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    suppress[box] = 0
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                mask = suppress < 1
                image_boxes.append(class_decoded_locs[mask])
                image_scores.append(class_scores[mask])
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))
            image_boxes = torch.cat(image_boxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            n_objects = image_scores.size(0)
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]
                image_boxes = image_boxes[sort_ind][:top_k]
                image_labels = image_labels[sort_ind][:top_k]
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)
        return all_images_boxes, all_images_labels, all_images_scores

class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)

        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)
            _, prior_for_each_object = overlap.max(dim=1)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            overlap_for_each_prior[prior_for_each_object] = 1.
            label_for_each_prior = labels[i][object_for_each_prior]
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
            true_classes[i] = label_for_each_prior
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)

        positive_priors = true_classes != 0
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])
        n_positives = positive_priors.sum(dim=1)
        n_hard_negatives = self.neg_pos_ratio * n_positives
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)
        conf_loss_pos = conf_loss_all[positive_priors]
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()
        return conf_loss + self.alpha * loc_loss

def train(train_loader, model, criterion, optimizer, epoch, loss_hist):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()

    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        predicted_locs, predicted_scores = model(images)
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        optimizer.step()
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    loss_hist.append(losses.avg)
    del predicted_locs, predicted_scores, images, boxes, labels


def evaluate(test_loader, model, criterion, epoch):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    
    with torch.no_grad():
        for i, (images, boxes, labels, _) in enumerate(test_loader):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            predicted_locs, predicted_scores = model(images)
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)
            start = time.time()
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(test_loader),
                                                                      batch_time=batch_time, loss=losses))
    
    print(' * Average Loss: {loss.avg:.4f}'.format(loss=losses))
    return losses.avg

def train_for_epochs(data_folder, model_path, checkpoint, epochs, decay_lr_at, decay_lr_to, lr, momentum, weight_decay, grad_clip, train_batch_size, test_batch_size, keep_difficult, workers, save_freq=5):
    global start_epoch, epoch

    train_dataset = PascalVOCDataset(data_folder, split='train', keep_difficult=keep_difficult)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_fn, num_workers=workers, pin_memory=True)

    test_dataset = PascalVOCDataset(data_folder, split='test', keep_difficult=keep_difficult)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                             collate_fn=test_dataset.collate_fn, num_workers=workers)

    loss_hist = []
    if checkpoint is None or not os.path.isfile(checkpoint):
        start_epoch = 1
        model = SSD300(n_classes=len(label_map))
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                        lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        try:
            checkpoint = torch.load(checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            loss_hist = checkpoint['loss_hist']
            # Update optimizer's learning rate if continuing from checkpoint
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (decay_lr_to ** (sum(start_epoch >= e for e in decay_lr_at)))
        except (EOFError, pickle.UnpicklingError):
            print("Checkpoint file is corrupted or incomplete. Starting from scratch.")
            start_epoch = 1
            model = SSD300(n_classes=len(label_map))
            biases = list()
            not_biases = list()
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    if param_name.endswith('.bias'):
                        biases.append(param)
                    else:
                        not_biases.append(param)
            optimizer = SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                            lr=lr, momentum=momentum, weight_decay=weight_decay)

    model = model.to(device)
    criterion = MultiBoxLoss(model.priors_cxcy).to(device)

    for epoch in range(start_epoch, epochs + 1):
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)
        train(train_loader, model, criterion, optimizer, epoch, loss_hist)
        if epoch % save_freq == 0 or epoch == epochs:
            save_checkpoint(epoch, model, optimizer, loss_hist)
        evaluate(test_loader, model, criterion, epoch)

    # Always save the final model
    save_checkpoint(epoch, model, optimizer, loss_hist)

    return loss_hist


def adjust_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.")
    print("The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

def save_checkpoint(epoch, model, optimizer, loss_hist):
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer, 'loss_hist': loss_hist}
    filename = f'checkpoint_ssd300_epoch_{epoch}.pth.tar'
    torch.save(state, filename)


def plot_loss_history(loss_hist):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_hist) + 1), loss_hist, marker='o')
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def draw(image, boxes, labels, scores=None):
    draw = ImageDraw.Draw(image)
    for i in range(boxes.size(0)):
        box_location = boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[rev_label_map[labels[i].item()]])
        if scores is not None:
            text = f'{rev_label_map[labels[i].item()]}: {scores[i]:.2f}'
        else:
            text = rev_label_map[labels[i].item()]
        text_size = font.getsize(text)
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[rev_label_map[labels[i].item()]])
        draw.text(xy=text_location, text=text, fill='white', font=font)
    return image


def visualize_predictions_at_epochs(image_path, model_paths, ground_truth_boxes, ground_truth_labels, min_score=0.01, max_overlap=0.45, top_k=200):
    image = Image.open(image_path, mode='r')
    image = image.convert('RGB')

    plt.figure(figsize=(20, 10))

    # Plot ground truth
    plt.subplot(2, len(model_paths) + 1, 1)
    plt.imshow(draw(image.copy(), ground_truth_boxes, ground_truth_labels))
    plt.title('Ground Truth')

    # Plot predictions for each epoch
    for i, model_path in enumerate(model_paths):
        checkpoint = torch.load(model_path, map_location=device)
        model = checkpoint['model']
        model = model.to(device)
        model.eval()

        image_tensor = FT.to_tensor(image).unsqueeze(0).to(device)
        predicted_locs, predicted_scores = model(image_tensor)
        
        n_priors = model.priors_cxcy.size(0)
        if predicted_locs.size(1) != n_priors or predicted_scores.size(1) != n_priors:
            print(f"Epoch {i+1}: n_priors={n_priors}, loc_size={predicted_locs.size(1)}, score_size={predicted_scores.size(1)}")
            print(f"Dimension mismatch at epoch {i+1}. Skipping this epoch.")
            continue

        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score, max_overlap, top_k)
        det_boxes = det_boxes[0].to('cpu')
        det_labels = det_labels[0].to('cpu')
        det_scores = det_scores[0].to('cpu')

        plt.subplot(2, len(model_paths) + 1, i + 2)
        plt.imshow(draw(image.copy(), det_boxes, det_labels, det_scores))
        plt.title(f'Predictions Epoch {i + 1}')

    plt.show()


def main():
    data_folder = 'data/VOC'
    model_path = 'models'
    checkpoint = None  # Replace with path to checkpoint if resuming training

    epochs = 2
    decay_lr_at = [160, 180]
    decay_lr_to = 0.1
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    grad_clip = None
    train_batch_size = 32
    test_batch_size = 32
    keep_difficult = True
    workers = 4
    print_freq = 200  # Define print frequency

    loss_hist = train_for_epochs(data_folder, model_path, checkpoint, epochs, decay_lr_at, decay_lr_to, lr, momentum, weight_decay, grad_clip, train_batch_size, test_batch_size, keep_difficult, workers)

    # Plot the loss history
    plot_loss_history(loss_hist)

    # Visualization Task 2.3
    image_path = 'data/VOC/test_image.jpg'  # Replace with an actual test image path
    with open(os.path.join(data_folder, 'TEST_images.json'), 'r') as j:
        test_images = json.load(j)
    with open(os.path.join(data_folder, 'TEST_objects.json'), 'r') as j:
        test_objects = json.load(j)

    test_image_idx = 0  # Select an image index for visualization
    image_path = test_images[test_image_idx]
    ground_truth_boxes = torch.FloatTensor(test_objects[test_image_idx]['boxes'])
    ground_truth_labels = torch.LongTensor(test_objects[test_image_idx]['labels'])

    model_paths = [
        'checkpoint_ssd300_epoch_1.pth.tar',
        'checkpoint_ssd300_epoch_2.pth.tar'
    ]

    visualize_predictions_at_epochs(image_path, model_paths, ground_truth_boxes, ground_truth_labels)


if __name__ == '__main__':
    main()
