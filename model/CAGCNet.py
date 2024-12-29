import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm.notebook import tqdm
import os
import cv2
from collections import Counter
from skimage.segmentation.slic_superpixels import slic
from IPython import display
from torch.nn.modules.module import Module
from torch_geometric.nn import GATConv
from torch.nn.parameter import Parameter
import math
import torch
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import torch_geometric.nn as geo_nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_norm(data):
    mean = torch.mean(data)
    std = torch.std(data)
    data = (data - mean) / std
    return data


def performance(predict_labels, gt_labels, class_num):
    matrix = np.zeros((class_num, class_num))
    predict_labels = torch.max(predict_labels, dim=1)[1]

    for j in range(len(predict_labels)):
        o = predict_labels[j]
        q = gt_labels[j]
        if q == 0:
            continue
        matrix[o - 1, q - 1] += 1
    OA = np.sum(np.trace(matrix)) / np.sum(matrix)

    ac_list = np.zeros((class_num))
    for k in range(len(matrix)):
        ac_k = matrix[k, k] / sum(matrix[:, k])
        ac_list[k] = round(ac_k, 4)

    AA = np.mean(ac_list)

    mm = 0
    for l in range(matrix.shape[0]):
        mm += np.sum(matrix[l]) * np.sum(matrix[:, l])
    pe = mm / (np.sum(matrix) * np.sum(matrix))
    pa = np.trace(matrix) / np.sum(matrix)
    kappa = (pa - pe) / (1 - pe)
    return OA, AA, kappa, ac_list


path = '/kaggle/input/trento/lidar.mat'
data = scio.loadmat(path)
data = data['data']
data_ = torch.FloatTensor(data.astype(float))
data = data_norm(data_)
# data size = 325*220*2 长*宽*channel
path = '/kaggle/input/trento/hsi.mat'
data1 = scio.loadmat(path)
data1 = data1['data']
data1_ = torch.FloatTensor(data1.astype(float))
data1 = data_norm(data1_)
# data1 size = 325*220*64
path = '/kaggle/input/trento/gt.mat'
ground_truth = scio.loadmat(path)
ground_truth = ground_truth['mask_test']
ground_truth = torch.FloatTensor(ground_truth.astype(int))
"""
new_data1 = data1__.reshape(-1, 64)
new_data1 = PCA(n_components=40).fit_transform(new_data1)
new_data1 = torch.tensor(new_data1).view(40,-1)
# new_data1 size = 40*71500
node_features = new_data1 # 节点特征
similarities = torch.tensor(cosine_similarity(node_features))# 计算余弦相似度
new_data1 = new_data1.permute(1, 0)
new_data1 = torch.DoubleTensor(new_data1)
# 基于相似度阈值创建边索引
threshold = 0  # 根据相似度阈值确定是否存在边
edges = torch.where(similarities > threshold)
first_edge = edges[0]
second_edge = edges[1]
edges = torch.stack((first_edge, second_edge)).to(torch.int64)
# 共有40个节点，828条边
"""

node_features = data1.permute(2, 0, 1).reshape(63, -1)
similarities = torch.tensor(cosine_similarity(node_features))  # similarities的中位数是0.6638
new_data1 = node_features
threshold = 0.85  # 根据相似度阈值确定是否存在边
edges = torch.where(similarities > threshold)
first_edge = edges[0]
second_edge = edges[1]
edges = torch.stack((first_edge, second_edge)).to(torch.int64)
new_data1 = new_data1.permute(1, 0)
graph_data = Data(x=new_data1, edge_index=edges)

data_width = data.shape[0]
data_height = data.shape[1]
channel_num = data.shape[2]
data_width1 = data1.shape[0]
data_height1 = data1.shape[1]
channel_num1 = data1.shape[2]

plt.imshow(ground_truth)
class_num = len(set(np.array(ground_truth.reshape(-1)))) - 1  # set去重
print('The number of classes is:', class_num)

Number_class = Counter(list(np.array(ground_truth.reshape(-1))))
count = np.zeros(class_num + 1)
count[np.array(list(Number_class.keys())).astype(int)] = list(Number_class.values())
count = count[1:]
train_count = list(np.around(count * 0.10).astype(int))
# train_count = [150, 150, 150,150, 150, 150,150, 150, 150,150, 150]
# Get class's position index
classes_index = []
for i in range(class_num + 1):  # with the background
    class_index = np.argwhere(np.array(ground_truth) == i)
    np.random.shuffle(class_index)
    classes_index.append(class_index)
# classes_index中包含第0类（不代表类型，而是代表背景），需要剔除，下面用i+1的方式剔除
test_count = []
train_index = []
test_index = []
for i in range(class_num):
    train_index.append(classes_index[i + 1][:train_count[i]])
    test_index.append(classes_index[i + 1][-(len(classes_index[i + 1]) - train_count[i]):])
    test_count.append(len(classes_index[i + 1]))
    # test_count.append(len(classes_index[i + 1]) - train_count[i])

# Get train and test mask
# mask用来指示哪些坐标的pixel在训练时是有效的,用来区分训练集和测试集（一张图片某些部分是训练集，其他是测试集）
train_mask = torch.zeros(data.shape[:2])
test_mask = torch.zeros(data.shape[:2])

for i in range(class_num):
    train_mask[train_index[i][:, 0], train_index[i][:, 1]] = 1
    test_mask[test_index[i][:, 0], test_index[i][:, 1]] = 1
# train_mask是二维tensor，pixel的值为1，代表该pixel属于某一类；其他为0，代表该pixel是背景


seg_index = slic(np.array(data_), n_segments=24)
seg_index1 = slic(np.array(data1_), n_segments=24)
seg_index = torch.Tensor(seg_index.copy())
seg_index1 = torch.Tensor(seg_index1.copy())
Block_num = len(set(np.array(seg_index.reshape(-1))))
Block_num1 = len(set(np.array(seg_index1.reshape(-1))))
print('Block_num:', Block_num)
print('Block_num1:', Block_num1)
# ！！！！！！！！seg_index和seg_index1相同
# initialize adjacency matrix
adj_mask = torch.ones(Block_num, Block_num).int().to(device)
adj_mask1 = torch.ones(Block_num1, Block_num1).int().to(device)


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, 200, kernel_size=3, stride=stride, padding=1).to(device)
        self.bn1 = nn.BatchNorm2d(200).to(device)
        self.conv2 = nn.Conv2d(200, 125, kernel_size=3, stride=stride, padding=1).to(device)
        self.bn2 = nn.BatchNorm2d(125).to(device)
        self.conv3 = nn.Conv2d(125, ch_out, kernel_size=3, stride=1, padding=1).to(device)
        self.bn3 = nn.BatchNorm2d(ch_out).to(device)

        self.extra = nn.Sequential().to(device)
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h , w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride).to(device),
                nn.BatchNorm2d(ch_out).to(device)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # short cut
        # element_wise add:[b, ch_in, h, w] with [b, ch_out, h ,w]
        out = self.extra(x) + out
        return out


class Graph2dConvolution(Module):
    def __init__(self, in_channels, out_channels, block_num, adj_mask=None, if_feature_update=True,
                 for_classification=False):
        super(Graph2dConvolution, self).__init__()
        self.weight = Parameter(torch.randn(in_channels, out_channels)).to(device)
        self.W = Parameter(torch.randn(out_channels, out_channels))
        self.bn = nn.BatchNorm2d(out_channels)
        self.reset_parameters()
        self.in_features = in_channels
        self.out_features = out_channels
        self.block_num = block_num
        self.adj_mask = adj_mask
        self.if_feature_update = if_feature_update
        self.for_classification = for_classification

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, input, index):
        input = (input.permute(0, 2, 3, 1)).matmul(self.weight).permute(0, 3, 1, 2)
        """
        四维的tensor(比如说为[N, C, H, W])和二维的tensor(比如为[C, D])相乘,需要满足一定的条件:
        1. 四维tensor的第2维(通道维C)需要与二维tensor的第一维大小相同。
        2. 二维tensor的第二维(D)会成为四维tensor乘法结果中的新的第2维。
        input的尺寸是batch_size*c*w*h,经过上述操作改变了channel维度上的信息
        """

        if self.if_feature_update:
            index = nn.UpsamplingNearest2d(size=(input.shape[2], input.shape[3]))(index.float()).long()
            index = index - 1
            index = index.to(device)
            batch_size = input.shape[0]
            channels = input.shape[1]
            # get one-hot label
            index_ex = torch.zeros(batch_size, self.block_num, input.shape[2], input.shape[3]).to(device)
            index_ex = index_ex.scatter_(1, index, 1)
            block_value_sum = torch.sum(index_ex, dim=(2, 3))

            # computing the regional mean of input
            input_ = input.repeat(self.block_num, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
            index_ex = index_ex.unsqueeze(2)
            input_means = torch.sum(index_ex * input_, dim=(3, 4)) / (
                    block_value_sum + (block_value_sum == 0).float()).unsqueeze(2)  # * mask.unsqueeze(2)

            # computing the adjacency matrix
            input_means_ = input_means.repeat(self.block_num, 1, 1, 1).permute(1, 2, 0, 3)
            input_means_ = (input_means_ - input_means.unsqueeze(1)).permute(0, 2, 1, 3)
            M = self.W.mm(self.W.T).to(device)
            adj = input_means_.reshape(batch_size, -1, channels).matmul(M)
            adj = torch.sum(adj * input_means_.reshape(batch_size, -1, channels), dim=2).view(batch_size,
                                                                                              self.block_num,
                                                                                              self.block_num)
            adj = torch.exp(-1 * adj) + torch.eye(self.block_num).repeat(batch_size, 1, 1).to(device)
            if self.adj_mask is not None:
                adj = adj * self.adj_mask

            # generating the adj_mean
            adj_means = input_means.repeat(self.block_num, 1, 1, 1).permute(1, 0, 2, 3) * adj.unsqueeze(3)
            adj_means = (1 - torch.eye(self.block_num).reshape(1, self.block_num, self.block_num, 1).to(
                device)) * adj_means
            adj_means = torch.sum(adj_means, dim=2)  # batch_size，self.block_num, channel_num

            # obtaining the graph update features
            features = torch.sum(index_ex * (input_ + adj_means.unsqueeze(3).unsqueeze(4)), dim=1)
            #             features = data_norm(features)
            features = features.cpu()
            features = self.bn(features)
            features = features.to(device)
        else:
            features = input
            features = self.bn(features)
        return features

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class CAGCNet(nn.Module):
    def __init__(self, in_channel, block_num, class_num, adj_mask=None, if_feature_update=True, scale_layer=4):
        super(CAGCNet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.gcn1 = Graph2dConvolution(in_channel, in_channel, block_num=block_num, adj_mask=adj_mask,
                                       if_feature_update=if_feature_update)

        self.gcn2 = Graph2dConvolution(in_channel, in_channel, block_num=block_num, adj_mask=adj_mask,
                                       if_feature_update=if_feature_update)

        self.gcn3 = Graph2dConvolution(in_channel, in_channel, block_num=block_num, adj_mask=adj_mask,
                                       if_feature_update=if_feature_update)

        self.gcn4 = Graph2dConvolution(in_channel, in_channel, block_num=block_num, adj_mask=adj_mask,
                                       if_feature_update=if_feature_update)

        self.block_num = block_num
        self.Softmax = nn.Softmax(dim=1)
        self.scale_layer = scale_layer

    def forward(self, hsimg, seg_index):
        Up = nn.UpsamplingBilinear2d(size=(hsimg.shape[2], hsimg.shape[3]))
        f1, f2, f3, f4, f1_, f2_, f3_, final_class = 0, 0, 0, 0, 0, 0, 0, []
        index = seg_index.long().to(device)

        if self.scale_layer >= 1:
            f1 = self.gcn1(hsimg, index)
            f1_ = self.maxpool(f1)
            f1 = Up(f1_)

        if self.scale_layer >= 2:
            f2 = self.gcn2(f1_, index)
            f2_ = self.maxpool(f2)
            f2 = Up(f2_)

        if self.scale_layer >= 3:
            f3 = self.gcn3(f2_, index)
            f3_ = self.maxpool(f3)
            f3 = Up(f3_)

        if self.scale_layer >= 4:
            f4 = self.gcn4(f3_, index)
            f4_ = self.maxpool(f4)
            f4 = Up(f4_)
        final_f = torch.cat((f1, f2, f3, f4), dim=1)
        return final_f
        # feature size is 1*12*325*220


class feature_extractor(nn.Module):
    def __init__(self, in_channel, in_channel1, block_num, block_num1, class_num,
                 adj_mask, adj_mask1, graph_data, index, index1, if_feature_update=True, scale_layer=4):
        super(feature_extractor, self).__init__()
        self.index = index
        self.index1 = index1
        self.seg_index2 = []
        self.count = 0
        self.graph_data = graph_data
        self.conv1 = GATConv(in_channels=63, out_channels=16, heads=8, dropout=0.6)
        self.conv2 = GATConv(in_channels=16 * 8, out_channels=63, heads=1, concat=False, dropout=0.6)
        self.net = CAGCNet(in_channel, block_num, class_num + 1, adj_mask,
                          if_feature_update=True, scale_layer=scale_layer)
        self.net1 = CAGCNet(in_channel1, block_num1, class_num + 1, adj_mask1,
                           if_feature_update=True, scale_layer=scale_layer)
        self.resnet = ResBlk((in_channel + 2 * in_channel1) * 4, class_num + 1)
        self.encoder1 = ResBlk(4 * in_channel, 64)
        self.encoder2 = ResBlk(4 * in_channel1, 64)
        self.decoder1 = ResBlk(64, 32)
        self.decoder2 = ResBlk(64, 32)
        self.fc = nn.Linear(166 * 600, 512)
        self.graph_data = graph_data

    def forward(self, x1, x2):
        f = self.net(x1, self.index)
        x, edge_index = graph_data.x, graph_data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        f2 = self.conv2(x, edge_index)
        f2 = f2.permute(0, 1).view(63, 166, -1)

        if self.count == 0:
            self.seg_index2 = slic(np.array(f2.permute(1, 2, 0).detach()), n_segments=20)
            self.seg_index2 = torch.Tensor(self.seg_index2.copy()).unsqueeze(0).unsqueeze(0).to(device)
            self.count += 1
        f2 = f2.unsqueeze(0)
        f2 = f2.to(device)
        f1 = self.net1(f2, self.seg_index2)
        f3 = self.net1(x2, self.index1)
        final_f = torch.cat((f, f1, f3), dim=1)

        # print(f.size())
        # print(f1.size())
        # print(f3.size())
        # torch.Size([1, 8, 166, 600]) torch.Size([1, 252, 166, 600]) torch.Size([1, 252, 166, 600])
        h1 = self.encoder1(f)
        h1 = self.decoder1(h1).squeeze(0)
        h1 = torch.mean(h1, dim=0, keepdim=True).cpu()
        f_ = torch.mean(f.squeeze(0), dim=0, keepdim=True).cpu()
        # h1.size = 1*166*600
        h2 = self.encoder2(f1)
        h2 = self.decoder2(h2).squeeze(0)
        h2 = torch.mean(h2, dim=0, keepdim=True).cpu()
        f1_ = torch.mean(f1.squeeze(0), dim=0, keepdim=True).cpu()
        h3 = self.encoder2(f3)
        h3 = self.decoder2(h3).squeeze(0)
        h3 = torch.mean(h3, dim=0, keepdim=True).cpu()
        f3_ = torch.mean(f3.squeeze(0), dim=0, keepdim=True).cpu()
        l1 = torch.norm(h2 - h1, p=2) + torch.norm(h3 - h1, p=2)
        l2 = torch.norm(h1 - f_, p=2) + torch.norm(h2 - f1_, p=2) + torch.norm(h3 - f3_, p=2)
        loss = l1 + 1 / 10 * l2
        final_class = self.resnet(final_f)

        return final_class, loss


best_kappa = 0
best_OA, best_AA, best_list = 0, 0, []
best_kappa1 = 0
best_OA1, best_AA1, best_list1 = 0, 0, []


def train_and_test():
    Net = feature_extractor(channel_num, channel_num1, Block_num, Block_num1, class_num, adj_mask, adj_mask1,
                            graph_data,
                            seg_index.unsqueeze(0).unsqueeze(0).to(device),
                            seg_index1.unsqueeze(0).unsqueeze(0).to(device),
                            if_feature_update=if_up, scale_layer=scale)

    lossf = nn.CrossEntropyLoss()
    EPOCHS = 1000
    FOUND_LR = 1e-3
    opt = torch.optim.Adam(Net.parameters(), lr=FOUND_LR)
    scheduler = lr_scheduler.StepLR(opt, step_size=500, gamma=0.6)
    losses = []
    global best_kappa, best_OA, best_AA, best_list
    hsimg = data.permute(2, 0, 1).unsqueeze(0).to(device)
    hsimg1 = data1.permute(2, 0, 1).unsqueeze(0).to(device)
    out_color = np.zeros((166, 600, 3))
    for epoch in range(EPOCHS):
        final_f, loss1 = Net(hsimg, hsimg1)  # 要求返回的final_class是1*channel*长*宽
        train_gt = (ground_truth * train_mask).to(device)
        pre_gt = torch.cat((train_gt.unsqueeze(0), final_f[0]), dim=0).view(class_num + 2, -1).permute(1, 0)
        pre_gt_ = pre_gt[torch.argsort(pre_gt[:, 0], descending=True)]
        pre_gt_ = pre_gt_[:int(train_sum)]
        # torch.Size([1457500, 13])
        # torch.Size([1457500, 13])
        # torch.Size([8052, 13])
        loss = lossf(pre_gt_[:, 1:], pre_gt_[:, 0].long()) + 1 / 1000 * loss1
        losses.append(float(loss))

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            final_f = Net(hsimg, hsimg1)[0].cpu().detach()
            test_gt = ground_truth * test_mask
            pre_gt = torch.cat((test_gt.unsqueeze(0), final_f[0]), dim=0).view(class_num + 2, -1).permute(1, 0)
            pre_gt_ = pre_gt[torch.argsort(pre_gt[:, 0], descending=True)]
            pre_gt_ = pre_gt_[:int(test_sum)]
            OA, AA, kappa, ac_list = performance(pre_gt_[:, 1:], pre_gt_[:, 0].long(), class_num)

            if best_kappa < kappa:
                best_kappa = kappa
                best_OA = OA
                best_AA = AA
                best_list = ac_list
                expected = torch.max(torch.softmax(final_f[0], dim=0), dim=0)[1].cpu() * (ground_truth > 0)
                for i in range(166):
                    for j in range(600):
                        if expected[i][j] == 0:
                            out_color[i][j] = [21, 23, 23]
                        if expected[i][j] == 1:
                            out_color[i][j] = [43, 62, 133]
                        if expected[i][j] == 2:
                            out_color[i][j] = [154, 73, 41]
                        if expected[i][j] == 3:
                            out_color[i][j] = [28, 92, 235]
                        if expected[i][j] == 4:
                            out_color[i][j] = [107, 187, 116]
                        if expected[i][j] == 5:
                            out_color[i][j] = [144, 72, 157]
                        if expected[i][j] == 6:
                            out_color[i][j] = [217, 164, 107]
                cv2.imwrite("trento.png", out_color)

                # plt.imshow(torch.max(torch.softmax(final_f[0], dim=0), dim=0)[1].cpu() * (ground_truth > 0).float())
                display.clear_output(wait=True)
                # plt.show()
                print('epoch', epoch, ':', 'OA:', OA, 'AA:', AA, 'KAPPA:', kappa)
                print('epoch', epoch, ':', 'Accuracy_list:', ac_list)

        if (epoch + 1) % 100 == 0:
            losses = []

    return best_kappa, best_OA, best_AA, best_list


train_sum = torch.sum(train_mask)
test_sum = torch.sum(test_mask)
# train_sum = 1650
# test_sum = 53687 - 1650
best_kappas, best_OAs, best_AAs, best_lists = [], [], [], []
list_best_OAs_mean, list_best_AAs_mean, list_best_kappas_mean = [], [], []
list_best_OAs_std, list_best_AAs_std, list_best_kappas_std = [], [], []

scale_list = [4]
feature_up_list = [True]

for scale in scale_list:
    for if_up in feature_up_list:
        for i in range(10):
            best_kappa, best_OA, best_AA, best_list = train_and_test()
            best_kappas.append(best_kappa)
            best_OAs.append(best_OA)
            best_AAs.append(best_AA)
            best_lists.append(best_list)

        list_best_OAs_mean.append(np.mean(best_OAs))
        list_best_AAs_mean.append(np.mean(best_AAs))
        list_best_kappas_mean.append(np.mean(best_kappas))

        list_best_OAs_std.append(np.std(best_OAs))
        list_best_AAs_std.append(np.std(best_AAs))
        list_best_kappas_std.append(np.std(best_kappas))

