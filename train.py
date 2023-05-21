from util.dataset3 import ISBI_Loader
from torch import optim
# from tensorboardX import SummaryWriter
import torchvision.transforms as Transforms
import torch.utils.data as data
import time
import glob
import cv2
from util.Evaluation import Evaluation
import numpy as np
import torch
import torch.nn as nn
from util.EvaluationNew import Index
def train_net(f,f1,f_epoch_loss,iter,net, device,data_path, epochs, batch_size, ModelName, is_Transfer):
    print('Conrently, Traning Model is :::::'+ModelName+':::::')
    if is_Transfer:
        print("Loading Transfer Learning Model.........")
        # BFENet.load_state_dict(torch.load('Pretrain_BFE_'+ModelName+'_model_epoch75_mIoU_89.657089.pth', map_location=device))
    else:
        print("No Using Transfer Learning Model.........")

    # 加载数据集
    isbi_dataset = ISBI_Loader(data_path=data_path, transform=Transforms.ToTensor())
    train_loader = data.DataLoader(dataset=isbi_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
    # 定义RMSprop算法
    # optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)

    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)
    # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70], gamma=0.9)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90], gamma=0.9)
    # 定义loss
    # weight = torch.tensor([5], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss()
    # criterion=nn.NLLLoss().to(device)
    #criterion = CDWithLogitsLoss(device, beta_EL=0, beta_FL=0, beta_BCE=1)
    # criterion = BCEFocalLoss(device)
    # best loss, 初始化为正无穷
    # best_loss = float('inf')
    # writer = SummaryWriter('runs/exp')
    f_loss = open('txt/train_loss.txt', 'a')
    f_time = open('train_time.txt', 'w')
    # 训练epochs次
    starttime1 = time.time()
    loss_=[]
    f_loss.write("第"+str(iter)+"次迭代： "+"\n")
    for epoch in range(1, epochs+1):
        net.train()
        # 训练模式
        # learning rate delay
        best_loss = float('inf')
        best_F1 = float('inf')
        # 按照batch_size开始训练
        num = 0
        starttime = time.time()
        sum_total=0
        print('==========================epoch = '+str(epoch)+'==========================')
        for image1, image2, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image1 = image1.to(device=device)
            image2 = image2.to(device=device)
            label = label.to(device=device)
            # 使用网络参数，输出预测结果
            list = []   # 0: out1,1: out2,2: feat1,3: feat2
            pred = net(image1, image2)
            # pred=net(torch.cat([image1,image2], dim=1))
            # pred = F.log_softmax(pred, dim=1)
            label=label.long()
            x_onehot = torch.zeros(label.shape[0], 2, 16, 16).long()
            x_onehot.scatter_(1, label, 1).float()
            # print(x_onehot.shape)
            # pred=torch.sigmoid(pred)
            # label = torch.squeeze(label, dim=1)
            # pred = torch.argmax(pred, dim=1)
            # pred= torch.unsqueeze(pred,  dim=1)
            # pred= pred.float()
            # print(pred.dtype)
            x_onehot=x_onehot.float()
            # 计算loss
            # print(pred.shape)
            # print(x_onehot.shape)
            total_loss = criterion(pred, x_onehot)

            # f_loss.write(str(num) + ',' + str(float('%5f' % total_loss)) + '\n')

            # writer.add_scalar('epoch', total_loss, global_step=epoch)
            # writer.add_scalar('train_total_num', total_loss, global_step=num)
            print(str(epoch)+'/' + str(epochs)+':::::'+'lr='+str(optimizer.param_groups[0]['lr'])+':::::'+str(num)+'/'+str(int(len(isbi_dataset)/batch_size)))
            print('Loss/train', total_loss.item())
            print('-----------------------------------------------------------------------')
            sum_total+=total_loss.item()

            # 保存loss值最小的网络参数
            # if epoch % 10 == 0:
            #     if total_loss < best_loss:
            #         best_loss = total_loss
            #         BFE_path = 'best_BFE_SPM_model_epoch' + str(epoch) + '.pth'
            #         BCD_path = 'best_BCD_SPM_model_epoch' + str(epoch) + '.pth'
            #         #torch.save(BFENet.state_dict(), BFE_path)
            #         torch.save(net.state_dict(), BCD_path)
            # 更新参数
            total_loss.requires_grad_(True)  # 加入此句就行了
            total_loss.backward()
            optimizer.step()
            num += 1
        # learning rate delay
        scheduler1.step()
        endtime = time.time()
        sum_total = round(sum_total/num, 5)
        loss_.append(sum_total)
        f_loss.write(str(sum_total)+"\n")
        # val
        # if epoch > 10 and epoch % 2 == 0:
        # if epoch > 10:
        #     with torch.no_grad():
        #         mOA, IoU = val(net, device, epoch)
        #         # best_F1 = F1
        #         print(str(epoch) + ':::::OA=' + str(float('%2f' % (mOA))) + ':::::mIoU=' + str(float('%2f' % (IoU))))
        #         modelpath = 'BestmIoU_' + str(ModelName) + '_epoch' + str(epoch) + '_mIoU_' + str(float('%2f' % IoU)) + '.pth'
        #         torch.save(net.state_dict(), modelpath)

        if epoch == 0:
            f_time.write('each epoch time\n')
        f_time.write(str(epoch)+','+str(starttime)+','+str(endtime)+','+str(float('%2f' % (starttime-endtime))) + '\n')
    torch.save(net.state_dict(), 'pth/'+str(ModelName)+'_'+str(iter)+'.pth')
    endtime1 = time.time()
    # f_landsat.write(str(iter)+","+str(float('%2f' % (endtime1-starttime1)))+",")
    f1.write(str(iter) + "," + str(float('%2f' % (endtime1 - starttime1)))+"\n")
    print(str(iter) + "," + str(float('%2f' % (endtime1 - starttime1)))+"\n")
    # torch.save(net.state_dict(), 'best_BCD_' + str(ModelName) + '_model_final.pth')
    # writer.close()
    f_loss.close()
    f_time.close()
    return loss_
def val(net1, device, epoc):
    net1.eval()
    tests1_path = glob.glob('./samples/LEVIR-CD/test/image1/*.png')
    tests2_path = glob.glob('./samples/LEVIR-CD/test/image2/*.png')
    label_path = glob.glob('./samples/LEVIR-CD/test/label/*.png')
    trans = Transforms.Compose([Transforms.ToTensor()])
    TPSum = 0
    TNSum = 0
    FPSum = 0
    FNSum = 0
    C_Sum_or = 0
    UC_Sum_or = 0
    num = 0
    val_acc = open('val_acc.txt', 'a')
    val_acc.write('===============================' + 'epoch=' + str(epoc) + '==============================\n')
    for tests1_path, tests2_path, label_path in zip(tests1_path, tests2_path, label_path):
        num += 1
        # 读取图片
        test1_img = cv2.imread(tests1_path)
        test2_img = cv2.imread(tests2_path)
        label_img = cv2.imread(label_path)
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
        test1_img = trans(test1_img)
        test2_img = trans(test2_img)
        test1_img = test1_img.unsqueeze(0)
        test2_img = test2_img.unsqueeze(0)
        test1_img = test1_img.to(device=device, dtype=torch.float32)
        test2_img = test2_img.to(device=device, dtype=torch.float32)
        # 使用网络参数，输出预测结果
        list = []
        out = net1(test1_img, test2_img)

        # 提取结果
        pred = np.array(out.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        monfusion_matrix = Evaluation(label=label_img, pred=pred)
        TP, TN, FP, FN, c_num_or, uc_num_or = monfusion_matrix.ConfusionMatrix()
        TPSum += TP
        TNSum += TN
        FPSum += FP
        FNSum += FN
        C_Sum_or += c_num_or
        UC_Sum_or += uc_num_or

        if num > 400 and num % 10 == 0:
            Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
            IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
            OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
            print("OA=", str(float('%4f' % OA)), "^^^^^", "mIoU=", str(float('%4f' % IoU)), "^^^^^", "c_mIoU=", str(float('%4f' % c_IoU)), "^^^^^", "uc_mIoU=", str(float('%4f' % uc_IoU)), "^^^^^", "Precision=",
                  str(float('%4f' % Precision)), "^^^^^", "Recall=", str(float('%4f' % Recall)), "^^^^^", "mF1=", str(float('%4f' % F1)))
            val_acc.write('mIou = ' + str(float('%2f' % IoU)) + ',' + 'c_mIoU = ' +
                          str(float('%2f' % (c_IoU))) + ',' +
                          'uc_mIoU = ' + str(float('%2f' % (uc_IoU))) + ',' +
                          'F1 = ' + str(float('%2f' % (F1))) + '\n')
    Indicators2 = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    OA, Precision, Recall, F1 = Indicators2.ObjectExtract_indicators()
    IoU, c_IoU, uc_IoU = Indicators2.IOU_indicator()
    return OA, IoU