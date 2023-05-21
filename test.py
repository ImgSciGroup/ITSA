from args import args
from model.MFCN import MFCN
from util.Evaluation import Evaluation
from util.EvaluationNew import Index
import torchvision.transforms as Transforms
import time
import glob
import cv2
import numpy as np
import torch
def get_CMI(pred_img,size,size1):
    a = torch.sigmoid(pred_img)
    b = torch.squeeze(a, dim=0)
    b = np.array(b.data.cpu())
    c = np.zeros((size,size1))
    for i in range(size):
        for j in range(size1):
            if (b[0][i][j] >= b[1][i][j]):
                c[i][j] = 1-b[0][i][j]
            if (b[0][i][j] < b[1][i][j]):
                c[i][j] = b[1][i][j]
    return c
def get_result(a,size,size1):
    a = torch.sigmoid(a)
    b = torch.argmax(a, dim=1)
    # b=torch.unsqueeze(b, dim=1)
    b = torch.squeeze(b, dim=0)
    b = np.array(b.data.cpu())
    img = np.zeros((size,size1), np.uint8)
    for i in range(size):
        for j in range(size1):
            if (b[i][j] == 0):
                img[i, j] = 0
            if (b[i][j] == 1):
                img[i, j] = 255
    return img
def get_throd(pred):
    temp=0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            temp+=pred[i][j]
    return temp/(pred.shape[0]*pred.shape[1])
def test(net,iter,f_acc,f_time,ModelName) :
    print('Starting test...')
    # 选择设备，有cuda用cuda，没有就用cpu
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device="cpu"
    # 加载网络，图片3通道，分类为1。
    # net = SiamUnet_conc(input_nbr=3, label_nbr=1)
    # net = SiamUnet_diff(input_nbr=3, label_nbr=1)
    # net = Unet(input_nbr=6, label_nbr=2)

    # 将网络拷贝到device
    net.to(device=device)

    # 加载模型参数
    net.load_state_dict(torch.load('pth/'+str(ModelName)+'_'+str(iter)+'.pth', map_location=device))

    # 测试模式
    net.eval()

    trans = Transforms.Compose([Transforms.ToTensor()])
    # 读取所有图片路径
    tests1_path = glob.glob('./'+args.path+'/pred/image1/*.tif')
    tests2_path = glob.glob('./'+args.path+'/pred/image2/*.tif')
    label_path = glob.glob('./'+args.path+'/pred/label/*.bmp')
    # 遍历所有图片
    num = 1
    TPSum = 0
    TNSum = 0
    FPSum = 0
    FNSum = 0
    C_Sum_or = 0
    UC_Sum_or = 0
    for tests1_path, tests2_path, label_path in zip(tests1_path, tests2_path, label_path):
        starttime = time.time()
        # 保存结果地址
        save_res_path = '.' + tests1_path.split('.')[1] +"_"+ str(iter)+'.png'
        save_res_path = save_res_path.replace('image1', 'result')
        name = tests1_path.split('/')[4].split('.')[0]
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
        # 将tensor拷贝到device中：有gpu就拷贝到gpu，否则就拷贝到cpu
        # 预测
        # 使用网络参数，输出预测结果
        pred_Img = net(test1_img, test2_img)
        # pred_Img = net(torch.cat([test1_img,test2_img], dim=1))
        #得到变化幅度图
        c=get_CMI(pred_Img,label_img.shape[0],label_img.shape[1])

        # cv2.normalize(c, c, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        pred = np.uint8(c * 255.0)

        cv2.imwrite(args.path+"/pred/CMI/"+str(iter)+"_CMI.png",pred)
        binary=get_result(pred_Img,pred.shape[0],pred.shape[1])
        # 提取结果

        # pred_Img=torch.sigmoid(pred_Img)
        # pred = np.array(pred_Img.data.cpu()[0])[0]
        # pred=np.uint8(pred*255.0)
        # print(pred.shape)
        # ret, binary = cv.threshold(pred,0, 255, cv2.THRESH_OTSU)
        # pred[pred>=t] = 255
        # pred[pred <t] = 0
        print(num, tests1_path)
        # 保存图片
        cv2.imwrite(save_res_path, binary)
        endtime = time.time()
        if num == 0:
            f_time.write('each pair images time\n')
        f_time.write(str(num)+','+str(starttime)+','+str(endtime)+','+'-------测试时间：'+str(float('%2f' % (endtime-starttime))) + '\n')
        # 评价精度
        monfusion_matrix = Evaluation(label=label_img, pred=binary)
        TP, TN, FP, FN, c_num_or, uc_num_or = monfusion_matrix.ConfusionMatrix()
        TPSum += TP
        TNSum += TN
        FPSum += FP
        FNSum += FN
        C_Sum_or += c_num_or
        UC_Sum_or += uc_num_or
        # 保存验证集loss和accuracy
        if num == 1:
            f_acc.write('=================================================================================\n')
            f_acc.write('|Note: (num, FileName, TP, TN, FP, FN)|\n')
            f_acc.write('|Note: (ACC: FileName, OA, FA, MA, TE, mIoU, c_IoU, uc_IoU, Precision, Recall, F1)|\n')
            f_acc.write('=================================================================================\n')
        f_acc.write(str(iter) +'================================================================================='+ '\n')
        f_acc.write(str(num) + ',' + str(name) + '.tif' + ',' + str(TP) + ',' + str(TN) + ',' +
                    str(FP) + ',' + str(FN) + '\n')

        num += 1
        # if num % 10 == 0:
        #     Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
        #     IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
        #     OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
        #     FA, MA, TE = Indicators.CD_indicators()
        #
        #     print("OA=", str(float('%4f' % OA)), "^^^^^", "mIoU=", str(float('%4f' % IoU)), "^^^^^", "c_mIoU=", str(float('%4f' % c_IoU)), "^^^^^", "uc_mIoU=", str(float('%4f' % uc_IoU)), "^^^^^", "Precision=",
        #           str(float('%4f' % Precision)), "^^^^^", "Recall=", str(float('%4f' % Recall)), "^^^^^", "mF1=", str(float('%4f' % F1)))
            # Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
            # IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
            # OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
            # save_res_path = '.' + tests1_path.split('.')[1] + str(iter) +"_"+str(OA)+ '.png'
            # save_res_path = save_res_path.replace('image1', 'result')
            # cv2.imwrite(save_res_path, binary)
    Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
    print("OA=", str(float('%4f' % OA)),  "^^^^^",  "Precision=",
          str(float('%4f' % Precision)), "^^^^^", "Recall=", str(float('%4f' % Recall)), "^^^^^", "mF1=",
          str(float('%4f' % F1)))
    f_acc.write('==========================================================================================================\n')
    f_acc.write('|SumConfusionMatrix:|  TP   |   TN   |  FP  |  FN   |\n')
    f_acc.write('|SumConfusionMatrix:|' + str(TPSum) + '|' + str(TNSum) + '|' + str(FPSum) + '|' + str(FNSum) + '|\n')
    f_acc.write('==========================================================================================================\n')
    f_acc.write('|TotalAcc:|   OA   |   FA   |   MA    |  TE   |  mIoU   |  c_IoU  | uc_IoU  |Precision| Recall  |   F1    |\n')
    f_acc.write('|TotalAcc:|' + str(float('%4f' % OA)) + '|' +
                str(float('%4f' % Precision)) + '|' + str(float('%4f' % Recall)) + '|' + str(float('%4f' % F1)) + '|\n')
    f_acc.write(
        '==========================================================================================================\n')
if __name__ == "__main__":
    iter=1
    f_acc = open('txt/test_acc_'+str(iter)+'.txt', 'w')
    f_time = open('txt/test_time_'+str(iter)+'.txt', 'w')
    net = MFCN(n_channels=3, n_classes=2)
    test(net,iter,f_acc,f_time,"siam_diff")