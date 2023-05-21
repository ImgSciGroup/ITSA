import cv2

from args import  args
from function.count import count_white_black
from model.MFCN import MFCN
from function.start import start01, save_img1
from train import  train_net
from test import test
from function.adddemo import  add

if __name__ == "__main__":

    iter=1
    # 第一次train,test
    im1=args.im1
    im2=args.im2
    im22=args.im22
    la=args.la
    pt_com=[]
    net = MFCN(n_channels=6, n_classes=2)
    pt_all1=[]
    pt_all0=[]
    pt_1,pt_0=start01(pt_com)
    save_img1(im1,im2,la,iter,pt_1,1)
    save_img1(im1,im2,la,iter,pt_0,0)
    pt_all1.append(pt_1)
    pt_all0.append(pt_0)
    loss_to=[]
    device='cpu'
    modelName="siam"
    f_acc=open('txt/test_acc.txt', 'w')
    f_init=open('txt/params.txt', 'w')
    f_init.write("condi:"+str(args.condi)+"  epoch:"+str(args.epoch)+" fc:"+str(args.fc))
    f_time=open('txt/test_time.txt', 'w')
    f_acc_train = open('txt/train_acc.txt', 'w')
    f_time_train = open('txt/train_time.txt', 'w')
    f_landsat_c = open('txt/f_landsat_c.txt', 'w')
    f_landsat_uc = open('txt/f_landsat_uc.txt', 'w')
    f_epoch_loss=open('txt/f_epoch_loss.txt', 'w')
    batch_size=3
    condi=args.condi
    while(iter<=2):
            loss_to=train_net(f_acc_train,f_time_train,f_epoch_loss,iter,net,device,args.path,args.epoch,3,modelName,is_Transfer=False)
            test(net, iter, f_acc, f_time,modelName)
            iter+=1
            pt_1=add(iter,pt_1,im1,im2,la,im22,1,pt_com)
            pt_all1.append(pt_1)
            pt_0 = add(iter, pt_0, im1, im2, la, im22,0,pt_com)
            pt_all0.append(pt_0)
    while(True):
        loss_to=train_net(f_acc_train,f_time_train,f_epoch_loss,iter, net, device, args.path,args.epoch, 3, modelName, is_Transfer=False)
        test(net, iter, f_acc, f_time, modelName)
        image1 = cv2.imread(args.path + "/pred/result/pre_" + (str(iter - 2) + ".png"))
        image2 = cv2.imread(args.path + "/pred/result/pre_" + (str(iter - 1) + ".png"))
        image3 = cv2.imread(args.path + "/pred/result/pre_" + (str(iter) + ".png"))
        black, white = count_white_black(image1, image2, image3)
        if (black < condi and white < condi):
            for i in range(len(loss_to)):
                f_epoch_loss.write(str(i+1)+","+str(loss_to[i])+"\n")
            break
        if(black>=condi):
            pt_0 = add(iter, pt_0, im1, im2, la, im22, 0, pt_com)
            pt_all0.append(pt_0)
        if(white>=condi):
            pt_1 = add(iter, pt_1, im1, im2, la, im22, 1, pt_com)
            pt_all1.append(pt_1)
        iter+=1
        batch_size+=1
    cv2.waitKey(0)
    f_acc.close()
    f_time.close()
    f_landsat_c.close()
    f_landsat_uc.close()
    f_acc_train.close()
    f_time_train.close()
    f_epoch_loss.close()
    f_init.close()