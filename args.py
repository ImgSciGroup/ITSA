import cv2
class args():
         im1=cv2.imread("data/dataset-4/pred/image1/pre.tif")
         im2=cv2.imread("data/dataset-4/pred/image2/post.tif")
         la=cv2.imread("data/dataset-4/pred/label/gt.bmp")
         im22 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
         im11=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
         la22 = cv2.cvtColor(la, cv2.COLOR_BGR2GRAY)
         condi=0.007      #终止条件
         epoch=20         #训练次数
         fc=1.5           #方差值
         path= "data/dataset-4"
         times = 3        #初始样本个数
         patch = 16       #块大小