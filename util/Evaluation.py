from sklearn import metrics
import numpy as np
import cv2

class Evaluation():
    # 计算各应用评价指标
    def __init__(self, label, pred):
        super(Evaluation, self).__init__()
        self.label = label / 255
        self.pred = pred / 255

    def ConfusionMatrix(self):
        raw = self.label.shape[0]
        col = self.label.shape[1]
        size = raw * col
        union = np.clip(((self.label + self.pred)), 0, 1)
        intersection = (self.label * self.pred)
        TP = int(intersection.sum())
        TN = int(size - union.sum())
        FP = int((self.pred - intersection).sum())
        FN = int((self.label - intersection).sum())

        # c_num_and = TP
        c_num_or = int(union.sum())
        # uc_num_and = TN
        uc_num_or = int(size - intersection.sum())

        return TP, TN, FP, FN, c_num_or, uc_num_or
        # return size, TP, TN, FP, FN

if __name__ == "__main__":
    pred_path = "D:\\PHD_Research\\DL\\U-Net\\data\\AriealData\\test\\image\\0_res.png"
    label_path = "D:\\PHD_Research\\DL\\U-Net\\data\\AriealData\\test\\label\\0.tif"
    pred = cv2.imread(pred_path)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    label = cv2.imread(label_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    Indicators = Evaluation(label, pred)
    OA, kappa, AA = Indicators.Classification_indicators()
    FA, MA, TE = Indicators.CD_indicators()
    CP, CR, AQ = Indicators.Landsilde_indicators()
    IOU = Indicators.IOU_indicator()
    Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
    print("(OA, KC, AA)", OA, kappa, AA)
    print("(FA, MA, TE)", FA, MA, TE)
    print("(CP, CR, AQ)", CP, CR, AQ)
    print("(IoU, Precision, Recall, F1-score)", IOU, Precision, Recall, F1)