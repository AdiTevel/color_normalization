import numpy as np
import pickle
import cv2

class ColorClassifier():
    def __init__(self,Color_Grading):
        self.Color_Grading = Color_Grading
        if Color_Grading:
            with open('./models/old_model/clf.pkl', 'rb') as file:
                self.classifier_net  = pickle.load(file)
            with open('./models/old_model/scaler.pkl','rb') as file:
            # with open('/weights/color_classifier/scaler.pkl', 'rb') as file:
                self.scalar  = pickle.load(file)



    def grade(self,rgbcrop, maskcrop):
        if self.Color_Grading:
            # bgrcrop = cv2.cvtColor(rgbcrop, cv2.COLOR_RGB2BGR)
            rgb_vector = rgbcrop[maskcrop]
            if len(rgb_vector) == 0:
                return -1
            r, g, b= rgb_vector[:, 0], rgb_vector[:, 1], rgb_vector[:, 2]
            b = b.astype(float)
            g = g.astype(float)
            r = r.astype(float)
            mean_sum_rgb = np.mean(r + g + b)
            Rn = np.mean(r) / mean_sum_rgb  # Normalized red of RGB (Rn)
            Gn = np.mean(g) / mean_sum_rgb  # Normalized green of RGB(Gn)
            Bn = np.mean(b) / mean_sum_rgb  # Normalized blue of RGB(Bn)

            hls = cv2.cvtColor(rgb_vector.reshape((-1, 1, 3)), cv2.COLOR_RGB2HLS).reshape((-1, 3))
            h, l, s = hls[:, 0].astype(float), hls[:, 1].astype(float), hls[:, 2].astype(float)
            meanL = np.mean(l)  # mean of L in HSL
            meanH = np.mean(h)
            meanS = np.mean(s)

            ycbcr = cv2.cvtColor(rgb_vector.reshape((-1, 1, 3)), cv2.COLOR_RGB2YCrCb).reshape((-1, 3))
            y, cr, cb = ycbcr[:, 0].astype(float), ycbcr[:, 1].astype(float), ycbcr[:, 2].astype(float)
            sdCb = np.std(cb)  # SD of Cb in YCbCr
            meanY = np.mean(y)  # mean of L in HSL
            meanCb = np.mean(cb)
            meanCr = np.mean(cr)

            lab = cv2.cvtColor(rgb_vector.reshape((-1, 1, 3)), cv2.COLOR_RGB2LAB).reshape((-1, 3))
            L, a, bu = lab[:, 0].astype(float), lab[:, 1].astype(float), lab[:, 2].astype(float)
            meanA = np.mean(a)  # mean of a in Lab
            meanB = np.mean(bu)  # mean of b in Lab
            meanLa = np.mean(L)
            hue_angle = np.mean(np.arctan(bu / a))

            EXR = 1.4 * Rn - Gn  # excess red
            cive = 0.441 * Rn - 0.811 * Gn + 0.385 * Bn + 18.78  # Color index for extracted vegetation cover(CIVE)
            rbi = (Rn - Bn) / (Rn + Bn)  # Red-blue contrast (RBI)

            parameters = [EXR, Rn, Gn, Bn, meanL, meanH, meanS,meanLa, meanA, meanB, meanCb, meanCr, meanY,sdCb, cive, rbi, hue_angle]
            features = self.scalar.transform(np.array(parameters).reshape(1, -1))
            grade = float(self.classifier_net.predict(features))

            return grade
        else:
            return -1