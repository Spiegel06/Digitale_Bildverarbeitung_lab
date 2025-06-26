import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

from . import Algorithm


class InvisCloak (Algorithm):

    """ init function """
    def __init__(self):
        self.capture_for_rgb_analysis = False
        self.click_triggered = False
        self.background = None          
        # pass

    """ Processes the input image"""
    def process(self, img):

        """ 2.1 Vorverarbeitung """
        """ 2.1.1 Rauschreduktion """
        plotNoise = True   # Schaltet die Rauschvisualisierung ein
        if plotNoise:
            self._plotNoise(img, "Rauschen vor Korrektur")
        img = self._211_Rauschreduktion(img)
        if plotNoise:
            self._plotNoise(img, "Rauschen nach Korrektur")
        """ 2.1.2 HistogrammSpreizung """
        img = self._212_HistogrammSpreizung(img)


        """ 2.2 Farbanalyse """
        """ 2.2.1 RGB """
        self._221_RGB(img)
        """ 2.2.2 HSV """
        self._222_HSV(img)


        """ 2.3 Segmentierung und Bildmdifikation """
        img = self._23_SegmentUndBildmodifizierung(img)

        return img

    """ Reacts on mouse callbacks """
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            print("A Mouse click happend! at position", x, y)
            self.capture_for_rgb_analysis = True
            self.click_triggered = True

    def _plotNoise(self, img, name:str):
        height, width = np.array(img.shape[:2])
        centY = (height / 2).astype(int)
        centX = (width / 2).astype(int)

        cutOut = 5
        tmpImg = deepcopy(img)
        tmpImg = tmpImg[centY - cutOut:centY + cutOut, centX - cutOut:centX + cutOut, :]

        outSize = 500
        tmpImg = cv2.resize(tmpImg, (outSize, outSize), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(name, tmpImg)
        cv2.waitKey(1)

    def _211_Rauschreduktion(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.1.1 (Rauschunterdrückung)
            - Implementierung Mittelwertbildung über N Frames
        """
        N = 5  # Anzahl der zu mittelden Frames, kann ggf. als Parameter extern gesetzt werden

        if not hasattr(self, "frame_buffer"):
            self.frame_buffer = []

        # 添加当前帧到缓冲区
        self.frame_buffer.append(img.copy())

        # 若超过N帧，删除最旧的
        if len(self.frame_buffer) > N:
            self.frame_buffer.pop(0)

        # 转换为 float32 进行求平均，防止溢出
        img = np.mean(np.array(self.frame_buffer).astype(np.float32), axis=0)

        # 再转回 uint8 图像
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def _212_HistogrammSpreizung(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.1.2 (Histogrammspreizung)
            - Transformation HSV
            - Histogrammspreizung berechnen
            - Transformation BGR
        """
        # 转换到 HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 计算 V 通道最小和最大值
        v_min = np.min(v)
        v_max = np.max(v)

        # 避免除以0
        if v_max - v_min == 0:
            v_stretched = v
        else:
            v_stretched = ((v - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        # 合并并转换回 BGR
        hsv_stretched = cv2.merge([h, s, v_stretched])
        img = cv2.cvtColor(hsv_stretched, cv2.COLOR_HSV2BGR)

        return img

    def _221_RGB(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.2.1 (RGB)
            - Histogrammberechnung und Analyse
        """
        if not self.capture_for_rgb_analysis:
            return  # 无点击时不进行处理

        print("▶ 正在保存当前图像与 RGB 直方图...")

        # 保存当前图像
        cv2.imwrite("rgb_input_image.png", img)

        # 绘制 RGB 三通道直方图
        colors = ('b', 'g', 'r')
        plt.figure()
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col, label=f"{col.upper()}-Kanal")
            plt.xlim([0, 256])
        
        plt.title("RGB Histogramm")
        plt.xlabel("Pixelwert")
        plt.ylabel("Anzahl der Pixel")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("rgb_histogram.png")
        plt.close()

        print("✅ 已保存：rgb_input_image.png 和 rgb_histogram.png")

        # 重置标志位
        # self.capture_for_rgb_analysis = False


    def _222_HSV(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.2.2 (HSV)
            - Histogrammberechnung und Analyse im HSV-Raum
        """
        if not self.capture_for_rgb_analysis:
            return  # 只在鼠标点击时进行分析

        print("▶ 正在保存 HSV 通道直方图...")

        # 转换到 HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        channels = [h, s, v]
        labels = ["Hue (H)", "Saturation (S)", "Value (V)"]
        colors = ["r", "g", "b"]

        plt.figure(figsize=(10, 6))
        for i in range(3):
            hist = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
            plt.plot(hist, color=colors[i], label=labels[i])
            plt.xlim([0, 256])

        plt.title("HSV Histogramm")
        plt.xlabel("Wert")
        plt.ylabel("Pixelanzahl")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("hsv_histogram.png")
        plt.close()

        print("✅ HSV 直方图已保存为 hsv_histogram.png")
        # 重置标志位
        self.capture_for_rgb_analysis = False


    def _23_SegmentUndBildmodifizierung (self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (StatischesSchwellwertverfahren)
            - Binärmaske erstellen
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 定义红色范围（可能需根据你的披风颜色调整）
        lower_red1 = np.array([0, 150, 120])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 120])
        upper_red2 = np.array([180, 255, 255])

        # 生成掩码（两个红色段）
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 目前只完成了掩码生成
        self.binary_mask = mask  # 保存下来，后续 2.3.2 和 2.3.3 要用


        """
            Hier steht Ihr Code zu Aufgabe 2.3.2 (Binärmaske)
            - Binärmaske optimieren mit Opening/Closing
            - Wahl größte zusammenhängende Region
        """
        kernel = np.ones((5, 5), np.uint8)

        mask_clean = cv2.morphologyEx(self.binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 找最大连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

        if num_labels > 1:
            # 跳过背景 label=0，从1开始
            max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_largest = np.uint8(labels == max_label) * 255
        else:
            mask_largest = mask_clean

        self.binary_mask = mask_largest  # 更新掩码用于后续背景替换

        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (Bildmodifizerung)
            - Hintergrund mit Mausklick definieren
            - Ersetzen des Hintergrundes
        """
        if not hasattr(self, 'background'):
            self.background = None
        if not hasattr(self, 'click_triggered'):
            self.click_triggered = False

        if self.click_triggered and self.background is None:
            # 用户点击后保存背景
            self.background = img.copy()
            print("✅ Hintergrund wurde gespeichert.")
            return img  # 本帧不做替换

        if self.background is None:
            return img  # 还没保存背景

        # 创建前景遮罩（非披风区域）
        mask_inv = cv2.bitwise_not(self.binary_mask)

        # 提取非披风区域的当前图像部分
        fg = cv2.bitwise_and(img, img, mask=mask_inv)

        # 提取披风区域的背景部分
        bg = cv2.bitwise_and(self.background, self.background, mask=self.binary_mask)

        # 合并：披风→背景，其它→保留
        img = cv2.add(fg, bg)

        return img