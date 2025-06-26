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
        N = 5  # Number of frames to be averaged, can be set externally as a parameter if necessary

        if not hasattr(self, "frame_buffer"):
            self.frame_buffer = []

        # Add the current frame to the buffer
        self.frame_buffer.append(img.copy())

        # If there are more than N frames, delete the oldest
        if len(self.frame_buffer) > N:
            self.frame_buffer.pop(0)

        # Convert to float32 for averaging to prevent overflow.
        img = np.mean(np.array(self.frame_buffer).astype(np.float32), axis=0)

        # Turning back to the uint8 image
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def _212_HistogrammSpreizung(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.1.2 (Histogrammspreizung)
            - Transformation HSV
            - Histogrammspreizung berechnen
            - Transformation BGR
        """
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Calculate V-channel minimum and maximum values
        v_min = np.min(v)
        v_max = np.max(v)

        # Avoid dividing by 0
        if v_max - v_min == 0:
            v_stretched = v
        else:
            v_stretched = ((v - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        # Merge and convert back to BGR
        hsv_stretched = cv2.merge([h, s, v_stretched])
        img = cv2.cvtColor(hsv_stretched, cv2.COLOR_HSV2BGR)

        return img

    def _221_RGB(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.2.1 (RGB)
            - Histogrammberechnung und Analyse
        """
        if not self.capture_for_rgb_analysis:
            return  # No processing when there are no clicks

        print("saving...")

        # Save current image
        cv2.imwrite("rgb_input_image.png", img)

        # Plotting RGB three-channel histograms
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

        print("saved：rgb_input_image.png 和 rgb_histogram.png")

        # self.capture_for_rgb_analysis = False


    def _222_HSV(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.2.2 (HSV)
            - Histogrammberechnung und Analyse im HSV-Raum
        """
        if not self.capture_for_rgb_analysis:
            return  

        print("saving...")

        # Convert to HSV
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

        print("saved hsv_histogram.png")
        self.capture_for_rgb_analysis = False


    def _23_SegmentUndBildmodifizierung (self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (StatischesSchwellwertverfahren)
            - Binärmaske erstellen
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the red range
        lower_red1 = np.array([0, 150, 120])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 120])
        upper_red2 = np.array([180, 255, 255])

        # Generate Mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        self.binary_mask = mask 


        """
            Hier steht Ihr Code zu Aufgabe 2.3.2 (Binärmaske)
            - Binärmaske optimieren mit Opening/Closing
            - Wahl größte zusammenhängende Region
        """
        kernel = np.ones((5, 5), np.uint8)

        mask_clean = cv2.morphologyEx(self.binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find the maximum connectivity area
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

        if num_labels > 1:
            max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_largest = np.uint8(labels == max_label) * 255
        else:
            mask_largest = mask_clean

        self.binary_mask = mask_largest

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
            self.background = img.copy()
            print("Background has been saved.")
            return img

        if self.background is None:
            return img

        # Create foreground mask
        mask_inv = cv2.bitwise_not(self.binary_mask)

        # Extract the current image portion of the non-cloaked area
        fg = cv2.bitwise_and(img, img, mask=mask_inv)

        # Extract the background portion of the cloak area
        bg = cv2.bitwise_and(self.background, self.background, mask=self.binary_mask)

        img = cv2.add(fg, bg)

        return img