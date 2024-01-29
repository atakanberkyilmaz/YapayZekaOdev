import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pyqtgraph import *
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PIL import Image

global resim_konum 
resim_konum="22_real_chess.jpg"

def clearLayout(layout):
  while layout.count():
    child = layout.takeAt(0)
    if child.widget():
      child.widget().deleteLater()

class mainWidget(QWidget):

    def __init__(self,parent=None):
        super(mainWidget, self).__init__(parent)

        self.setWindowTitle(
            "YapayZeka")
        
        # font family
        font = QFont()
        font.setFamily("Arial")
        self.setFont(font)

        # layouts
        button_layout = QVBoxLayout()
        chart_layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        main_layout = QGridLayout()
        #image_layout.alignment("start")
        image_scroll=QScrollArea()
        # Buttons

        #-Original
        draw_original_button = QPushButton("Original")
        draw_original_button.setObjectName("draw_original_button")
        draw_original_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        draw_original_button.clicked.connect(self.on_click_draw_original_button)
        draw_original_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_original_button)

        #-Otsu
        draw_otsu_button = QPushButton("Otsu")
        draw_otsu_button.setObjectName("draw_otsu_button")
        draw_otsu_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        draw_otsu_button.clicked.connect(self.on_click_draw_otsu_button)
        draw_otsu_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_otsu_button)

        #-Gray
        draw_gray_button = QPushButton("Gray")
        draw_gray_button.setObjectName("draw_gray_button")
        draw_gray_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        draw_gray_button.clicked.connect(self.on_click_draw_gray_button)
        draw_gray_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_gray_button)

        #-BrdrCnst
        draw_brdr_cnst_button = QPushButton("BrdrCnst")
        draw_brdr_cnst_button.setObjectName("draw_brdr_cnst_button")
        draw_brdr_cnst_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        draw_brdr_cnst_button.clicked.connect(self.on_click_draw_brdr_cnst_button)
        draw_brdr_cnst_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_brdr_cnst_button)

        #-BrdrRplct
        draw_brdr_rplct_button = QPushButton("BrdrRplct")
        draw_brdr_rplct_button.setObjectName("draw_brdr_rplct_button")
        draw_brdr_rplct_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        draw_brdr_rplct_button.clicked.connect(self.on_click_draw_brdr_rplct_button)
        draw_brdr_rplct_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_brdr_rplct_button)

        #-Filtre
        draw_filtre_button = QPushButton("Filtre")
        draw_filtre_button.setObjectName("draw_filtre_button")
        draw_filtre_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        draw_filtre_button.clicked.connect(self.on_click_draw_filtre_button)
        draw_filtre_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_filtre_button)

        #-Filtre2d
        draw_filtre2d_button = QPushButton("Filtre2d")
        draw_filtre2d_button.setObjectName("draw_filtre2d_button")
        draw_filtre2d_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        draw_filtre2d_button.clicked.connect(self.on_click_draw_filtre2d_button)
        draw_filtre2d_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_filtre2d_button)

        #-Gamma
        draw_gamma_button = QPushButton("Gamma")
        draw_gamma_button.setObjectName("draw_gamma_button")
        draw_gamma_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        draw_gamma_button.clicked.connect(self.on_click_draw_gamma_button)
        draw_gamma_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_gamma_button)

        #-Hist
        draw_hist_button = QPushButton("Hist")
        draw_hist_button.setObjectName("draw_hist_button")
        draw_hist_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        draw_hist_button.clicked.connect(self.on_click_draw_hist_button)
        draw_hist_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_hist_button)

        #-HistEsit
        draw_hist_esit_button = QPushButton("HistEsit")
        draw_hist_esit_button.setObjectName("draw_hist_esit_button")
        draw_hist_esit_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        draw_hist_esit_button.clicked.connect(self.on_click_draw_hist_esit_button)
        draw_hist_esit_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_hist_esit_button)

        #-Sobel
        draw_sobel_button = QPushButton("Sobel")
        draw_sobel_button.setObjectName("draw_sobel_button")
        draw_sobel_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        
        draw_sobel_button.clicked.connect(self.on_click_draw_sobel_button)
        draw_sobel_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_sobel_button)

        #-Laplacian
        draw_laplacian_button = QPushButton("Laplacian")
        draw_laplacian_button.setObjectName("draw_laplacian_button")
        draw_laplacian_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        
        draw_laplacian_button.clicked.connect(self.on_click_draw_laplacian_button)
        draw_laplacian_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_laplacian_button)

        #-Canny
        draw_canny_button = QPushButton("Canny")
        draw_canny_button.setObjectName("draw_canny_button")
        draw_canny_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        
        draw_canny_button.clicked.connect(self.on_click_draw_canny_button)
        draw_canny_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_canny_button)

        #-Deriche
        draw_deriche_button = QPushButton("Deriche")
        draw_deriche_button.setObjectName("draw_deriche_button")
        draw_deriche_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        
        draw_deriche_button.clicked.connect(self.on_click_draw_deriche_button)
        draw_deriche_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_deriche_button)
         
        #-Harris
        draw_harris_button = QPushButton("Harris")
        draw_harris_button.setObjectName("draw_harris_button")
        draw_harris_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        
        draw_harris_button.clicked.connect(self.on_click_draw_harris_button)
        draw_harris_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_harris_button)

        #-FaceDetector
        draw_facedetector_button = QPushButton("FaceDetector")
        draw_facedetector_button.setObjectName("draw_facedetector_button")
        draw_facedetector_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        
        draw_facedetector_button.clicked.connect(self.on_click_draw_facedetector_button)
        draw_facedetector_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_facedetector_button)
        #-Kontür
        draw_contourdetection_button = QPushButton("ContourDetection")
        draw_contourdetection_button.setObjectName("draw_contourdetection_button")
        draw_contourdetection_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        
        draw_contourdetection_button.clicked.connect(self.on_click_draw_contourdetection_button)
        draw_contourdetection_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_contourdetection_button)

        #-Watershed
        draw_watershed_button = QPushButton("Watershed")
        draw_watershed_button.setObjectName("draw_watershed_button")
        draw_watershed_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px; border:none;border-radius: 8px;font-size:12px; color:white")
        draw_watershed_button.clicked.connect(self.on_click_draw_watershed_button)
        draw_watershed_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_watershed_button)

        #-Sec
        draw_sec_button = QPushButton("Sec")
        draw_sec_button.setObjectName("draw_sec_button")
        draw_sec_button.setStyleSheet(
            "outline:none;background-color:blue;padding:4px 4px;border:none;border-radius:8px;font-size:12px;color:white")
        
        draw_sec_button.clicked.connect(self.on_click_draw_sec_button)
        draw_sec_button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        button_layout.addWidget(draw_sec_button)





        # Charts
        #chart
        self.img_figure = plt.figure()
        img_chart = FigureCanvas(self.img_figure)
        img_toolbar = NavigationToolbar(img_chart, self)
        img_chart.setObjectName("img_chart")
        chart_layout.addWidget(img_toolbar)
        chart_layout.addWidget(img_chart)

        #options
        image_layout.setObjectName("image_layout")
        main_layout.addLayout(button_layout,0,0,1,1)
        main_layout.addLayout(chart_layout,0,1,1,1)
        main_layout.addLayout(image_layout,1,0,1,2)

        main_layout.setRowStretch(0, 2)
        main_layout.setColumnStretch(0, 2)

        # Self options
        self.setLayout(main_layout)

    @pyqtSlot()
    def on_click_draw_original_button(self):
        clearLayout(self.findChild(QHBoxLayout,"image_layout"))
        global resim_konum 
        self.im_original=QPixmap(resim_konum)
        self.label_im_original=QLabel()
        self.label_im_original.setPixmap(self.im_original)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_original)

    @pyqtSlot()
    def on_click_draw_otsu_button(self):
        global resim_konum 
        clearLayout(self.findChild(QHBoxLayout,"image_layout"))

        imageGray =cv2.imread(resim_konum,cv2.IMREAD_GRAYSCALE)
        _, otsu_threshold= cv2.threshold(imageGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = otsu_threshold.shape[:2]
        bytesPerLine = 1* w
        qimage = QImage(otsu_threshold.data, w, h, bytesPerLine, QImage.Format.Format_Grayscale8) 
        
        self.im_otsu=QPixmap.fromImage(qimage)
        self.label_im_otsu=QLabel()
        self.label_im_otsu.setPixmap(self.im_otsu)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_otsu)
    
    @pyqtSlot()
    def on_click_draw_gray_button(self):
        global resim_konum 
        clearLayout(self.findChild(QHBoxLayout,"image_layout"))

        imageGray =cv2.imread(resim_konum,cv2.IMREAD_GRAYSCALE)
        
        h, w = imageGray.shape[:2]
        bytesPerLine = 1* w
        qimage = QImage(imageGray.data, w, h, bytesPerLine, QImage.Format.Format_Grayscale8) 
        
        self.im_otsu=QPixmap.fromImage(qimage)
        self.label_im_otsu=QLabel()
        self.label_im_otsu.setPixmap(self.im_otsu)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_otsu)

    @pyqtSlot()
    def on_click_draw_brdr_cnst_button(self):
        global resim_konum 
        orjinal_resim =cv2.imread(resim_konum)
        clearLayout(self.findChild(QHBoxLayout,"image_layout"))

        border_color=(0, 0, 0)
        border_width =10
        image_with_border_constant =cv2.copyMakeBorder(orjinal_resim, border_width,border_width,border_width,border_width,borderType=cv2.BORDER_CONSTANT,value=[120,12,240])

        h, w = image_with_border_constant.shape[:2]
        bytesPerLine = 3* w
        qimage = QImage(image_with_border_constant.data, w, h, bytesPerLine, QImage.Format.Format_BGR888) 
        
        self.im_otsu=QPixmap.fromImage(qimage)
        self.label_im_otsu=QLabel()
        self.label_im_otsu.setPixmap(self.im_otsu)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_otsu)


    @pyqtSlot()
    def on_click_draw_brdr_rplct_button(self):
        global resim_konum 
        orjinal_resim =cv2.imread(resim_konum)
        clearLayout(self.findChild(QHBoxLayout,"image_layout"))

        #BORDER_REPLİCATE:yeni eklenen tüm piksellerin değeri resmin dışına en yakın piksel değeri ile belirlenir.
        border_color=(0, 0, 0)
        border_width =10
        image_with_border_replicate =cv2.copyMakeBorder(orjinal_resim, border_width,border_width,border_width,border_width,borderType=cv2.BORDER_REPLICATE,value=border_color)
        
        h, w = image_with_border_replicate.shape[:2]
        bytesPerLine = 3* w
        qimage = QImage(image_with_border_replicate.data, w, h, bytesPerLine, QImage.Format.Format_BGR888) 
        
        self.im_otsu=QPixmap.fromImage(qimage)
        self.label_im_otsu=QLabel()
        self.label_im_otsu.setPixmap(self.im_otsu)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_otsu)



    @pyqtSlot()
    def on_click_draw_filtre_button(self):
        global resim_konum 
        orjinal_resim =cv2.imread(resim_konum)

        clearLayout(self.findChild(QHBoxLayout,"image_layout"))

        kernel_size = 5
        blurred_image = cv2.blur(orjinal_resim, (kernel_size, kernel_size))
        median_blurred_image = cv2.medianBlur(orjinal_resim, kernel_size)
        box_filtered_image = cv2.boxFilter(orjinal_resim, -1, (kernel_size,kernel_size))
        bilateral_filtered_image = cv2.bilateralFilter(orjinal_resim, 9, 75, 75)
        gaussian_blurred_image = cv2.GaussianBlur(orjinal_resim, (kernel_size,kernel_size), 0)

        
        h, w = blurred_image.shape[:2]
        bytesPerLine = 3* w
        
        
        qimage = QImage(blurred_image.data, w, h, bytesPerLine, QImage.Format.Format_BGR888) 
        self.im_otsu=QPixmap.fromImage(qimage)
        self.label_im_otsu=QLabel()
        self.label_im_otsu.setPixmap(self.im_otsu)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_otsu)
        
       
        qimage = QImage(median_blurred_image.data, w, h, bytesPerLine, QImage.Format.Format_BGR888) 
        self.im_otsu=QPixmap.fromImage(qimage)
        self.label_im_otsu=QLabel()
        self.label_im_otsu.setPixmap(self.im_otsu)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_otsu)
        
        
        qimage = QImage(box_filtered_image.data, w, h, bytesPerLine, QImage.Format.Format_BGR888) 
        self.im_otsu=QPixmap.fromImage(qimage)
        self.label_im_otsu=QLabel()
        self.label_im_otsu.setPixmap(self.im_otsu)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_otsu)
        
        
        qimage = QImage(bilateral_filtered_image.data, w, h, bytesPerLine, QImage.Format.Format_BGR888) 
        self.im_otsu=QPixmap.fromImage(qimage)
        self.label_im_otsu=QLabel()
        self.label_im_otsu.setPixmap(self.im_otsu)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_otsu)
        
        qimage = QImage(gaussian_blurred_image.data, w, h, bytesPerLine, QImage.Format.Format_BGR888) 
        self.im_otsu=QPixmap.fromImage(qimage)
        self.label_im_otsu=QLabel()
        self.label_im_otsu.setPixmap(self.im_otsu)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_otsu)

    @pyqtSlot()
    def on_click_draw_filtre2d_button(self):
        global resim_konum 
        orjinal_resim =cv2.imread(resim_konum)

        clearLayout(self.findChild(QHBoxLayout,"image_layout"))

        kernel=np.array([[-1,-1,-1],[-1,8,-1]])
        #kernel1=np.float32([[-1,-1,-1],[-1,8,-1],[-1.-1,-1]])
        sonuc=cv2.filter2D(orjinal_resim,-1,kernel)

        h, w = sonuc.shape[:2]
        bytesPerLine = 3* w

        qimage = QImage(sonuc.data, w, h, bytesPerLine, QImage.Format.Format_BGR888) 
        self.im_otsu=QPixmap.fromImage(qimage)
        self.label_im_otsu=QLabel()
        self.label_im_otsu.setPixmap(self.im_otsu)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_otsu)

    @pyqtSlot()
    def on_click_draw_gamma_button(self):
        global resim_konum 
        orjinal_resim =cv2.imread(resim_konum)

        clearLayout(self.findChild(QHBoxLayout,"image_layout"))
        def apply_gamma_correction(gamma=1.0):
            image_normalized = orjinal_resim / 255.0
            gamma_corrected=np.power(image_normalized, gamma)
            gamma_corrected=np.uint8(gamma_corrected* 255)
            return gamma_corrected
        
        gamma_value = 0.5
        gamma_corrected_image =apply_gamma_correction(gamma=gamma_value)
        
        h, w = gamma_corrected_image.shape[:2]
        bytesPerLine = 3* w

        qimage = QImage(gamma_corrected_image.data, w, h, bytesPerLine, QImage.Format.Format_BGR888) 
        self.im_otsu=QPixmap.fromImage(qimage)
        self.label_im_otsu=QLabel()
        self.label_im_otsu.setPixmap(self.im_otsu)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_otsu)

    @pyqtSlot()
    def on_click_draw_hist_button(self):
        global resim_konum 
        clearLayout(self.findChild(QHBoxLayout,"image_layout"))
        image1=cv2.imread(resim_konum,cv2.IMREAD_GRAYSCALE)
        #histogram hesapla
        hist=cv2.calcHist([image1],channels=[0],mask=None,histSize=[250],ranges=[0,256])
        
        # Original()
        self.im_original=QPixmap(resim_konum)
        self.label_im_original=QLabel()
        self.label_im_original.setPixmap(self.im_original)
        self.findChild(QHBoxLayout,"image_layout").addWidget(self.label_im_original)

        # clearing old figure
        self.img_figure.clear()

        ax = self.img_figure.add_subplot(111)
        ax.plot(hist)
        ax.set_xlabel('pixel sayısı')
        ax.set_ylabel('pixel değeri')
        ax.set_title("Histogram Eğrisi")
        self.findChild(FigureCanvas, "img_chart").draw()

    @pyqtSlot()
    def on_click_draw_hist_esit_button(self):
        global resim_konum 
        clearLayout(self.findChild(QHBoxLayout,"image_layout"))
        # #Histogram eşitleme uygula
        resim=cv2.imread(resim_konum,cv2.IMREAD_GRAYSCALE)
        equalized_image=cv2.equalizeHist(resim)
        
        plt.subplot(1,2,1)
        plt.imshow(resim,cmap='gray')
        plt.title('Orjinal görüntü')
        plt.subplot(1,2,2)
        plt.imshow(resim,cmap='gray')
        plt.title('Histogram eşitleme sonrası')
        plt.show()
    @pyqtSlot()    
    def on_click_draw_sobel_button(self):
        global resim_konum
        clearLayout(self.findChild(QHBoxLayout,"image_layout"))
        image= cv2.imread(resim_konum, cv2.IMREAD_GRAYSCALE)
        #Sobel Operatörünü uygula
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobelx= np.abs(sobelx)
        sobely= np.abs(sobely)
        edges= cv2.bitwise_or(sobelx, sobely)

        height, width = edges.shape
        bytes_per_line = 1 * width
        qimage = QImage(edges.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        self.im_sobel = QPixmap.fromImage(qimage)
        self.label_im_sobel = QLabel()
        self.label_im_sobel.setPixmap(self.im_sobel)
        self.findChild(QHBoxLayout, "image_layout").addWidget(self.label_im_sobel)

    @pyqtSlot()
    def on_click_draw_laplacian_button(self):
       global resim_konum
       clearLayout(self.findChild(QHBoxLayout, "image_layout"))

       image = cv2.imread(resim_konum, cv2.IMREAD_GRAYSCALE)
       sonuc1 = cv2.Laplacian(image, cv2.CV_64F)
       sonuc1 = np.uint8(np.absolute(sonuc1))
       imgBlurred = cv2.GaussianBlur(image, (3, 3), 0)
       sonuc2 = cv2.Laplacian(imgBlurred, cv2.CV_64F, ksize=3)
       sonuc2 = np.uint8(np.absolute(sonuc2))

    
       height, width = sonuc2.shape
       bytes_per_line = 1 * width
       qimage = QImage(sonuc2.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
       self.im_laplacian = QPixmap.fromImage(qimage)
       self.label_im_laplacian = QLabel()
       self.label_im_laplacian.setPixmap(self.im_laplacian)
       self.findChild(QHBoxLayout, "image_layout").addWidget(self.label_im_laplacian)

    @pyqtSlot()
    def on_click_draw_canny_button(self):
        global resim_konum
        clearLayout(self.findChild(QHBoxLayout,"image_layout"))  
        image = cv2.imread(resim_konum, cv2.IMREAD_GRAYSCALE)
        low_threshold = 50
        high_threshold = 150
        sonuc = cv2.Canny(image, low_threshold, high_threshold, L2gradient=True)
        
        height, width = sonuc.shape
        bytes_per_line = 1 * width

        qimage = QImage(sonuc.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        self.im_canny = QPixmap.fromImage(qimage)
        self.label_im_canny = QLabel()
        self.label_im_canny.setPixmap(self.im_canny)
        self.findChild(QHBoxLayout, "image_layout").addWidget(self.label_im_canny)
   
    @pyqtSlot()
    def on_click_draw_deriche_button(self):
        global resim_konum
        clearLayout(self.findChild(QHBoxLayout,"image_layout"))  
        image= cv2.imread(resim_konum, cv2.IMREAD_GRAYSCALE)
        #Deriche filteresi için kernel oluştur
        alpha = 0.5 #Deriche filtresi parametresi
        kernel_size = 3 # Filtre boyutu
        kx, ky = cv2.getDerivKernels(1, 1, kernel_size, normalize=True)
        deriche_kernel_x = alpha * kx
        deriche_kernel_y= alpha * ky
        deriche_x = cv2.filter2D(image, -1, deriche_kernel_x)
        deriche_y = cv2.filter2D(image, -1, deriche_kernel_y)
        edges = np.sqrt(deriche_x**2 + deriche_y**2)
        f,eksen= plt.subplots(1,2, figsize = (17,7))
        eksen[0].imshow(image, cmap="gray")
        eksen[1].imshow(edges, cmap="gray")

        height, width = edges.shape
        bytes_per_line = 1 * width
        qimage = QImage(edges.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        self.im_deriche = QPixmap.fromImage(qimage)
        self.label_im_deriche = QLabel()
        self.label_im_deriche.setPixmap(self.im_deriche)
        self.findChild(QHBoxLayout, "image_layout").addWidget(self.label_im_deriche)
    @pyqtSlot()
    def on_click_draw_harris_button(self):
        global resim_konum
        clearLayout(self.findChild(QHBoxLayout,"image_layout"))
        img = cv2.imread(resim_konum)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corner_quality = 0.04
        min_distance = 10
        block_size = 3
        corners = cv2.cornerHarris(gray, block_size, 3, corner_quality)
        corners = cv2.dilate(corners, None)
        img[corners > 0.01 * corners.max()] = [0, 0, 255]

        height, width, channel = img.shape 
        bytes_per_line = 3 * width
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.im_harris = QPixmap.fromImage(qimage)
        self.label_im_harris = QLabel()
        self.label_im_harris.setPixmap(self.im_harris)
        self.findChild(QHBoxLayout, "image_layout").addWidget(self.label_im_harris)
    @pyqtSlot()
    def on_click_draw_facedetector_button(self):
        faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        conf_1=cv2.imread(resim_konum,0)
        conf_2=conf_1.copy()

        faces_1=faceCascade.detectMultiScale(conf_1)
        for(x,y,w,h) in faces_1:
         cv2.rectangle(conf_1,(x,y),(x+w,y+h),(255,0,0),10)
        faces_2=faceCascade.detectMultiScale(conf_2,scaleFactor=1.3,minNeighbors=6)
        for(x,y,w,h) in faces_2:
         cv2.rectangle(conf_2,(x,y),(x+w,y+h),(255,0,0),10)
        f,eksen=plt.subplots(1,2,figsize=(20,10))
        eksen[0].imshow(conf_1,cmap="gray")
        eksen[1].imshow(conf_2,cmap="gray")

        height, width = conf_1.shape
        bytes_per_line = 1 * width
        qimage = QImage(conf_1.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        self.im_facedetector = QPixmap.fromImage(qimage)
        self.label_im_facedetector = QLabel()
        self.label_im_facedetector.setPixmap(self.im_facedetector)
        self.findChild(QHBoxLayout, "image_layout").addWidget(self.label_im_facedetector)
        


    def on_click_draw_contourdetection_button(self):
        img = cv2.imread(resim_konum)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        height, width, channel = img.shape
        bytes_per_line = 3 * width
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        self.label_im_contourdetection = QLabel()
        self.label_im_contourdetection.setPixmap(pixmap)
        self.findChild(QHBoxLayout, "image_layout").addWidget(self.label_im_contourdetection)

    def on_click_draw_watershed_button(self):
        imgOrj = cv2.imread(resim_konum)
        imgBlr = cv2.medianBlur(imgOrj, 31)
        imgGray = cv2.cvtColor(imgBlr, cv2.COLOR_BGR2GRAY)
        ret, imgTH = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        imgOPN = cv2.morphologyEx(imgTH, cv2.MORPH_OPEN, kernel, iterations=7)
        sureBG = cv2.dilate(imgOPN, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(imgOPN, cv2.DIST_L2, 5)
        _, sureFG = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sureFG = np.uint8(sureFG)
        unknown = cv2.subtract(sureBG, sureFG)

    # Etiketleme işlemi
        ret, markers = cv2.connectedComponents(sureFG, labels=5)

    # Bilinmeyen pikselleri etiketle
        markers = markers + 1
        markers[unknown == 255] = 0

    # Watershed algoritması uygula
        markers = cv2.watershed(imgOrj, markers)
    
        imgCopy = imgOrj.copy()
        contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
         if hierarchy[0][i][3] == -1:
            cv2.drawContours(imgCopy, contours, i, (255, 0, 0), 5)

        f, eksen = plt.subplots(3, 3, figsize=(30, 30))
        eksen[0, 0].imshow(imgOrj)
        eksen[0, 1].imshow(imgBlr)
        eksen[0, 2].imshow(imgGray, cmap='gray')
        eksen[1, 0].imshow(imgTH, cmap='gray')
        eksen[1, 1].imshow(imgOPN, cmap='gray')
        eksen[1, 2].imshow(sureBG, cmap='gray')
        eksen[2, 0].imshow(dist_transform, cmap='gray')
        eksen[2, 1].imshow(sureFG, cmap='gray')
        eksen[2, 2].imshow(imgCopy)
        plt.show()
    
        height, width, channel = imgOrj.shape
        bytes_per_line = 3 * width  # Adjust for BGR format
        qimage = QImage(imgOrj.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        self.label_im_watershed = QLabel()
        self.label_im_watershed.setPixmap(pixmap)
        self.findChild(QHBoxLayout, "image_layout").addWidget(self.label_im_watershed)


   
    
    @pyqtSlot()
    def on_click_draw_sec_button(self):
        global resim_konum 
        # dir_ =QFileDialog.getExistingDirectory(None, 'Select a folder:', 'C:\\', QFileDialog.ReadOnly)
        # print(dir_)
        fname=QFileDialog.getOpenFileNames(self, 'Resim Aç', 'c:\\',"Image files (*.jpg)")
        
        try:
            resim_konum=fname[0][0]
        except:
            print("file select error")

def main():
    app = QApplication(sys.argv)
    ex = mainWidget()
    # opening window in maximized size
    ex.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()