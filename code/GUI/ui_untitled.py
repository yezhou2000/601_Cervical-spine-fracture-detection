# Form implementation generated from reading ui file '/Users/zhouye/Documents/GitHub/601_Cervical-spine-fracture-detection/code/GUI/untitled.ui'
#
# Created by: PyQt6 UI code generator 6.4.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(973, 595)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.bg_frame = QtWidgets.QFrame(self.centralwidget)
        self.bg_frame.setStyleSheet("background-color: rgba(0, 143, 189, 89);\n"
"font: 14pt \"American Typewriter\";")
        self.bg_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.bg_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.bg_frame.setObjectName("bg_frame")
        self.input_frame = QtWidgets.QFrame(self.bg_frame)
        self.input_frame.setGeometry(QtCore.QRect(30, 80, 300, 350))
        self.input_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.input_frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.input_frame.setObjectName("input_frame")
        self.textBrowser = QtWidgets.QTextBrowser(self.input_frame)
        self.textBrowser.setGeometry(QtCore.QRect(20, 20, 256, 311))
        self.textBrowser.setObjectName("textBrowser")
        self.input_pushButton = QtWidgets.QPushButton(self.bg_frame)
        self.input_pushButton.setGeometry(QtCore.QRect(190, 460, 100, 30))
        self.input_pushButton.setMouseTracking(False)
        self.input_pushButton.setObjectName("input_pushButton")
        self.frame = QtWidgets.QFrame(self.bg_frame)
        self.frame.setGeometry(QtCore.QRect(430, 80, 461, 350))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setObjectName("frame")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.frame)
        self.textBrowser_2.setGeometry(QtCore.QRect(20, 20, 421, 311))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.output_pushButton = QtWidgets.QPushButton(self.bg_frame)
        self.output_pushButton.setGeometry(QtCore.QRect(580, 460, 150, 30))
        self.output_pushButton.setObjectName("output_pushButton")
        self.toolButton = QtWidgets.QToolButton(self.bg_frame)
        self.toolButton.setGeometry(QtCore.QRect(50, 460, 81, 31))
        self.toolButton.setObjectName("toolButton")
        self.label = QtWidgets.QLabel(self.bg_frame)
        self.label.setGeometry(QtCore.QRect(330, 20, 321, 51))
        font = QtGui.QFont()
        font.setFamily("American Typewriter")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.bg_frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 973, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_file = QtGui.QAction(MainWindow)
        self.actionOpen_file.setObjectName("actionOpen_file")
        self.actionSave_file = QtGui.QAction(MainWindow)
        self.actionSave_file.setObjectName("actionSave_file")
        self.actionExport = QtGui.QAction(MainWindow)
        self.actionExport.setObjectName("actionExport")
        self.menuFile.addAction(self.actionOpen_file)
        self.menuFile.addAction(self.actionSave_file)
        self.menuFile.addAction(self.actionExport)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.output_pushButton, self.toolButton)
        MainWindow.setTabOrder(self.toolButton, self.input_pushButton)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.input_pushButton.setText(_translate("MainWindow", "Load file"))
        self.textBrowser_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'American Typewriter\'; font-size:14pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.output_pushButton.setText(_translate("MainWindow", "Run Prediction"))
        self.toolButton.setText(_translate("MainWindow", "Clear"))
        self.label.setText(_translate("MainWindow", "Cervical Fracture Detection Tool"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen_file.setText(_translate("MainWindow", "Open file"))
        self.actionSave_file.setText(_translate("MainWindow", "Save file"))
        self.actionExport.setText(_translate("MainWindow", "Export"))