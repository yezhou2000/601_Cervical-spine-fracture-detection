import sys

<<<<<<< Updated upstream
from PyQt6.QtWidgets import QApplication, QWidget,QMainWindow, QFileDialog
=======
sys.path.append("..")
sys.path.append("../Conv3D")
from PyQt6.QtWidgets import QApplication, QWidget,QMainWindow, QFileDialog, QTableWidget,QTableWidgetItem,QVBoxLayout
>>>>>>> Stashed changes
from PyQt6.QtCore import Qt

from PyQt6 import uic
from pathlib import Path
#from Conv3D.test import predict_fracture

# name=["1","2","3","4","5","6","7","8"]
rates=[("Red", "1"),
          ("Green", "2"),
          ("Blue", "3"),
          ("Black", "#4"),
          ("White", "#5"),
          ("Electric Green", "6"),
          ("Dark Blue", "7"),
          ("Yellow", "#8")]

def showTable(self):
		# app1 = QApplication()
		# table = QTableWidget()
		self.tableWidget.setRowCount(8)
		self.tableWidget.setColumnCount(2)
		self.tableWidget.setHorizontalHeaderLabels(["name", "rate"])
		
		for i, (name, rate) in enumerate(rates):
			item_name = QTableWidgetItem(name)
			item_rate = QTableWidgetItem(rate)
			self.tableWidget.setItem(i, 0, item_name)
			self.tableWidget.setItem(i, 1, item_rate)
		# table.show()
		# sys.exit(app1.exec())

class MyAPP(QMainWindow):
	def __init__(self):
		super().__init__()
		uic.loadUi('/Users/zhouye/Documents/GitHub/601_Cervical-spine-fracture-detection/code/GUI/untitled.ui',self)
		self.input_pushButton.clicked.connect(self.openFileNamesDialog)
		self.output_pushButton.clicked.connect(self.predict)
		self.predict_list = []

	def openFileNamesDialog(self):
		home_dir = str(Path.home())
		fname = QFileDialog.getOpenFileName(self, 'Open file', home_dir)

		if fname:
			self.textBrowser.append(fname[0].split('/')[-1])
			print(fname)
			self.predict_list.append(fname[0])
	
	
			

	def predict(self):

		showTable(self)
		if len(self.predict_list) == 0:
			return
		# for file_name in self.predict_list:
		# 	#result = predict_fracture(self.predict_list)
		# 	#self.textBrowser_2.append(result.to_string())






def main(args):
    app = QApplication(args)

    myapp = MyAPP()	
    myapp.show()

    sys.exit(app.exec())

 
if __name__=="__main__":
    main(sys.argv)