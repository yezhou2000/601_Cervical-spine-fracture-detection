import sys

from PyQt6.QtWidgets import QApplication, QWidget,QMainWindow, QFileDialog
from PyQt6.QtCore import Qt
from PyQt6 import uic
from pathlib import Path
from Conv3D.test import predict_fracture


class MyAPP(QMainWindow):
	def __init__(self):
		super().__init__()
		uic.loadUi('./untitled.ui',self)
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
		if len(self.predict_list) == 0:
			return
		for file_name in self.predict_list:
			result = predict_fracture(self.predict_list)
			self.textBrowser_2.append(result.to_string())

if __name__ == '__main__':
	app = QApplication(sys.argv)
	
	myapp = MyAPP()


	myapp.show()

	sys.exit(app.exec())