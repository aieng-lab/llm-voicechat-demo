# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets

class ChatWindow(QtWidgets.QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout()
        self.text = QtWidgets.QTextBrowser(parent=self)
        layout.addWidget(self.text)
        self.setLayout(layout)
        
class Ui_MainWindow(object):        
    
    def setupUi(self, MainWindow, window_size):
        MainWindow.setObjectName("MainWindow")
        self.window_size = (window_size.width(), window_size.height())
        self.status_size = int(0.03 * self.window_size[0])
        self.button_text_size = int(0.5 * self.status_size)
        MainWindow.resize(self.window_size[0], self.window_size[1])
        
        #Main widget (background)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background-color: rgb(0, 0, 0);border: 0px;")
        self.centralwidget.setObjectName("centralwidget")
        #Main layout
        self.gridLayout_1 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_1.setObjectName("gridLayout_1")
        
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(self.window_size[0], #width
                                           int(0.2 * self.window_size[1]), #height
                                           QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 1, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        
        
        self.gridLayout_3= QtWidgets.QGridLayout()
        self.gridLayout_3.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout_3.setObjectName("gridLayout_3")
        
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setObjectName("label_6")
        ###
        self.label_6.setStyleSheet(f''' font-size: {self.status_size}px; color: Red;''')
        # self.label_6.setIndent(60)
        ###
        self.gridLayout_3.addWidget(self.label_6, 2, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_3)
        
        self.gridLayout_2.addWidget(self.groupBox, 3, 1, 1, 1)
        
        self.groupBox1 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox1.setObjectName("groupBox1")
        
        self.verticalLayout1 = QtWidgets.QVBoxLayout(self.groupBox1)
        self.verticalLayout1.setObjectName("verticalLayout1")
        
        self.buttonsLayout = QtWidgets.QVBoxLayout()
        self.buttonsLayout.setObjectName("buttonsLayout")
        ###
        #Left, above, right, under
        self.buttonsLayout.setContentsMargins(int(0.1 * self.window_size[0]), int(0.05 * self.window_size[1]), int(0.1 * self.window_size[0]), 0)
        ###
        
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        
        self.startButton = QtWidgets.QPushButton(self.groupBox1)
        self.startButton.setObjectName("startButton")
        ###
        self.startButton.setStyleSheet(f'background-color: #555555; font-size: {self.button_text_size}px; color: white;')
        ###

        self.gridLayout_4.addWidget(self.startButton, 0, 0, 1, 1)
        
        self.resetButton = QtWidgets.QPushButton(self.groupBox1)
        self.resetButton.setObjectName("resetButton")
        ###
        self.resetButton.setStyleSheet(f'background-color: #555555; font-size: {self.button_text_size}px; color: white;')
        ###
        
        self.gridLayout_4.addWidget(self.resetButton, 0, 1, 1, 1)
        
        self.chatWindow = ChatWindow()
        self.chatButton = QtWidgets.QPushButton(self.groupBox1)
        self.chatButton.setStyleSheet(f'background-color: #555555; font-size: {self.button_text_size}px; color: white;')
        
        self.gridLayout_5.addWidget(self.chatButton, 0, 0, 1, 1)
        
        self.buttonsLayout.addLayout(self.gridLayout_4)
        self.buttonsLayout.addLayout(self.gridLayout_5)
        
        self.verticalLayout1.addLayout(self.buttonsLayout)
        
        self.gridLayout_2.addWidget(self.groupBox1, 4, 1, 1, 1)
        
        
        self.gridLayout_1.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Alvi - Das freundliche, Passauer KI Model"))
        self.label_6.setText(_translate("MainWindow", "Ich schlafe  ..."))
        self.startButton.setText(_translate("MainWindow", "Starte Gespräch"))
        self.resetButton.setText(_translate("MainWindow", "Beende Gespräch"))
        self.chatButton.setText(_translate("MainWindow", "Zeig Gespräch an"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
