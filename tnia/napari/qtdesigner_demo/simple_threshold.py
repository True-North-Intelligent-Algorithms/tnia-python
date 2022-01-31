# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'simple_threshold.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        self.pushButtonThreshold = QtWidgets.QPushButton(Form)
        self.pushButtonThreshold.setGeometry(QtCore.QRect(140, 100, 75, 23))
        self.pushButtonThreshold.setObjectName("pushButtonThreshold")
        self.radioButtonGlobal = QtWidgets.QRadioButton(Form)
        self.radioButtonGlobal.setGeometry(QtCore.QRect(140, 40, 82, 17))
        self.radioButtonGlobal.setChecked(True)
        self.radioButtonGlobal.setObjectName("radioButtonGlobal")
        self.radioButtonAdaptive = QtWidgets.QRadioButton(Form)
        self.radioButtonAdaptive.setGeometry(QtCore.QRect(140, 60, 82, 17))
        self.radioButtonAdaptive.setObjectName("radioButtonAdaptive")
        self.sliderThreshold = QtWidgets.QSlider(Form)
        self.sliderThreshold.setGeometry(QtCore.QRect(100, 150, 160, 22))
        self.sliderThreshold.setOrientation(QtCore.Qt.Horizontal)
        self.sliderThreshold.setObjectName("sliderThreshold")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButtonThreshold.setText(_translate("Form", "Threshold"))
        self.radioButtonGlobal.setText(_translate("Form", "Global"))
        self.radioButtonAdaptive.setText(_translate("Form", "Adaptive"))

