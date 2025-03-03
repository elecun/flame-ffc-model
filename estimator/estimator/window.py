'''
Windows
@Author Byunghun Hwang<bh.hwang@iae.re.kr>
'''

import os, sys
import pathlib
import queue
import time
import numpy as np
from datetime import datetime
import random
import json
import pandas as pd
import joblib

try:
    # using PyQt5
    from PyQt5.QtGui import QImage, QPixmap, QCloseEvent, QStandardItem, QStandardItemModel
    from PyQt5.QtWidgets import QApplication, QFrame, QMainWindow, QLabel, QPushButton, QMessageBox
    from PyQt5.QtWidget import QProgressBar, QFileDialog, QComboBox, QLineEdit, QSlider, QCheckBox, QComboBox
    from PyQt5.uic import loadUi
    from PyQt5.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal
except ImportError:
    # using PyQt6
    from PyQt6.QtGui import QImage, QPixmap, QCloseEvent, QStandardItem, QStandardItemModel
    from PyQt6.QtWidgets import QApplication, QFrame, QMainWindow, QLabel, QPushButton, QCheckBox, QComboBox
    from PyQt6.QtWidgets import QMessageBox, QProgressBar, QFileDialog, QComboBox, QLineEdit, QSlider, QVBoxLayout
    from PyQt6.uic import loadUi
    from PyQt6.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal
    
from util.logger.console import ConsoleLogger
from tensorflow.keras import models

class AppWindow(QMainWindow):
    def __init__(self, config:dict):
        """ initialization """
        super().__init__()
        
        self.__console = ConsoleLogger.get_logger() # logger
        self.__config = config  # copy configuration data

        self.__current_scaler = joblib.load("ampere.joblib")
        self.__voltage_scaler = joblib.load("voltage.joblib")
        self.__velocity_scaler = joblib.load("velocity.joblib")
        self.__temperature_scaler = joblib.load("temperature.joblib")
        self.__copper_sulfate_scaler = joblib.load("cs.joblib")
        self.__sulfuric_acid_scaler = joblib.load("sa.joblib")
        self.__brightener_scaler = joblib.load("brightener.joblib")
        self.__carrier_scaler = joblib.load("carrier.joblib")
        self.__optime_scaler = joblib.load("optime.joblib")

        try:            
            if "gui" in config:

                # load UI File
                ui_path = pathlib.Path(config["app_path"]) / config["gui"]
                if os.path.isfile(ui_path):
                    loadUi(ui_path, self)
                else:
                    raise Exception(f"Cannot found UI file : {ui_path}")
                
                
                # register button event callback function
                self.btn_estimate.clicked.connect(self.on_btn_estimate)

        except Exception as e:
            self.__console.error(f"{e}")

    def clear_all(self):
        """ clear graphic view """
        try:
            pass
        except Exception as e:
            self.__console.error(f"{e}")

    def on_btn_estimate(self):
        """ initialize all """
        model = models.load_model(filepath="mvnn.keras")
        model.summary()

        optime = self.__optime_scaler.transform(np.array([[float(self.edit_optime.text())]])).flatten()[0]
        voltage = self.__voltage_scaler.transform(np.array([[float(self.edit_voltage.text())]])).flatten()[0]
        current = self.__current_scaler.transform(np.array([[float(self.edit_current.text())]])).flatten()[0]
        temperature = self.__temperature_scaler.transform(np.array([[float(self.edit_temperature.text())]])).flatten()[0]
        copper_sulfate = self.__copper_sulfate_scaler.transform(np.array([[float(self.edit_copper_sulfate.text())]])).flatten()[0]
        sulfuric_acid = self.__sulfuric_acid_scaler.transform(np.array([[float(self.edit_sulfuric_acid.text())]])).flatten()[0]
        velocity = self.__velocity_scaler.transform(np.array([[float(self.edit_velocity.text())]])).flatten()[0]

        # sample : 8.0, 2.1, 41, 26.7, 143.5, 132.79, 0.06

        input = [optime, voltage, current, temperature, copper_sulfate, sulfuric_acid, velocity]

        input = pd.DataFrame(data=[input], columns=['optime', 'voltage', 'Ampere', "Temperature", "copper_sulfate", "sulfuric_acid", "Velocity"])
        result = model.predict(input).ravel()

        brightener = self.__brightener_scaler.inverse_transform([[result[0]]]).flatten()[0]
        carrier = self.__carrier_scaler.inverse_transform([[result[1]]]).flatten()[0]
        print("brightener : ", brightener)
        print("carrier : ", carrier)

        self.edit_carrier.setText(str(int(carrier)))
        self.edit_brightener.setText(str(int(brightener)))
        
