"""
GUI_plot.py for Big Data Analytics A.A. 2021/22 Final Exam GUI by The Missing Values team
"""
import os
import sys
import pandas as pd
import numpy as np
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from joblib import load
import shap
from PyQt5.QtWidgets import *
# noinspection PyUnresolvedReferences
from PyQt5.uic import loadUi
# DiCE imports
import dice_ml


def resource_path(folder_name):
    """
    PyInstaller creates a temp folder and stores path in _MEIPASS
    :param folder_name: the name of the folder looking for
    :return:
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(folder_name)

    return base_path


class MatplotlibWidget(QMainWindow):
    """
    telemetry plotting widget
    """
    def __init__(self):
        # noinspection PyArgumentList
        QMainWindow.__init__(self)
        self.y_cls = None
        self.X_cls = None
        self.data_cls = None
        self.gbc_shap_values = None
        self.gbc_explainer = None
        self.eda_df = None
        self.data_reg = None
        self.y_reg = None
        self.X_reg = None
        self.idx = -1
        self.gbr_shap_values = None
        loadUi(os.path.join(resource_path("uis"), "plot.ui"), self)
        self.setWindowTitle("Big Data Analytics A.A. 2021/22 Final Exam GUI by The Missing Values team")
        self.pushButton_load.clicked.connect(self.pushbutton_handler)
        self.pushButton_next.clicked.connect(self.next_handler)
        self.pushButton_home.clicked.connect(self.home_handler)
        self.BtnDescribe.clicked.connect(self.dataHead)
        self.CF_Button.toggled.connect(self.show_cf)
        self.pushButton_previous.clicked.connect(self.previous_handler)
        self.comboBox_task.activated[str].connect(self.plot)
        self.comboBox_plot.activated[str].connect(self.plot)
        self.spinBox_3.valueChanged.connect(self.show_cf)
        self.gbr_rfe_support = np.load(os.path.join(resource_path("models"), 'gbr_rfe_support.npy'))
        self.gbr = load(os.path.join(resource_path("models"), 'GradientBoostingRegressor.joblib'))
        self.gbr_explainer = shap.TreeExplainer(self.gbr)
        self.gbc_rfe_support = np.load(os.path.join(resource_path("models"), 'gbc_rfe_support.npy'))
        self.gbc = load(os.path.join(resource_path("models"), 'GradientBoostingClassifier.joblib'))
        self.gbc_explainer = shap.TreeExplainer(self.gbc)
        self.tabWidget.setCurrentIndex(1)
        self.home_handler()

    def next_handler(self):
        """
        Handle the next button pressed event
        """
        print("Next button pressed")
        if self.data_reg is None:
            self.label_text.setText(f"Please load the data first !")
            self.label_text.setFont(QFont('Times', 30))
        else:
            self.idx = (self.idx + 1) % len(self.gbr_shap_values.values)
            self.plot()

    def home_handler(self):
        """
        Handle the home button pressed event
        """
        pixmap = QPixmap(os.path.join(resource_path("imgs"), 'MoviesLens.png'))
        self.label_plot.setPixmap(pixmap.scaled(1031, 357, Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation))
        self.label_plot.setScaledContents(True)
        self.label_text.setText(f"Data Analytics on Movies Dataset")
        self.label_text.setFont(QFont('Times', 30))
        self.show()

    def previous_handler(self):
        """
        Handle the previous button pressed event
        """
        print("Previous button pressed")
        if self.data_reg is None:
            self.label_text.setText(f"Please load the data first !")
            self.label_text.setFont(QFont('Times', 30))
        else:
            self.idx = (self.idx - 1) % len(self.gbr_shap_values.values)
            self.plot()

    def pushbutton_handler(self):
        """
        Handle the dialog button pressed event
        """
        print("Dialog button pressed")
        self.open_dialog_box()

    def open_dialog_box(self):
        """
        Open a dialog box to load the data
        :return:
        """
        try:
            self.label_text.setText(f"Loading and Analyzing the data ...")
            self.label_text.setFont(QFont('Times', 30))
            filename = QFileDialog.getOpenFileNames()
            if filename[1] == '':
                self.label_text.setText(f"Empty File ! Please load the correct data !")
                self.label_text.setFont(QFont('Times', 30))
                return
            path = filename[0][0]
            df = pd.read_csv(path, index_col=0)
        except:
            print("path : ", filename[0])
            self.label_text.setText(f"Wrong File ! Please load the correct data !")
            self.label_text.setFont(QFont('Times', 30))
            return
        print("Data loaded")
        df['profitability'] = df['return_log'].apply(lambda x: 1 if x >= 0 else 0)
        self.X_reg, self.y_reg = df.drop(['return_log', 'profitability'], axis=1).loc[:, self.gbr_rfe_support], df['return_log']
        self.data_reg = self.X_reg.copy()
        self.data_reg.loc[:, 'predict_reg'] = np.round(self.gbr.predict(self.data_reg), 2)
        self.eda_df = pd.read_csv(os.path.join(resource_path("inps"), 'eda_df.csv')).iloc[df.index, [11, 0, 18, 6, 9, 20, 1, 8]]  # we need this dataset to show the features encode with clustering
        self.X_cls, self.y_cls = df.drop(['return_log', 'profitability'], axis=1).loc[:, self.gbc_rfe_support], df['profitability']
        self.data_cls = self.X_cls.copy()
        self.data_cls.loc[:, 'predict_cls'] = self.gbc.predict(self.data_cls)
        self.spinBox_1.setValue(len(self.data_reg.index))
        self.spinBox_2.setValue(len(self.data_reg.columns))
        self.gbr_shap_values = self.gbr_explainer(self.data_reg)
        self.gbc_explainer = shap.Explainer(self.gbc.predict, self.X_cls)
        self.gbc_shap_values = self.gbc_explainer.shap_values(self.X_cls)
        print("Model applied")
        shap.initjs()
        fig = shap.plots.heatmap(self.gbr_shap_values, show=False)
        fig.savefig(os.path.join(resource_path("imgs"), 'regression_heatmap.png'), dpi=150, bbox_inches='tight')
        fig.clf()
        fig = shap.plots.heatmap(self.gbc_shap_values, show=False)
        fig.savefig(os.path.join(resource_path("imgs"), 'classification_heatmap.png'), dpi=150, bbox_inches='tight')
        fig.clf()
        for i in range(len(self.gbr_shap_values.values)):
            fig = shap.plots._waterfall.waterfall_legacy(self.gbr_shap_values.base_values[0][0], self.gbr_shap_values.values[i], features=self.gbr_shap_values.data[i], feature_names=self.X_reg.columns, show=False)
            fig.savefig(os.path.join(resource_path("imgs"), f'regression_waterfall_{i}.png'), dpi=150, bbox_inches='tight')
            fig.clf()
        for i in range(len(self.gbc_shap_values.values)):
            fig = shap.plots._waterfall.waterfall_legacy(self.gbc_shap_values.base_values[i], self.gbc_shap_values.values[i], features=self.gbc_shap_values.data[i], feature_names=self.X_cls.columns, show=False)
            fig.savefig(os.path.join(resource_path("imgs"), f'classification_waterfall_{i}.png'), dpi=150, bbox_inches='tight')
            fig.clf()
        for i in range(len(self.gbr_shap_values.values)):
            fig = shap.force_plot(self.gbr_explainer.expected_value, self.gbr_shap_values.values[i], self.data_reg.iloc[i], matplotlib=True, show=False)
            fig.savefig(os.path.join(resource_path("imgs"), f'regression_force_{i}.png'), dpi=150, bbox_inches='tight')
            fig.clf()
        for i in range(len(self.gbc_shap_values.values)):
            fig = shap.force_plot(self.gbc_shap_values.base_values[i], self.gbc_shap_values.values[i], self.X_cls.iloc[i], matplotlib=True, show=False)
            fig.savefig(os.path.join(resource_path("imgs"), f'classification_force_{i}.png'), dpi=150, bbox_inches='tight')
            fig.clf()
        for i in range(len(self.gbr_shap_values.values)):
            fig = shap.plots.bar(self.gbr_shap_values[i], show=False)
            fig.savefig(os.path.join(resource_path("imgs"), f'regression_bar_{i}.png'), dpi=150, bbox_inches='tight')
            fig.clf()
        for i in range(len(self.gbc_shap_values.values)):
            fig = shap.plots.bar(self.gbc_shap_values[i], show=False)
            fig.savefig(os.path.join(resource_path("imgs"), f'classification_bar_{i}.png'), dpi=150, bbox_inches='tight')
            fig.clf()
        print("Images generated")
        self.idx = 0
        self.plot()
        self.dataHead()
        self.tabWidget.setCurrentIndex(0)

    def show_cf(self):
        """
        Generate counterfactuals
        Handle the CF button checked event and spin box 3 value changed event
        """
        if self.CF_Button.isChecked():
            if self.comboBox_task.currentText() == "Classification":
                if self.data_reg is not None:
                    cf_idx = self.spinBox_3.value()
                    if cf_idx >= len(self.data_cls.index):
                        self.spinBox_3.setValue(0)
                        cf_idx = 0

                    # Pre-defined Data frame
                    d = dice_ml.Data(dataframe=self.data_cls,
                                     continuous_features=['budget_tmdb', 'N_spoken_languages', 'month_sin', 'month_cos',
                                                          'runtime', 'year'], outcome_name='predict_cls')
                    # Pre-trained ML model
                    m = dice_ml.Model(model=self.gbc, backend='sklearn')
                    # DiCE explanation instance
                    exp = dice_ml.Dice(d, m, method="random")
                    dice_exp = exp.generate_counterfactuals(
                        self.data_cls.iloc[[cf_idx]].iloc[:, :-1],
                        total_CFs=5,
                        desired_class="opposite",
                        random_seed=42,
                        verbose=False,
                        features_to_vary=self.data_cls.iloc[:, :-1][
                            self.data_cls.iloc[:, :-1].columns.difference(
                                ['year', 'month_sin', 'month_cos'])].columns.to_list())
                    cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                    ori = pd.concat([self.data_cls.iloc[[cf_idx]]]*5, ignore_index=True)
                    cf_df = cf_df.compare(ori, keep_shape=True).dropna(axis=1, how='all').replace(np.nan, '-').reset_index(drop=True)
                    NumRows = len(cf_df.index)
                    numColomn = len(cf_df.columns)
                    self.tableWidget_2.setRowCount(NumRows)
                    self.tableWidget_2.setColumnCount(numColomn)
                    self.tableWidget_2.setHorizontalHeaderLabels(['_'.join(col) for col in cf_df.columns])
                    for i in range(NumRows):
                        for j in range(numColomn):
                            self.tableWidget_2.setItem(i, j, QTableWidgetItem(str(cf_df.iat[i, j])))
                    self.tableWidget_2.resizeColumnsToContents()
                    self.tableWidget_2.resizeRowsToContents()
                    self.spinBox_1.setValue(len(self.data_reg.index))
                    self.spinBox_2.setValue(len(self.data_reg.columns))
                    self.dataHead()
                else:
                    self.CF_Button.setChecked(False)
                    self.spinBox_3.setValue(0)
                    self.label_text.setText(f"Please load the data first !")
                    self.label_text.setFont(QFont('Times', 30))
                    self.tabWidget.setCurrentIndex(1)
            else:
                self.CF_Button.setChecked(False)
                self.spinBox_3.setValue(0)
                self.label_text.setText(f"Please Choose the Classification task !")
                self.label_text.setFont(QFont('Times', 30))
                self.tableWidget_2.setRowCount(0)
                self.tableWidget_2.setColumnCount(0)
                self.tabWidget.setCurrentIndex(1)
        else:
            self.tableWidget_2.setRowCount(0)
            self.tableWidget_2.setColumnCount(0)
            self.spinBox_3.setValue(0)

    def dataHead(self):
        """
        Demonstrate the loaded data and describe the data with eda_df.csv
        """
        if self.data_reg is not None and self.data_cls is not None:
            if self.comboBox_task.currentText() == "Regression":
                data = self.data_reg
            else:
                data = self.data_cls

            NumRows = self.spinBox_1.value()
            numColomn = self.spinBox_2.value()
            if NumRows == 0 or NumRows > len(data.index):
                self.spinBox_1.setValue(len(data.index))
                NumRows = len(data.index)
            if numColomn == 0 or numColomn > len(data.columns):
                self.spinBox_2.setValue(len(data.columns))
                numColomn = len(data.columns)

            self.tableWidget.setColumnCount(numColomn)
            self.tableWidget_1.setColumnCount(min(numColomn, len(self.eda_df.columns)))
            self.tableWidget.setRowCount(NumRows)
            self.tableWidget_1.setRowCount(NumRows)
            self.tableWidget.setHorizontalHeaderLabels(data.columns)
            self.tableWidget_1.setHorizontalHeaderLabels(self.eda_df.columns)

            for i in range(NumRows):
                for j in range(numColomn):
                    self.tableWidget.setItem(i, j, QTableWidgetItem(str(data.iat[i, j])))

            for i in range(NumRows):
                for j in range(min(numColomn, len(self.eda_df.columns))):
                    self.tableWidget_1.setItem(i, j, QTableWidgetItem(str(self.eda_df.iat[i, j])))

            self.tableWidget.resizeColumnsToContents()
            self.tableWidget_1.resizeColumnsToContents()
            self.tableWidget.resizeRowsToContents()
            self.tableWidget_1.resizeRowsToContents()
        else:
            self.label_text.setText(f"Please load the data first !")
            self.label_text.setFont(QFont('Times', 30))
            self.tabWidget.setCurrentIndex(1)

    def plot(self):
        """
        Plot the images generated by shap
        Handle the comboBox_task and comboBox_plot change event
        """
        if self.data_reg is None and self.data_cls is None:
            self.label_text.setText(f"Please load the data first !")
            self.label_text.setFont(QFont('Times', 30))
        else:
            if self.comboBox_task.currentText() == "Regression":
                prefix = "regression"
                model = "GradientBoosting Regressor"
                label = self.y_reg
                pred = self.data_reg['predict_reg']
                self.CF_Button.nextCheckState()
            else:
                prefix = "classification"
                model = "GradientBoosting Classifier"
                label = self.y_cls
                pred = self.data_cls['predict_cls']

            self.dataHead()
            if self.comboBox_plot.currentText() == "Heatmap":
                pixmap = QPixmap(os.path.join(resource_path("imgs"), f'{prefix}_{self.comboBox_plot.currentText()}.png'))
                self.label_plot.setPixmap(pixmap.scaled(698, 409, Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation))
                self.label_plot.setScaledContents(True)
                self.label_text.setText(f"\nInterpret {model} - {self.comboBox_plot.currentText()} Plot")
                self.label_text.setFont(QFont('Times', 13))
            else:
                if self.comboBox_plot.currentText() == "Force":
                    pixmap = QPixmap(os.path.join(resource_path("imgs"), f'{prefix}_{self.comboBox_plot.currentText()}_{self.idx}.png'))
                    self.label_plot.setPixmap(pixmap.scaled(1178, 272, Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation))
                elif self.comboBox_plot.currentText() == "Waterfall":
                    pixmap = QPixmap(os.path.join(resource_path("imgs"), f'{prefix}_{self.comboBox_plot.currentText()}_{self.idx}.png'))
                    self.label_plot.setPixmap(pixmap.scaled(661, 445, Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation))
                elif self.comboBox_plot.currentText() == "Bar":
                    pixmap = QPixmap(os.path.join(resource_path("imgs"), f'{prefix}_{self.comboBox_plot.currentText()}_{self.idx}.png'))
                    self.label_plot.setPixmap(pixmap.scaled(660, 426, Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation))
                self.label_plot.setScaledContents(True)
                self.label_text.setText(
                    f"\nSample.{self.idx}    Interpret {model}\nBase value : {label.mean():.2f}    Target : {label.values[self.idx]:.2f}    Prediction : {pred.values[self.idx]:.2f}\n")
                self.label_text.setFont(QFont('Times', 13))
            self.show()


app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()
