import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 配置中文字体支持
try:
    # 尝试使用系统中的中文字体
    font_path = font_manager.findfont(font_manager.FontProperties(family='SimHei'))
    if font_path:
        plt.rcParams['font.family'] = 'SimHei'
    else:
        # 备用方案，尝试其他常见中文字体
        for font in ['Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']:
            try:
                font_path = font_manager.findfont(font_manager.FontProperties(family=font))
                if font_path:
                    plt.rcParams['font.family'] = font
                    break
            except:
                continue
    
    # 设置正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告：无法加载中文字体，图表中的中文可能无法正常显示")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QComboBox, QTabWidget, 
                            QStyleFactory, QGroupBox, QGridLayout, QSplitter, QFrame)
from PyQt5.QtCore import Qt, QSize, QProcess
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap

# 自定义图表画布类
class MplCanvas(FigureCanvas):
    def __init__(self, width=10, height=6, dpi=100):
        # 创建图形
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.fig.patch.set_facecolor('#2D2D30')  # 设置图背景颜色
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#2D2D30')  # 设置坐标区背景颜色
        
        # 设置坐标轴颜色
        self.axes.spines['bottom'].set_color('white')
        self.axes.spines['top'].set_color('white')
        self.axes.spines['left'].set_color('white')
        self.axes.spines['right'].set_color('white')
        
        # 设置坐标轴标签颜色
        self.axes.tick_params(axis='x', colors='white')
        self.axes.tick_params(axis='y', colors='white')
        
        # 设置标题和标签颜色
        self.axes.yaxis.label.set_color('white')
        self.axes.xaxis.label.set_color('white')
        self.axes.title.set_color('white')
        
        # 设置网格
        self.axes.grid(True, color='gray', linestyle='--', alpha=0.3)
        
        super(MplCanvas, self).__init__(self.fig)

# 主应用窗口
class PredictionVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("5G小区流量预测分析系统")
        self.setMinimumSize(1200, 800)
        
        # 设置应用样式
        self.set_app_style()
        
        # 创建主部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建布局
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建顶部控制栏
        self.setup_control_bar()
        
        # 创建分析区域
        self.setup_analysis_area()
        
        # 初始化数据
        self.data = None
        
    def set_app_style(self):
        """设置应用样式，增加科技感"""
        # 设置深色主题
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(45, 45, 48))
        dark_palette.setColor(QPalette.WindowText, QColor(200, 200, 200))
        dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
        dark_palette.setColor(QPalette.ToolTipText, QColor(200, 200, 200))
        dark_palette.setColor(QPalette.Text, QColor(200, 200, 200))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 57))
        dark_palette.setColor(QPalette.ButtonText, QColor(200, 200, 200))
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        self.setPalette(dark_palette)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D30;
            }
            QWidget {
                color: #CCCCCC;
                background-color: #2D2D30;
            }
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1C86EE;
            }
            QPushButton:pressed {
                background-color: #0063B1;
            }
            QComboBox {
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 6px;
                background-color: #252526;
                color: #CCCCCC;
                min-width: 6em;
            }
            QComboBox:hover {
                border: 1px solid #0078D7;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #3F3F46;
                background-color: #252526;
                color: #CCCCCC;
            }
            QLabel {
                color: #CCCCCC;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #3F3F46;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #0078D7;
            }
            QTabWidget::pane {
                border: 1px solid #3F3F46;
                background-color: #2D2D30;
            }
            QTabBar::tab {
                background-color: #252526;
                color: #CCCCCC;
                border: 1px solid #3F3F46;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
            }
            QTabBar::tab:selected {
                background-color: #3F3F46;
                border-bottom: 2px solid #0078D7;
            }
            QTabBar::tab:hover {
                background-color: #333337;
            }
        """)
    
    def setup_control_bar(self):
        """设置顶部控制栏"""
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel)
        control_layout = QHBoxLayout(control_frame)
        
        # 创建文件选择区域
        file_group = QGroupBox("数据文件选择")
        file_layout = QVBoxLayout(file_group)
        
        # 创建文件选择按钮和显示标签
        file_row = QHBoxLayout()
        self.file_label = QLabel("未选择文件")
        self.file_label.setMinimumWidth(300)
        
        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self.select_file)
        
        file_row.addWidget(self.file_label)
        file_row.addWidget(self.browse_button)
        file_layout.addLayout(file_row)
        
        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Lasso回归", "多层感知机(MLP)", "支持向量机(SVM)", "决策树", "XGBoost"])
        self.model_combo.currentIndexChanged.connect(self.update_param_label)
        model_layout.addWidget(self.model_combo)
        
        # 参数显示
        self.param_label = QLabel("参数：-")
        model_layout.addWidget(self.param_label)
        
        # 分析按钮
        analyze_group = QGroupBox("操作")
        analyze_layout = QVBoxLayout(analyze_group)
        
        self.analyze_button = QPushButton("分析预测结果")
        self.analyze_button.setEnabled(False)  # 默认禁用，直到选择文件
        self.analyze_button.clicked.connect(self.analyze_prediction)
        analyze_layout.addWidget(self.analyze_button)
        
        # 添加运行模型按钮
        self.run_model_button = QPushButton("运行选择的模型")
        self.run_model_button.clicked.connect(self.run_selected_model)
        analyze_layout.addWidget(self.run_model_button)
        
        # 添加到控制栏
        control_layout.addWidget(file_group, 3)
        control_layout.addWidget(model_group, 1)
        control_layout.addWidget(analyze_group, 1)
        
        self.main_layout.addWidget(control_frame)
    
    def setup_analysis_area(self):
        """设置分析区域"""
        # 创建标签页
        self.tabs = QTabWidget()
        
        # 预测结果可视化标签页
        self.prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(self.prediction_tab)
        
        # 添加图表
        self.prediction_canvas = MplCanvas(width=10, height=6)
        prediction_layout.addWidget(self.prediction_canvas)
        
        # 创建结果指标区域
        metrics_frame = QFrame()
        metrics_layout = QGridLayout(metrics_frame)
        
        self.mse_label = QLabel("MSE: -")
        self.mae_label = QLabel("MAE: -")
        self.r2_label = QLabel("R²: -")
        
        metrics_layout.addWidget(QLabel("<b>模型性能指标:</b>"), 0, 0)
        metrics_layout.addWidget(self.mse_label, 0, 1)
        metrics_layout.addWidget(self.mae_label, 0, 2)
        metrics_layout.addWidget(self.r2_label, 0, 3)
        
        prediction_layout.addWidget(metrics_frame)
        
        # 区域分析标签页
        self.analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(self.analysis_tab)
        
        # 添加误差分析图表
        self.error_canvas = MplCanvas(width=10, height=6)
        analysis_layout.addWidget(self.error_canvas)
        
        # 添加标签页
        self.tabs.addTab(self.prediction_tab, "预测结果")
        self.tabs.addTab(self.analysis_tab, "误差分析")
        
        self.main_layout.addWidget(self.tabs)
        
        # 添加状态信息区域
        self.status_label = QLabel("就绪，请选择预测结果文件")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)
    
    def select_file(self):
        """选择CSV文件"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择预测结果文件", "../results/", 
            "CSV文件 (*.csv);;所有文件 (*)", options=options
        )
        
        if file_name:
            self.file_label.setText(os.path.basename(file_name))
            self.file_path = file_name
            self.analyze_button.setEnabled(True)
            self.status_label.setText(f"已选择文件: {os.path.basename(file_name)}")
            
            # 根据文件名自动选择对应的模型
            for i, model_name in enumerate(['Lasso回归', '多层感知机(MLP)', '支持向量机(SVM)', '决策树', 'XGBoost']):
                if model_name.split('(')[0].strip().lower() in os.path.basename(file_name).lower():
                    self.model_combo.setCurrentIndex(i)
                    break
    
    def analyze_prediction(self):
        """分析预测结果"""
        if not hasattr(self, 'file_path'):
            return
        
        try:
            # 读取CSV文件
            self.status_label.setText("正在加载数据...")
            self.data = pd.read_csv(self.file_path)
            
            if '真实值' not in self.data.columns or '预测值' not in self.data.columns:
                self.status_label.setText("错误: CSV文件格式不正确，需要包含'真实值'和'预测值'列")
                return
            
            # 计算指标
            y_true = self.data['真实值'].values
            y_pred = self.data['预测值'].values
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # 更新UI
            self.mse_label.setText(f"MSE: {mse:.4f}")
            self.mae_label.setText(f"MAE: {mae:.4f}")
            self.r2_label.setText(f"R²: {r2:.4f}")
            
            # 绘制预测结果图表
            self.plot_prediction(y_true, y_pred, mse, mae)
            
            # 绘制误差分析图表
            self.plot_error_analysis(y_true, y_pred)
            
            # 更新状态
            selected_model = self.model_combo.currentText()
            self.status_label.setText(f"分析完成，当前模型: {selected_model}")
            
        except Exception as e:
            self.status_label.setText(f"分析过程中出错: {str(e)}")
    
    def plot_prediction(self, y_true, y_pred, mse, mae):
        """绘制预测结果图表"""
        # 清除现有图表
        self.prediction_canvas.axes.clear()
        
        # 绘制实际值和预测值
        time_idx = np.arange(len(y_true))
        self.prediction_canvas.axes.plot(time_idx, y_true, 'b-', label='实际值', linewidth=2)
        self.prediction_canvas.axes.plot(time_idx, y_pred, 'r--', label='预测值', linewidth=2)
        
        # 设置标题和标签（确保使用正确的中文字体）
        selected_model = self.model_combo.currentText()
        title_text = f'小区5045流量预测结果对比 (MSE: {mse:.4f}, MAE: {mae:.4f})'
        self.prediction_canvas.axes.set_title(title_text, color='white', fontsize=14, fontfamily='SimHei')
        self.prediction_canvas.axes.set_xlabel('时间', color='white', fontfamily='SimHei')
        self.prediction_canvas.axes.set_ylabel('流量值', color='white', fontfamily='SimHei')
        
        # 设置图例，确保中文正确显示
        legend = self.prediction_canvas.axes.legend(prop={'family': 'SimHei'})
        for text in legend.get_texts():
            text.set_color('white')
            
        self.prediction_canvas.axes.grid(True, linestyle='--', alpha=0.7)
        
        # 刷新画布
        self.prediction_canvas.draw()
    
    def plot_error_analysis(self, y_true, y_pred):
        """绘制误差分析图表"""
        # 清除现有图表
        self.error_canvas.axes.clear()
        
        # 计算误差
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        # 创建多子图布局
        self.error_canvas.fig.clear()
        gs = self.error_canvas.fig.add_gridspec(2, 2)
        
        # 误差随时间变化图
        ax1 = self.error_canvas.fig.add_subplot(gs[0, :])
        time_idx = np.arange(len(errors))
        ax1.plot(time_idx, errors, 'g-', alpha=0.7)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title('预测误差随时间变化', color='white', fontfamily='SimHei')
        ax1.set_ylabel('误差', color='white', fontfamily='SimHei')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_facecolor('#2D2D30')
        
        # 误差分布直方图
        ax2 = self.error_canvas.fig.add_subplot(gs[1, 0])
        ax2.hist(errors, bins=20, color='skyblue', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_title('误差分布', color='white', fontfamily='SimHei')
        ax2.set_xlabel('误差', color='white', fontfamily='SimHei')
        ax2.set_ylabel('频次', color='white', fontfamily='SimHei')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_facecolor('#2D2D30')
        
        # 散点图: 实际值 vs 预测值
        ax3 = self.error_canvas.fig.add_subplot(gs[1, 1])
        ax3.scatter(y_true, y_pred, alpha=0.5, color='lightgreen')
        
        # 添加对角线 (理想预测)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax3.set_title('实际值 vs 预测值', color='white', fontfamily='SimHei')
        ax3.set_xlabel('实际值', color='white', fontfamily='SimHei')
        ax3.set_ylabel('预测值', color='white', fontfamily='SimHei')
        ax3.tick_params(axis='x', colors='white')
        ax3.tick_params(axis='y', colors='white')
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.set_facecolor('#2D2D30')
        
        # 调整布局并刷新
        self.error_canvas.fig.tight_layout()
        self.error_canvas.draw()
        
        # 分析哪些区域预测较好
        # 找出误差最小的前20%时间点
        good_idx = np.argsort(abs_errors)[:int(len(abs_errors)*0.2)]
        
        # 获取误差较大的时间点
        poor_idx = np.argsort(abs_errors)[-int(len(abs_errors)*0.2):]
        
        # 计算平均流量水平
        avg_flow = np.mean(y_true)
        
        # 分析不同流量水平的预测表现
        high_flow_idx = np.where(y_true > avg_flow)[0]
        low_flow_idx = np.where(y_true <= avg_flow)[0]
        
        high_flow_mae = mean_absolute_error(y_true[high_flow_idx], y_pred[high_flow_idx])
        low_flow_mae = mean_absolute_error(y_true[low_flow_idx], y_pred[low_flow_idx])
        
        # 确定预测更好的区域
        better_region = "低流量区域" if low_flow_mae < high_flow_mae else "高流量区域"

    def run_selected_model(self):
        """运行选中的模型进行训练和预测"""
        # 获取选择的模型
        selected_model = self.model_combo.currentText()
        
        # 映射UI显示的模型名称到experiment3.py中使用的模型名称
        model_mapping = {
            'Lasso回归': 'Lasso',
            '多层感知机(MLP)': 'MLP',
            '支持向量机(SVM)': 'SVR',
            '决策树': 'Decision Tree',
            'XGBoost': 'XGBoost',
            '线性回归': 'Linear Regression'
        }
        
        # 获取对应的模型名称
        model_name = model_mapping.get(selected_model, 'Lasso')
        
        # 设置状态
        self.status_label.setText(f"正在运行{selected_model}模型，请稍候...")
        
        try:
            # 获取experiment3.py的路径
            experiment3_path = os.path.abspath(os.path.join('..', '内容三', 'experiment3.py'))
            print("查找experiment3.py的路径为：", experiment3_path)
            if not os.path.exists(experiment3_path):
                self.status_label.setText(f"错误: 找不到experiment3.py文件，查找路径为：{experiment3_path}")
                return
            
            # 执行命令
            cmd = f"python \"{experiment3_path}\" \"{model_name}\""
            self.status_label.setText(f"执行命令: {cmd}")
            
            # 使用QProcess执行命令
            process = QProcess()
            process.setProcessChannelMode(QProcess.MergedChannels)
            
            # 连接信号
            process.readyReadStandardOutput.connect(
                lambda: self.status_label.setText(f"模型运行中...")
            )
            
            process.finished.connect(
                lambda: self.process_completed(model_name)
            )
            
            # 启动进程
            process.start(cmd)
            
        except Exception as e:
            self.status_label.setText(f"运行模型时出错: {str(e)}")
    
    def process_completed(self, model_name):
        """处理模型训练完成后的操作"""
        # 构建结果文件路径
        result_file = os.path.abspath(os.path.join('..', 'results', f"{model_name.replace(' ', '_')}_prediction_results.csv"))
        
        # 检查结果文件是否存在
        if os.path.exists(result_file):
            # 自动选择该文件
            self.file_path = result_file
            self.file_label.setText(os.path.basename(result_file))
            self.analyze_button.setEnabled(True)
            
            # 自动分析结果
            self.analyze_prediction()
            
            self.status_label.setText(f"{model_name}模型训练完成，已自动加载结果")
        else:
            self.status_label.setText(f"{model_name}模型训练可能已完成，但未找到结果文件: {result_file}")

    def update_param_label(self):
        """根据当前选择的模型，显示其参数"""
        model_mapping = {
            'Lasso回归': 'Lasso',
            '多层感知机(MLP)': 'MLP',
            '支持向量机(SVM)': 'SVR',
            '决策树': 'Decision Tree',
            'XGBoost': 'XGBoost',
            '线性回归': 'Linear Regression'
        }
        selected_model = self.model_combo.currentText()
        model_name = model_mapping.get(selected_model, 'Lasso')
        try:
            df = pd.read_csv(os.path.abspath(os.path.join('..', 'results', 'model_comparison.csv')))
            row = df[df['model'] == model_name]
            if not row.empty:
                params = row.iloc[0]['params']
                self.param_label.setText(f"参数：{params}")
            else:
                self.param_label.setText("参数：-")
        except Exception as e:
            self.param_label.setText("参数：读取失败")

def main():
    app = QApplication(sys.argv)
    window = PredictionVisualizer()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 