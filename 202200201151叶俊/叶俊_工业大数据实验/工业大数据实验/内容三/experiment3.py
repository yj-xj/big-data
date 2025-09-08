import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，不需要Tcl/Tk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# 条件导入xgboost
try:
    import xgboost as XGBRegressor
    xgboost_available = True
except ImportError:
    xgboost_available = False
    print("警告: 未安装xgboost包，XGBoost模型将不可用")

import logging
import os
import sys
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficPredictor:
    """小区流量预测模型类"""
    
    def __init__(self, data_path, model_type='Lasso', test_size=0.3, random_state=42):
        """
        初始化预测器
        
        参数:
            data_path: CSV数据文件路径
            model_type: 模型类型，可选['Lasso', 'MLP', 'SVR', 'Decision Tree', 'XGBoost', 'Linear Regression']
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.data_path = data_path
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.y_pred = None
        self.model_params = {}  # 存储模型参数
    
    def load_data(self):
        """加载CSV文件数据"""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"成功加载数据，共{len(self.data)}条记录")
            logger.info(f"数据列: {self.data.columns.tolist()}")
            return True
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            return False
    
    def prepare_data(self):
        """准备训练集和测试集"""
        if self.data is None:
            logger.error("请先加载数据")
            return False
            
        try:
            # 提取特征和目标变量
            X = self.data.drop('Y', axis=1)
            y = self.data['Y']
            
            # 划分训练集和测试集（按时间顺序，不打乱）
            train_size = int(len(X) * (1 - self.test_size))
            self.X_train, self.X_test = X[:train_size], X[train_size:]
            self.y_train, self.y_test = y[:train_size], y[train_size:]
            
            logger.info(f"数据集划分完成: 训练集 {len(self.X_train)} 样本, 测试集 {len(self.X_test)} 样本")
            return True
        
        except Exception as e:
            logger.error(f"数据准备失败: {str(e)}")
            return False
    
    def build_model(self):
        """构建预测模型"""
        try:
            if self.model_type == 'Lasso':
                alpha = 0.1
                self.model = Lasso(alpha=alpha, random_state=self.random_state)
                self.model_params = {'alpha': alpha}
                logger.info(f"已构建Lasso模型，alpha={alpha}")
            
            elif self.model_type == 'MLP':
                hidden_layer_sizes = (100, 50)
                max_iter = 1000
                self.model = MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=max_iter,
                    random_state=self.random_state
                )
                self.model_params = {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'max_iter': max_iter
                }
                logger.info(f"已构建MLP模型，隐藏层={hidden_layer_sizes}，最大迭代={max_iter}")
            
            elif self.model_type == 'SVR':
                kernel = 'rbf'
                C = 1.0
                epsilon = 0.1
                self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
                self.model_params = {'kernel': kernel, 'C': C, 'epsilon': epsilon}
                logger.info(f"已构建SVR模型，kernel={kernel}, C={C}, epsilon={epsilon}")
            
            elif self.model_type == 'Decision Tree':
                max_depth = 10
                self.model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    random_state=self.random_state
                )
                self.model_params = {'max_depth': max_depth}
                logger.info(f"已构建决策树模型，max_depth={max_depth}")
            
            elif self.model_type == 'XGBoost':
                # 检查XGBoost是否可用
                if not xgboost_available:
                    logger.error("XGBoost模型不可用，请先安装xgboost包")
                    self.model_type = 'Decision Tree'  # 回退到决策树模型
                    logger.info("已自动切换到决策树模型")
                    return self.build_model()  # 递归调用，使用决策树模型
                    
                n_estimators = 100
                learning_rate = 0.1
                max_depth = 5
                self.model = XGBRegressor.XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=self.random_state
                )
                self.model_params = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth
                }
                logger.info(f"已构建XGBoost模型，n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
            
            elif self.model_type == 'Linear Regression':
                self.model = LinearRegression()
                self.model_params = {}
                logger.info("已构建线性回归模型")
            
            else:
                logger.error(f"不支持的模型类型: {self.model_type}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"构建模型失败: {str(e)}")
            return False
    
    def train_and_evaluate(self):
        """训练和评估模型"""
        if self.X_train is None or self.y_train is None:
            logger.error("请先准备数据")
            return False
            
        try:
            # 训练模型
            logger.info(f"训练{self.model_type}模型")
            self.model.fit(self.X_train, self.y_train)
            
            # 预测
            self.y_pred = self.model.predict(self.X_test)
            
            # 计算评估指标
            mse = mean_squared_error(self.y_test, self.y_pred)
            mae = mean_absolute_error(self.y_test, self.y_pred)
            r2 = r2_score(self.y_test, self.y_pred)
            
            logger.info(f"{self.model_type} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # 保存模型评估指标到CSV
            self.save_metrics(mse, mae, r2)
            
            return True
                
        except Exception as e:
            logger.error(f"模型训练/评估失败: {str(e)}")
            return False
    
    def save_metrics(self, mse, mae, r2):
        """保存模型评估指标到CSV文件"""
        try:
            # 指标保存位置
            metrics_file = "../results/model_comparison.csv"
            os.makedirs("../results", exist_ok=True)
            
            # 创建评估指标记录
            metrics_row = {
                'model': self.model_type,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'params': str(self.model_params)
            }
            
            # 检查文件是否存在
            if os.path.exists(metrics_file):
                # 加载现有数据
                metrics_df = pd.read_csv(metrics_file)
                
                # 检查是否已有该模型记录
                if self.model_type in metrics_df['model'].values:
                    # 更新记录
                    metrics_df.loc[metrics_df['model'] == self.model_type] = pd.Series(metrics_row)
                else:
                    # 添加新记录
                    metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)
            else:
                # 创建新的DataFrame
                metrics_df = pd.DataFrame([metrics_row])
            
            # 保存指标
            metrics_df.to_csv(metrics_file, index=False)
            logger.info(f"模型评估指标已保存到 {metrics_file}")
            
        except Exception as e:
            logger.warning(f"保存模型评估指标失败: {str(e)}")
    
    def save_results(self):
        """保存预测结果到CSV文件"""
        if self.y_test is None or self.y_pred is None:
            logger.error("没有预测结果可保存")
            return False
            
        try:
            # 创建结果DataFrame
            results_df = pd.DataFrame({
                '真实值': self.y_test.values,
                '预测值': self.y_pred
            })
            
            # 创建结果目录
            os.makedirs("../results", exist_ok=True)
            
            # 保存结果表格
            output_file = f"../results/{self.model_type.replace(' ', '_')}_prediction_results.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(f"预测结果已保存到 {output_file}")
            
            # 绘制预测结果图表
            self.plot_prediction(output_file)
            
            return True
            
        except Exception as e:
            logger.error(f"结果保存失败: {str(e)}")
            return False
    
    def plot_prediction(self, csv_file):
        """绘制预测结果图表"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            plt.plot(self.y_test.values, 'b-', label='实际值', linewidth=1.5)
            plt.plot(self.y_pred, 'r--', label='预测值', linewidth=1.5)
            
            # 计算性能指标
            mse = mean_squared_error(self.y_test, self.y_pred)
            mae = mean_absolute_error(self.y_test, self.y_pred)
            
            # 设置图表标题和标签
            plt.title(f'小区5045流量预测结果对比 (MSE: {mse:.4f}, MAE: {mae:.4f})')
            plt.xlabel('时间')
            plt.ylabel('流量值')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 保存图表
            img_file = f"../results/{self.model_type}_prediction.png"
            plt.savefig(img_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"预测结果图表已保存到 {img_file}")
            
        except Exception as e:
            logger.warning(f"绘制预测结果图表失败: {str(e)}")
    
    def prediction_pipeline(self):
        """执行完整的预测流程"""
        # 1. 加载数据
        if not self.load_data():
            return False
        
        # 2. 准备数据
        if not self.prepare_data():
            return False
        
        # 3. 构建模型
        if not self.build_model():
            return False
        
        # 4. 训练和评估模型
        if not self.train_and_evaluate():
            return False
        
        # 5. 保存结果
        if not self.save_results():
            return False
        
        print(f"✅ 预测流程已成功完成，结果保存为 ../results/{self.model_type.replace(' ', '_')}_prediction_results.csv")
        return True

def main():
    """主函数"""
    # 获取命令行参数
    model_type = 'Lasso'  # 默认模型
    
    # 如果提供了命令行参数，使用指定的模型
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        
    # 支持的模型列表
    supported_models = ['Lasso', 'MLP', 'SVR', 'Decision Tree', 'XGBoost', 'Linear Regression']
    
    if model_type not in supported_models:
        logger.error(f"不支持的模型类型: {model_type}")
        logger.info(f"支持的模型: {', '.join(supported_models)}")
        return
    
    # 指定CSV文件路径
    csv_file_path = r"E:\叶俊_工业大数据实验\工业大数据实验\内容二\cell_5045_traffic_data.csv"
    
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        # 尝试从相对路径加载
        csv_file_path = "../内容二/cell_5045_traffic_data.csv"
        if not os.path.exists(csv_file_path):
            logger.error(f"找不到数据文件，请确保文件存在: {csv_file_path}")
            return
    
    # 创建预测器
    predictor = TrafficPredictor(
        data_path=csv_file_path,
        model_type=model_type,
        test_size=0.2,
        random_state=42
    )
    
    # 执行预测流程
    success = predictor.prediction_pipeline()
    
    if success:
        logger.info(f"{model_type}预测流程成功完成")
    else:
        logger.error(f"{model_type}预测流程失败")

if __name__ == "__main__":
    main() 