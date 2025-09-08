import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，不需要Tcl/Tk
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionAnalyzer:
    """预测结果分析类"""
    
    def __init__(self, results_path):
        """
        初始化分析器
        
        参数:
            results_path: 预测结果CSV文件路径
        """
        self.results_path = results_path
        self.results_df = None
        
        # 创建结果文件夹
        os.makedirs("plots", exist_ok=True)
    
    def load_results(self):
        """加载预测结果数据"""
        try:
            self.results_df = pd.read_csv(self.results_path)
            logger.info(f"成功加载预测结果，共{len(self.results_df)}条记录")
            return True
        except Exception as e:
            logger.error(f"预测结果加载失败: {str(e)}")
            return False
    
    def calculate_metrics(self):
        """计算MSE和MAE指标"""
        if self.results_df is None:
            logger.error("请先加载预测结果")
            return None
            
        try:
            y_true = self.results_df['真实值']
            y_pred = self.results_df['预测值']
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            logger.info(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
            
            return {
                'mse': mse,
                'mae': mae
            }
        except Exception as e:
            logger.error(f"指标计算失败: {str(e)}")
            return None
    
    def visualize_comparison(self):
        """可视化预测值与真实值对比"""
        if self.results_df is None:
            logger.error("请先加载预测结果")
            return False
            
        try:
            # 创建时间索引
            time_idx = np.arange(len(self.results_df))
            
            # 获取真实值和预测值
            y_true = self.results_df['真实值']
            y_pred = self.results_df['预测值']
            
            # 计算指标
            metrics = self.calculate_metrics()
            if metrics is None:
                return False
            
            # 创建对比图
            plt.figure(figsize=(12, 6))
            plt.plot(time_idx, y_true, 'b-', label='实际值', linewidth=2)
            plt.plot(time_idx, y_pred, 'r--', label='预测值', linewidth=2)
            plt.title(f'小区5045流量预测结果对比 (MSE: {metrics["mse"]:.4f}, MAE: {metrics["mae"]:.4f})')
            plt.xlabel('时间')
            plt.ylabel('标准化流量')
            plt.legend()
            plt.grid(True)
            
            # 保存对比图
            plt.savefig("plots/prediction_comparison.png")
            plt.close()
            
            logger.info(f"预测对比图已保存至 plots/prediction_comparison.png")
            return True
            
        except Exception as e:
            logger.error(f"可视化对比失败: {str(e)}")
            return False
    
    def analyze_prediction_quality(self):
        """分析预测质量（找出预测好与不好的时刻）"""
        if self.results_df is None:
            logger.error("请先加载预测结果")
            return False
            
        try:
            # 添加绝对误差列
            self.results_df['绝对误差'] = np.abs(self.results_df['真实值'] - self.results_df['预测值'])
            
            # 计算四分位数
            q1 = self.results_df['绝对误差'].quantile(0.25)
            q3 = self.results_df['绝对误差'].quantile(0.75)
            
            # 定义好与差的预测
            good_predictions = self.results_df[self.results_df['绝对误差'] <= q1]
            bad_predictions = self.results_df[self.results_df['绝对误差'] >= q3]
            
            # 可视化分析
            plt.figure(figsize=(12, 8))
            
            # 绘制所有预测点
            plt.scatter(
                np.arange(len(self.results_df)), 
                self.results_df['绝对误差'], 
                c='gray', alpha=0.5, label='所有点'
            )
            
            # 绘制预测较好的点
            plt.scatter(
                good_predictions.index, 
                good_predictions['绝对误差'], 
                c='green', marker='o', label='预测良好(Q1)'
            )
            
            # 绘制预测较差的点
            plt.scatter(
                bad_predictions.index, 
                bad_predictions['绝对误差'], 
                c='red', marker='x', label='预测较差(Q3)'
            )
            
            plt.title('小区5045流量预测质量分析')
            plt.xlabel('时间')
            plt.ylabel('绝对误差')
            plt.legend()
            plt.grid(True)
            
            # 保存分析图
            plt.savefig("plots/prediction_quality_analysis.png")
            plt.close()
            
            # 创建细分区域分析图 - 按流量值范围分析预测质量
            plt.figure(figsize=(10, 6))
            
            # 按真实值将数据分成几个区间
            bins = 5
            self.results_df['流量区间'] = pd.qcut(self.results_df['真实值'], bins, labels=False)
            
            # 计算每个区间的平均误差
            error_by_value = self.results_df.groupby('流量区间')['绝对误差'].mean()
            
            # 获取每个区间的代表值
            interval_reps = self.results_df.groupby('流量区间')['真实值'].mean()
            
            # 绘制条形图
            plt.bar(range(bins), error_by_value, width=0.7)
            plt.xticks(
                range(bins), 
                [f"{interval_reps[i]:.2f}" for i in range(bins)], 
                rotation=0
            )
            plt.title('不同流量区间的预测误差分析')
            plt.xlabel('流量值区间（代表值）')
            plt.ylabel('平均绝对误差')
            plt.tight_layout()
            
            # 保存分析图
            plt.savefig("plots/error_by_traffic_level.png")
            plt.close()
            
            # 生成详细分析报告
            report = self.generate_analysis_report(good_predictions, bad_predictions)
            
            # 保存分析报告
            with open("plots/analysis_report.txt", "w", encoding="utf-8") as f:
                f.write(report)
            
            logger.info(f"预测质量分析已完成，报告保存至 plots/analysis_report.txt")
            return True
            
        except Exception as e:
            logger.error(f"预测质量分析失败: {str(e)}")
            return False
    
    def generate_analysis_report(self, good_predictions, bad_predictions):
        """生成分析报告"""
        metrics = self.calculate_metrics()
        
        # 计算好的预测所占比例
        good_ratio = len(good_predictions) / len(self.results_df) * 100
        bad_ratio = len(bad_predictions) / len(self.results_df) * 100
        
        # 分析流量的时序特征
        high_traffic = self.results_df[self.results_df['真实值'] > self.results_df['真实值'].mean()]
        low_traffic = self.results_df[self.results_df['真实值'] <= self.results_df['真实值'].mean()]
        
        high_traffic_mse = mean_squared_error(high_traffic['真实值'], high_traffic['预测值'])
        low_traffic_mse = mean_squared_error(low_traffic['真实值'], low_traffic['预测值'])
        
        report = f"""小区5045流量预测分析报告
===========================

1. 预测性能评估
----------------------------
总体MSE: {metrics['mse']:.4f}
总体MAE: {metrics['mae']:.4f}

2. 预测质量分布
----------------------------
预测良好的样本数: {len(good_predictions)} ({good_ratio:.2f}%)
预测较差的样本数: {len(bad_predictions)} ({bad_ratio:.2f}%)

3. 不同流量水平的预测性能
----------------------------
高于平均流量时的MSE: {high_traffic_mse:.4f}
低于平均流量时的MSE: {low_traffic_mse:.4f}

4. 结论
----------------------------
模型在{'低' if low_traffic_mse < high_traffic_mse else '高'}流量区域表现更好。
数据集中有{len(good_predictions)}个点(约{good_ratio:.2f}%)的预测误差很小，显示模型有较好的预测能力。
数据集中约有{bad_ratio:.2f}%的点预测误差较大，可能是由于这些时刻的流量波动较为异常。

5. 改进建议
----------------------------
1. 可考虑增加特征工程，如添加时间特征（小时、星期几等）
2. 尝试其他模型如XGBoost或LSTM，更好地捕捉时间序列特征
3. 针对预测较差的时刻，可进一步分析其特征，找出规律
"""
        return report
    
    def analysis_pipeline(self):
        """执行完整的分析流程"""
        # 1. 加载预测结果
        if not self.load_results():
            return False
        
        # 2. 可视化预测对比
        if not self.visualize_comparison():
            return False
        
        # 3. 分析预测质量
        if not self.analyze_prediction_quality():
            return False
        
        print("✅ 预测结果分析已完成，结果保存在plots文件夹中")
        return True

def main():
    """主函数"""
    # 指定预测结果CSV文件路径
    results_path = r"E:\叶俊_工业大数据实验\工业大数据实验\内容三\lasso_prediction_results.csv"
    
    # 检查文件是否存在
    if not os.path.exists(results_path):
        # 尝试从相对路径加载
        results_path = "../内容三/lasso_prediction_results.csv"
        if not os.path.exists(results_path):
            logger.error(f"找不到预测结果文件，请确保文件存在: {results_path}")
            return
    
    # 创建分析器
    analyzer = PredictionAnalyzer(results_path=results_path)
    
    # 执行分析流程
    success = analyzer.analysis_pipeline()
    
    if success:
        logger.info("分析流程成功完成")
    else:
        logger.error("分析流程失败")
        
    # 直接打印简要分析结果
    if success:
        metrics = analyzer.calculate_metrics()
        print("\n预测结果简要分析:")
        print(f"- MSE: {metrics['mse']:.4f}")
        print(f"- MAE: {metrics['mae']:.4f}")
        print("- 详细分析报告已保存至plots/analysis_report.txt")
        print("- 可视化图表已保存至plots文件夹")

if __name__ == "__main__":
    main() 