import h5py
import numpy as np
import pandas as pd
import os

import logging
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端避免显示问题
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理和存储类"""
    
    def __init__(self, data_path, cell_id=5045, window_size=4, prediction_size=1, standardize=True, validate=True):
        """
        初始化数据处理器
        
        参数:
            data_path: h5格式数据文件路径
            cell_id: 要处理的小区ID
            window_size: 滑动窗口大小
            prediction_size: 预测窗口大小
            standardize: 是否对数据进行标准化
            validate: 是否验证生成的数据
        """
        self.data_path = data_path
        self.cell_id = cell_id
        self.window_size = window_size
        self.prediction_size = prediction_size
        self.standardize = standardize
        self.validate = validate
        self.data = None
        self.raw_data = None  # 保存原始数据，便于验证
        self.processed_data = None
        
        # CSV文件路径
        self.csv_file_path = f'cell_{cell_id}_traffic_data.csv'
    
    def load_data(self):
        """从h5文件加载指定小区的流量数据"""
        try:
            with h5py.File(self.data_path, 'r') as f:
                # 获取所有时间点的第5种业务数据
                all_data = f['data'][:, :, 4]  # 索引4表示第5种业务
                # 提取指定小区的数据
                self.raw_data = all_data[:, self.cell_id].copy()
                self.data = self.raw_data.copy()
                
                # 可选的数据标准化
                if self.standardize:
                    mean_val = np.mean(self.data)
                    std_val = np.std(self.data)
                    self.data = (self.data - mean_val) / std_val
                    logger.info(f"数据已标准化: 均值={mean_val:.4f}, 标准差={std_val:.4f}")
                
            logger.info(f"成功加载小区{self.cell_id}的第5种业务数据，共{len(self.data)}个时间点")
            
            # 可视化部分数据用于验证
            if self.validate:
                self._visualize_sample_data()
                
            return True
        
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            return False
    
    def _visualize_sample_data(self):
        """可视化部分数据用于验证"""
        try:
            # 选择前168个时间点(一周)可视化
            sample_size = min(168, len(self.data))
            plt.figure(figsize=(10, 4))
            plt.plot(range(sample_size), self.data[:sample_size])
            plt.title(f"小区{self.cell_id}流量数据样本(前{sample_size}个时间点)")
            plt.xlabel("时间点")
            plt.ylabel("流量" + ("(标准化后)" if self.standardize else ""))
            plt.grid(True)
            plt.savefig(f"cell_{self.cell_id}_sample_data.png")
            plt.close()
            logger.info(f"数据样本可视化已保存为 cell_{self.cell_id}_sample_data.png")
        except Exception as e:
            logger.warning(f"数据可视化失败: {str(e)}")
    
    def create_sliding_windows(self):
        """创建滑动窗口数据"""
        if self.data is None:
            logger.error("请先加载数据")
            return False
        
        try:
            # 创建滑动窗口数据
            X = []
            y = []
            
            # 对每个可能的窗口起点
            for i in range(len(self.data) - self.window_size):
                # 提取窗口数据作为特征
                X.append(self.data[i:i+self.window_size])
                # 提取窗口后的数据作为目标
                y.append(self.data[i+self.window_size])
            
            # 创建DataFrame
            df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(self.window_size)])
            df['Y'] = y
            
            self.processed_data = df
            
            # 数据验证
            if self.validate:
                self._validate_sliding_windows()
            
            logger.info(f"成功创建滑动窗口数据，共{len(self.processed_data)}条记录")
            return True
            
        except Exception as e:
            logger.error(f"滑动窗口创建失败: {str(e)}")
            return False
    
    def _validate_sliding_windows(self):
        """验证滑动窗口数据正确性"""
        # 检查部分样本确保数据一致性
        try:
            sample_idx = min(10, len(self.processed_data) - 1)
            
            # 提取一个样本行
            sample_row = self.processed_data.iloc[sample_idx]
            
            # 检查X值是否与原始数据一致
            for i in range(self.window_size):
                expected_value = self.data[sample_idx + i]
                actual_value = sample_row[f'X{i+1}']
                if abs(expected_value - actual_value) > 1e-6:
                    logger.warning(f"数据验证警告: 样本 {sample_idx}, X{i+1} 值不符合预期")
            
            # 检查Y值是否与原始数据一致
            expected_y = self.data[sample_idx + self.window_size]
            actual_y = sample_row['Y']
            if abs(expected_y - actual_y) > 1e-6:
                logger.warning(f"数据验证警告: 样本 {sample_idx}, Y 值不符合预期")
                
            logger.info("滑动窗口数据验证通过")
        except Exception as e:
            logger.warning(f"数据验证过程出错: {str(e)}")
    
    def store_data(self):
        """将处理后的数据存储到CSV文件"""
        if self.processed_data is None or len(self.processed_data) == 0:
            logger.error("没有可存储的数据")
            return False
        
        try:
            # 存储到CSV文件
            self.processed_data.to_csv(self.csv_file_path, index=False)
            
            # 检查文件是否成功创建且大小合理
            if os.path.exists(self.csv_file_path):
                file_size = os.path.getsize(self.csv_file_path) / 1024  # KB
                logger.info(f"数据已成功存储到CSV文件 {self.csv_file_path} (大小: {file_size:.2f} KB)")
                
                # 简单验证：检查行数是否匹配
                saved_df = pd.read_csv(self.csv_file_path)
                if len(saved_df) != len(self.processed_data):
                    logger.warning(f"CSV文件行数 ({len(saved_df)}) 与原数据行数 ({len(self.processed_data)}) 不匹配!")
                else:
                    logger.info(f"CSV文件验证通过: 包含 {len(saved_df)} 行数据")
                    
                return True
            else:
                logger.error("CSV文件创建失败")
                return False
            
        except Exception as e:
            logger.error(f"数据存储失败: {str(e)}")
            return False
    
    def process_pipeline(self):
        """执行完整的数据处理流程"""
        # 1. 加载数据
        if not self.load_data():
            return False
        
        # 2. 创建滑动窗口
        if not self.create_sliding_windows():
            return False
        
        # 3. 存储数据
        if not self.store_data():
            return False
        
        print(f"✅ 数据已保存为 {self.csv_file_path}，样本数：{len(self.processed_data)}")
        return True

def find_h5_file():
    """查找H5数据文件的位置"""
    # 可能的文件路径列表
    possible_paths = [
        r"E:\叶俊_工业大数据实验\工业大数据实验\all_data_ct.h5",
        r"工业大数据实验\all_data_ct.h5",
        r"all_data_ct.h5",
        r"电信数据\all_data_ct.h5",
        r"..\all_data_ct.h5",
        r"..\..\all_data_ct.h5"
    ]
    
    # 请用户提供文件路径
    print("正在查找数据文件...")
    user_path = input("如果自动查找失败，请输入h5文件路径(直接回车则进行自动查找): ").strip()
    if user_path and os.path.exists(user_path):
        logger.info(f"使用用户提供的文件路径: {user_path}")
        return user_path
    
    # 添加当前文件夹及子文件夹中的所有h5文件
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.h5'):
                possible_paths.append(os.path.join(root, file))
    
    # 尝试这些路径
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"找到H5文件: {path}")
            return path
    
    # 如果都没找到，返回None
    logger.error("未找到H5文件，请手动指定正确的文件路径")
    return None

def main():
    """主函数"""
    # 查找H5文件
    data_file_path = find_h5_file()
    
    if data_file_path is None:
        logger.error("无法找到数据文件，程序退出")
        return
    
    # 创建数据处理器
    processor = DataProcessor(
        data_path=data_file_path,
        cell_id=5045,  # 处理5045小区的数据
        window_size=4,
        prediction_size=1,
        standardize=False,  # 修改为False，不进行标准化
        validate=True      # 进行数据验证
    )
    
    # 执行数据处理流程
    success = processor.process_pipeline()
    
    if success:
        logger.info("数据处理流程成功完成")
        print(f"数据处理完成，已生成CSV文件: {processor.csv_file_path}")
        print(f"- 总样本数: {len(processor.processed_data)}")
        print(f"- 特征数: {processor.window_size}")
        print(f"- 数据已标准化: {'是' if processor.standardize else '否'}")
        print("您可以继续进行内容三的模型训练")
    else:
        logger.error("数据处理流程失败")

if __name__ == "__main__":
    main()