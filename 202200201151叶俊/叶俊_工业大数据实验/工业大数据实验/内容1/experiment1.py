import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.offline import plot
import os

# 设置英文字体支持
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用Arial字体
plt.rcParams['axes.unicode_minus'] = True  # 正常显示负号

def plot_cell_traffic_time_series(data_path, cell_id_list, start_hour=168):
    """
    Plot weekly traffic changes for specified cells
    
    Parameters:
        data_path: h5 format data file path
        cell_id_list: List of cell IDs to plot
        start_hour: Starting hour for the week to plot (default: 168, which is the second week)
    """
    # Read data
    with h5py.File(data_path, 'r') as f:
        # 获取指定的一周数据（从start_hour开始的168小时）
        data = f['data'][start_hour:start_hour+168, :, 0]
    
    # Create chart
    plt.figure(figsize=(14, 8))
    
    # Set seaborn style
    sns.set_style("darkgrid")
    
    # Create x-axis ticks (hours)
    hours = np.arange(168)
    
    # Set color list - red and orange series
    color_list = ['#FF0000', '#FF4500', '#FF8C00']
    
    # Plot traffic changes for each cell
    line_objects = []
    for i, cell_id in enumerate(cell_id_list):
        line, = plt.plot(hours, data[:, cell_id], linewidth=2, label=f'Cell {cell_id}', 
                 color=color_list[i % len(color_list)], alpha=0.85)
        line_objects.append(line)
    
    # Add title and labels
    week_number = start_hour // 168 + 1
    plt.title(f'Week {week_number} Cell Traffic Changes', fontsize=16, fontweight='bold')
    plt.xlabel('Hours', fontsize=14)
    plt.ylabel('Normalized Traffic Value', fontsize=14)
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks, one tick every 24 hours
    plt.xticks(np.arange(0, 168, 24), labels=[f'Day {i+1}' for i in range(7)])
    
    # Add legend
    plt.legend(fontsize=12, framealpha=0.7)
    
    # Add day/night alternating background
    for i in range(7):
        # Night time (assuming 18-6 is night)
        plt.axvspan(i*24+18, (i+1)*24+6, alpha=0.1, color='gray')
    
    # Add numeric annotations - annotate peak points for each cell
    for i, cell_id in enumerate(cell_id_list):
        # Find peak point
        peak_index = np.argmax(data[:, cell_id])
        peak_value = data[peak_index, cell_id]
        # Add annotation
        plt.annotate(f'{peak_value:.2f}', 
                    xy=(peak_index, peak_value),
                    xytext=(peak_index+5, peak_value+0.05),
                    arrowprops=dict(facecolor=color_list[i], shrink=0.05, width=1.5),
                    fontsize=10)
        
        # Annotate daily average for each day
        for day in range(7):
            start = day * 24
            end = (day + 1) * 24
            daily_avg = np.mean(data[start:end, cell_id])
            # Annotate at middle position of each day
            plt.text(start + 12, daily_avg, f'Avg: {daily_avg:.2f}', 
                    color=color_list[i], fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.6))
    
    # Save image
    plt.tight_layout()
    plt.savefig(f'Week_{week_number}_Cell_Traffic.png', dpi=300)
    plt.close()
    
    print(f"Cell traffic time series has been saved as 'Week_{week_number}_Cell_Traffic.png'")


def generate_heatmap(data_path, time=12):
    """
    Generate heatmap for a 10*10 area at specified time
    
    Parameters:
        data_path: h5 format data file path
        time: Time (hour) to display
    """
    # Read data
    with h5py.File(data_path, 'r') as f:
        single_frame_data = f['data'][time, :, 0]
    
    # Reshape 1D data to 100*100 grid
    grid_data = single_frame_data.reshape(100, 100)
    
    # Extract target area (45:55, 45:55)
    target_area = grid_data[45:55, 45:55]
    
    # Method 1: Generate static heatmap using matplotlib
    plt.figure(figsize=(10, 8))
    
    # Use custom color map - red and orange series
    color_map = LinearSegmentedColormap.from_list('custom', ['#FFFFE0', '#FFEDA0', '#FED976', 
                                                  '#FEB24C', '#FD8D3C', '#FC4E2A', 
                                                  '#E31A1C', '#BD0026', '#800026'])
    
    # Generate heatmap
    heatmap = plt.imshow(target_area, cmap=color_map, interpolation='nearest')
    
    # Add color bar
    color_bar = plt.colorbar(heatmap)
    color_bar.set_label('Traffic Intensity', fontsize=12)
    
    # Set axis labels
    plt.xticks(np.arange(10), np.arange(45, 55))
    plt.yticks(np.arange(10), np.arange(45, 55))
    
    # Add axis titles
    plt.xlabel('X Coordinate (45-54)', fontsize=12)
    plt.ylabel('Y Coordinate (45-54)', fontsize=12)
    
    # Add grid lines
    plt.grid(False)
    
    # 计算时间信息
    week_number = time // 168 + 1
    day_in_week = (time % 168) // 24 + 1
    hour_in_day = time % 24
    
    # Add title with detailed time information
    plt.title(f'第{week_number}周 第{day_in_week}天 {hour_in_day}时 10*10小区流量热力图', fontsize=14, fontweight='bold')
    
    # Add numeric annotations - annotate hotspot areas with specific values
    # Find max and min value positions
    max_value = np.max(target_area)
    min_value = np.min(target_area)
    max_position = np.unravel_index(np.argmax(target_area), target_area.shape)
    min_position = np.unravel_index(np.argmin(target_area), target_area.shape)
    
    # Annotate max and min values
    plt.text(max_position[1], max_position[0], f'Max: {max_value:.2f}', 
             color='white', fontweight='bold', ha='center', va='center',
             bbox=dict(facecolor='black', alpha=0.7))
    
    plt.text(min_position[1], min_position[0], f'Min: {min_value:.2f}', 
             color='black', fontweight='bold', ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Annotate other significant points
    # Calculate mean and standard deviation
    mean_value = np.mean(target_area)
    std_dev = np.std(target_area)
    
    # Annotate points higher than mean + std_dev
    high_points = target_area > (mean_value + std_dev)
    for i in range(10):
        for j in range(10):
            if high_points[i, j] and (i, j) != max_position:
                plt.text(j, i, f'{target_area[i, j]:.2f}', 
                         color='white', fontsize=8, ha='center', va='center')
    
    # Save image
    plt.tight_layout()
    plt.savefig('Cell_Heatmap.png', dpi=300)
    plt.close()
    
    # Method 2: Generate interactive heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=target_area,
        x=list(range(45, 55)),
        y=list(range(45, 55)),
        colorscale='Reds',  # Red series
        hoverongaps=False,
        text=[[f'{target_area[i, j]:.2f}' for j in range(10)] for i in range(10)],  # Add value labels
        hovertemplate='X: %{x}<br>Y: %{y}<br>Traffic Value: %{text}<extra></extra>'
    ))
    
    # 计算时间信息
    week_number = time // 168 + 1
    day_in_week = (time % 168) // 24 + 1
    hour_in_day = time % 24
    
    # 创建详细的时间标题
    time_title = f'第{week_number}周 第{day_in_week}天 {hour_in_day}时 小区流量分布 (交互式)'
    
    fig.update_layout(
        title=time_title,
        xaxis_title='X坐标 (45-54)',
        yaxis_title='Y坐标 (45-54)',
        width=900,  # Increased width
        height=800,  # Increased height
        margin=dict(l=50, r=50, t=80, b=50)  # Adjusted margins
    )
    
    # 添加时间水印注释
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        text=f"时间点: 第{week_number}周 第{day_in_week}天 {hour_in_day}:00",
        showarrow=False,
        font=dict(size=14, color="gray"),
        align="center",
        opacity=0.8
    )
    
    # Add annotation for max value
    fig.add_annotation(
        x=45 + max_position[1],
        y=45 + max_position[0],
        text=f"最大值: {max_value:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#E31A1C",
        font=dict(size=12, color="white"),
        bgcolor="#BD0026",
        bordercolor="#800026",
        borderwidth=2,
        borderpad=4,
        opacity=0.8
    )
    
    # 保存为HTML文件 - 移除html_template参数
    plot(fig, filename='Interactive_Cell_Heatmap.html', auto_open=False)
    
    # 创建自定义CSS文件
    css_content = """
    body {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        margin: 0;
        padding: 0;
        background-color: #f5f5f5;
    }
    .plotly-graph-div {
        transform: scale(1.2);
        transform-origin: center center;
        margin: 0 auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-radius: 8px;
    }
    """
    
    # 保存CSS文件
    with open('heatmap_style.css', 'w') as f:
        f.write(css_content)
    
    # 修改HTML文件的标题，包含时间信息
    html_file = f'Interactive_Cell_Heatmap_Week{week_number}_Day{day_in_week}_Hour{hour_in_day}.html'
    
    # 读取CSS样式
    css_file = 'heatmap_style.css'
    with open(css_file, 'r') as f:
        css_style = f.read()
    
    # 添加自定义HTML头部，包含时间信息的标题和样式
    custom_html_head = f"""
    <head>
        <title>小区流量热力图 - 第{week_number}周第{day_in_week}天{hour_in_day}时</title>
        <style>
            {css_style}
            .time-info {{
                position: fixed;
                top: 10px;
                right: 10px;
                background-color: rgba(255,255,255,0.8);
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                z-index: 1000;
                font-size: 16px;
                font-weight: bold;
                color: #E31A1C;
            }}
        </style>
    </head>
    """
    
    # 添加时间信息的HTML元素
    time_info_html = f"""
    <div class="time-info">
        时间点: 第{week_number}周 第{day_in_week}天 {hour_in_day}:00
    </div>
    """
    
    # 保存为HTML文件，包含自定义头部和时间信息
    plot_html = plot(fig, include_plotlyjs=True, output_type='div')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(f"<!DOCTYPE html>\n<html>\n{custom_html_head}\n<body>\n{time_info_html}\n{plot_html}\n</body>\n</html>")
    
    print(f"交互式热力图已保存为 '{html_file}'")
    
    # 同时保存一个标准版本，保持兼容性
    plot(fig, filename='Interactive_Cell_Heatmap.html', auto_open=False)
    
    return fig

if __name__ == "__main__":
    # Use raw string notation to avoid path problems
    data_file_path = r"E:\叶俊_工业大数据实验\工业大数据实验\all_data_ct.h5"
    
    # 绘制第二周的小区流量时序图 (从第168小时开始)
    plot_cell_traffic_time_series(data_file_path, [1045, 5045, 8000], start_hour=168)
    
    # 也可以绘制第三周的数据
    # plot_cell_traffic_time_series(data_file_path, [1045, 5045, 8000], start_hour=336)
    
    # 生成中午12点的热度图
    generate_heatmap(data_file_path, time=12)