"""
批量测试脚本 - 按顺序补充实验报告中的所有数据

运行顺序：
1. 测试 MC Learning Agent（如果有训练好的模型）
2. 对比表格式特征表示 vs 线性函数近似
3. 汇总所有结果

使用方法：
python run_all_tests.py
"""

import os
import sys
import json
import subprocess

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def run_script(script_name, description):
    """运行Python脚本"""
    print_section(description)
    print(f"Running: {script_name}")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n错误: 脚本执行失败")
        print(f"错误信息: {e}")
        return False
    except FileNotFoundError:
        print(f"\n错误: 找不到脚本 {script_name}")
        return False

def check_file_exists(filepath):
    """检查文件是否存在"""
    return os.path.exists(filepath)

def main():
    print("="*70)
    print(" " * 20 + "实验数据补充测试")
    print("="*70)
    
    results_summary = {}
    
    # 1. 测试 MC Learning
    print_section("任务 1: 测试 MC Learning Agent")
    
    mc_checkpoint = './checkpoints/mc_learning/mc_agent_smallGrid_500ep.pkl'
    
    if check_file_exists(mc_checkpoint):
        print(f"✓ 找到 MC Learning 检查点: {mc_checkpoint}")
        choice = input("\n是否测试 MC Learning? (y/n): ").strip().lower()
    else:
        print(f"✗ 未找到 MC Learning 检查点: {mc_checkpoint}")
        print("提示: 需要先训练 MC Learning Agent")
        choice = input("\n是否训练并测试 MC Learning? (y/n): ").strip().lower()
    
    if choice == 'y':
        if run_script('train_and_test_mc.py', 'MC Learning 测试'):
            print("\n✓ MC Learning 测试完成")
            # 读取结果
            result_file = 'test_results_MCLearningAgent_smallGrid_500train_100test.json'
            if check_file_exists(result_file):
                with open(result_file, 'r') as f:
                    results_summary['mc_learning'] = json.load(f)
        else:
            print("\n✗ MC Learning 测试失败")
    else:
        print("跳过 MC Learning 测试")
    
    # 2. 表格式特征表示 vs 线性函数近似
    print_section("任务 2: 表格式特征表示 vs 线性函数近似对比")
    
    choice = input("是否运行对比实验? (y/n): ").strip().lower()
    
    if choice == 'y':
        if run_script('test_tabular_feature.py', '特征表示方法对比'):
            print("\n✓ 对比实验完成")
            # 读取结果
            result_file = 'comparison_tabular_vs_approx.json'
            if check_file_exists(result_file):
                with open(result_file, 'r') as f:
                    results_summary['tabular_vs_approx'] = json.load(f)
        else:
            print("\n✗ 对比实验失败")
    else:
        print("跳过对比实验")
    
    # 3. 汇总结果
    print_section("测试完成 - 结果汇总")
    
    if 'mc_learning' in results_summary:
        mc_data = results_summary['mc_learning']
        print("【MC Learning 数据】")
        print(f"平均分数: {mc_data['average_score']:.1f}")
        print(f"胜率: {mc_data['win_rate']:.1f}%")
        print(f"最高分数: {mc_data['highest_score']:.1f}")
        print(f"\nLaTeX 代码:")
        print(f"MC Learning & {mc_data['average_score']:.1f} & {mc_data['win_rate']:.1f} & {mc_data['highest_score']:.1f} \\\\")
        print()
    
    if 'tabular_vs_approx' in results_summary:
        comparison = results_summary['tabular_vs_approx']
        tabular = comparison['tabular_feature']
        approx = comparison['linear_approximation']
        
        print("【表格式特征表示 vs 线性函数近似】")
        print(f"表格式特征表示:")
        print(f"  平均分数: {tabular['average_score']:.1f}")
        print(f"  胜率: {tabular['win_rate']:.1f}%")
        print(f"\n线性函数近似:")
        print(f"  平均分数: {approx['average_score']:.1f}")
        print(f"  胜率: {approx['win_rate']:.1f}%")
        print(f"\nLaTeX 代码:")
        print(f"表格式特征表示 & {tabular['average_score']:.1f} & {tabular['win_rate']:.1f} & 差 \\\\")
        print(f"线性函数近似 & {approx['average_score']:.1f} & {approx['win_rate']:.1f} & 好 \\\\")
        print()
    
    # 保存汇总结果
    if results_summary:
        summary_file = 'all_test_results_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"所有结果已保存到: {summary_file}")
    
    print("\n" + "="*70)
    print("测试流程结束")
    print("="*70)
    
    # 提示还需要补充的数据
    print("\n【待补充数据清单】")
    if 'mc_learning' not in results_summary:
        print("  ✗ MC Learning 在 smallGrid 上的性能数据")
    else:
        print("  ✓ MC Learning 在 smallGrid 上的性能数据")
    
    if 'tabular_vs_approx' not in results_summary:
        print("  ✗ 表格式特征表示 vs 线性函数近似对比数据")
    else:
        print("  ✓ 表格式特征表示 vs 线性函数近似对比数据")
    
    print("\n  ? 内存占用数据（可选，使用 analyze_memory.py）")
    print("  ? 仅距离特征的消融实验（可选）")
    print("  ? 特征权重演化分析（可选）")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
        sys.exit(0)
