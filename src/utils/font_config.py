"""
Font Configuration Utilities for Chinese Display in Matplotlib

This module provides utilities to configure Chinese fonts for matplotlib plots.

Author: Quantitative Trading Team
Date: 2025-01-01
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

def setup_chinese_fonts():
    """
    Setup Chinese fonts for matplotlib plots
    
    Returns:
    --------
    bool : True if Chinese font was found and set, False otherwise
    """
    # 嘗試設定中文字體（按優先級排序）
    chinese_fonts = [
        'Microsoft JhengHei',      # 微軟正黑體 (Windows)
        'Microsoft YaHei',         # 微軟雅黑 (Windows)
        'SimHei',                  # 黑體 (Windows)
        'PingFang SC',             # 蘋方 (macOS)
        'Hiragino Sans GB',        # 冬青黑體 (macOS)
        'DejaVu Sans',             # Linux 備用
        'WenQuanYi Micro Hei',     # 文泉驛微米黑 (Linux)
        'Noto Sans CJK SC',        # Google Noto (Linux)
    ]
    
    font_found = False
    used_font = None
    
    for font_name in chinese_fonts:
        try:
            plt.rcParams['font.family'] = font_name
            
            # 測試字體是否可用
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '測試中文', fontsize=12)
                plt.close(fig)
            
            used_font = font_name
            font_found = True
            break
            
        except Exception:
            continue
    
    if not font_found:
        print("警告: 未找到中文字體，圖表中的中文可能顯示為方塊")
        print("請嘗試以下解決方案:")
        print("1. 安裝中文字體（如微軟雅黑、SimHei等）")
        print("2. 使用以下代碼查看系統可用字體:")
        print("   import matplotlib.font_manager as fm")
        print("   print([f.name for f in fm.fontManager.ttflist if any(word in f.name.lower() for word in ['chinese', 'zh', 'cjk', 'simhei', 'yahei'])])")
    else:
        print(f"✅ 成功設定中文字體: {used_font}")
    
    # 解決負號顯示問題
    plt.rcParams['axes.unicode_minus'] = False
    
    return font_found

def get_available_chinese_fonts():
    """
    Get list of available Chinese fonts on the system
    
    Returns:
    --------
    list : List of available Chinese font names
    """
    chinese_keywords = ['chinese', 'zh', 'cjk', 'simhei', 'yahei', 'jhenghei', 'pingfang', 'hiragino', 'noto']
    
    available_fonts = []
    for font in fm.fontManager.ttflist:
        font_name = font.name.lower()
        if any(keyword in font_name for keyword in chinese_keywords):
            available_fonts.append(font.name)
    
    return sorted(list(set(available_fonts)))

def test_chinese_display():
    """
    Test Chinese character display in matplotlib
    
    Returns:
    --------
    bool : True if Chinese characters display correctly
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 測試各種中文字符
        test_strings = [
            'BTC 收盤價格走勢',
            '預測結果',
            '時間',
            '價格 (USD)',
            '收益率',
            '統計指標'
        ]
        
        for i, text in enumerate(test_strings):
            ax.text(0.1, 0.9 - i * 0.1, text, fontsize=14, transform=ax.transAxes)
        
        ax.set_title('中文字體測試', fontsize=16, fontweight='bold')
        ax.set_xlabel('測試 X 軸標籤', fontsize=12)
        ax.set_ylabel('測試 Y 軸標籤', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ 中文字體顯示測試完成")
        return True
        
    except Exception as e:
        print(f"❌ 中文字體測試失敗: {e}")
        return False

def configure_plot_style():
    """Configure matplotlib style for better Chinese display"""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.unicode_minus': False,  # 解決負號顯示問題
    })

if __name__ == "__main__":
    print("=== 中文字體配置工具 ===")
    
    print("\n1. 可用的中文字體:")
    available_fonts = get_available_chinese_fonts()
    if available_fonts:
        for font in available_fonts[:10]:  # 只顯示前10個
            print(f"   - {font}")
        if len(available_fonts) > 10:
            print(f"   ... 還有 {len(available_fonts) - 10} 個字體")
    else:
        print("   沒有找到中文字體")
    
    print("\n2. 設定中文字體:")
    success = setup_chinese_fonts()
    
    print("\n3. 配置圖表樣式:")
    configure_plot_style()
    
    if success:
        print("\n4. 測試中文顯示:")
        test_chinese_display()
    
    print("\n✅ 字體配置完成！")