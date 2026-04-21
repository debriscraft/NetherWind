# __init__.py
"""
NetherWind Air Combat Framework
核心模块已编译保护，仅 red_ai.py 开放源码供定制。
"""

__version__ = "0.1.0"

# 显式导入，确保编译模块可用
try:
    from . import dogfight_env
    from . import missile
    from . import multi_logger
    from . import train_and_acmi
except ImportError as e:
    print(f"[NetherWind] 核心模块加载异常: {e}")

# red_ai 保持源码开放，用户可直接修改
from . import red_ai

__all__ = [
    "dogfight_env",
    "missile",
    "multi_logger",
    "red_ai",
    "train_and_acmi",
]