# release_clean.py
import os
import glob
import shutil

# 编译后需要删除的原始文件（保护源码）
DELETE_PATTERNS = [
    "dogfight_env.py",
    "dogfight_env.pyx",
    "missile.py",
    "missile.pyx",
    "multi_logger.py",
    "multi_logger.pyx",
    "simple_train.py",
    "simple_train.pyx",
    "test_dogfight.py",
    "test_dogfight.pyx",
    "test_trained_agent.py",
    "test_trained_agent.pyx",
    "train_and_acmi.py",
    "train_and_acmi.pyx",
]

# 保留的文件（不删除）
KEEP_FILES = [
    "__init__.py",
    "red_ai.py",
    "setup_cython.py",  # 可选：发布时也可删除
    "release_clean.py",  # 可选：发布时也可删除
]


def clean_release():
    print("[发布清理] 开始删除原始源码，仅保留二进制和开放接口...")

    for pattern in DELETE_PATTERNS:
        for filepath in glob.glob(pattern):
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"  [删除] {filepath}")

    # 删除 build 目录（中间文件）
    if os.path.exists("build"):
        shutil.rmtree("build")
        print("  [删除] build/")

    # 删除 __pycache__
    for pycache in glob.glob("**/__pycache__", recursive=True):
        shutil.rmtree(pycache, ignore_errors=True)
        print(f"  [删除] {pycache}/")

    print("\n[保留文件检查]")
    for f in KEEP_FILES:
        status = "✓ 存在" if os.path.exists(f) else "✗ 缺失"
        print(f"  {status}: {f}")

    print("\n[编译产物检查]")
    for ext in ["*.so", "*.pyd"]:
        files = glob.glob(ext)
        for f in files:
            size = os.path.getsize(f) / 1024
            print(f"  ✓ {f} ({size:.1f} KB)")


if __name__ == "__main__":
    confirm = input("⚠️  此操作将永久删除原始 .py 源码！确认吗？ [yes/no]: ")
    if confirm.lower() in ["yes", "y"]:
        clean_release()
        print("\n✅ 发布包已就绪：red_ai.py 源码开放，其余模块已编译保护。")
    else:
        print("已取消。")