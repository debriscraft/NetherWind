# setup_cython.py
import os
import shutil
import glob
import sys
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

# ========== 配置区 ==========
# 要编译的核心模块（不含 red_ai.py 和 __init__.py）
COMPILE_MODULES = [
    "dogfight_env",
    "missile",
    "multi_logger",
    "simple_train",
    "test_dogfight",
    "test_trained_agent",
    "train_and_acmi",
]

# 编译优化级别
if os.name == 'nt':  # Windows
    EXTRA_COMPILE_ARGS = ["/O2"]
else:
    EXTRA_COMPILE_ARGS = ["-O3", "-Wall"]


# ========== 自动处理 ==========
def prepare_pyx():
    """将 .py 复制为 .pyx，用于 Cython 编译"""
    for name in COMPILE_MODULES:
        py_file = f"{name}.py"
        pyx_file = f"{name}.pyx"
        if os.path.exists(py_file):
            if os.path.exists(pyx_file):
                os.remove(pyx_file)
            shutil.copy2(py_file, pyx_file)
            print(f"[准备] {py_file} -> {pyx_file}")
        else:
            print(f"[跳过] 找不到 {py_file}")


def get_extensions():
    """生成 Extension 列表"""
    extensions = []
    for name in COMPILE_MODULES:
        pyx_file = f"{name}.pyx"
        if os.path.exists(pyx_file):
            ext = Extension(
                name=name,
                sources=[pyx_file],
                extra_compile_args=EXTRA_COMPILE_ARGS,
            )
            extensions.append(ext)
    return extensions


def clean_build():
    """清理编译残留"""
    patterns = ["*.c", "*.cpp", "*.so", "*.pyd", "build", "dist", "*.egg-info"]
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                    print(f"[清理] {path}")
                except:
                    pass


if __name__ == "__main__":
    # 如果带 --clean 参数，只清理不编译
    if "--clean" in sys.argv:
        sys.argv.remove("--clean")
        clean_build()
        print("[完成] 清理完毕")
        exit(0)

    # 准备 .pyx 文件
    prepare_pyx()

    # 执行编译
    setup(
        name="netherwind_core",
        version="0.1.0",
        ext_modules=cythonize(
            get_extensions(),
            compiler_directives={
                'language_level': "3",
                'embedsignature': False,
            },
            annotate=False,
        ),
        zip_safe=False,
    )

    print("\n" + "=" * 50)
    print("[编译完成] 请运行 release_clean.py 删除原始 .py 和 .pyx")
    print("=" * 50)