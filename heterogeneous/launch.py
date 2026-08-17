"""Generic launcher for the compiled (.pyd) entry modules.

Usage: python launch.py <module> [args...]
Examples:
  python launch.py tests.smoke_bridge
  python launch.py run_phase6_train
  python launch.py run_ladder_eval --episodes 100
"""
import importlib
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    name = sys.argv[1]
    sys.argv = [name] + sys.argv[2:]
    mod = importlib.import_module(name)
    main = getattr(mod, 'main', None)
    if callable(main):
        main()
