"""番茄钟桌面应用 - 入口模块"""
from .ui import PomodoroApp


def main():
    app = PomodoroApp()
    app.run()


if __name__ == "__main__":
    main()
