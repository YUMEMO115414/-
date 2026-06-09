"""配置管理与统计持久化模块"""
import json
import os
from datetime import date

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".pomodoro")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

DEFAULT_CONFIG = {
    "durations": {
        "work": 25,         # 工作时间 (分钟)
        "short_break": 5,   # 短休息 (分钟)
        "long_break": 15,   # 长休息 (分钟)
        "long_break_interval": 4,  # 每几个番茄后进入长休息
    },
    "statistics": {
        "total_pomodoros": 0,
        "daily": {}         # {"2026-06-09": 3, ...}
    }
}


def ensure_config_dir():
    """确保配置目录存在"""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)


def load_config():
    """加载配置，如果不存在则创建默认配置"""
    ensure_config_dir()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                # 确保所有默认键都存在
                merged = DEFAULT_CONFIG.copy()
                _deep_merge(merged, config)
                return merged
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_CONFIG.copy()


def save_config(config):
    """保存配置到文件"""
    ensure_config_dir()
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _deep_merge(base, override):
    """递归合并配置字典"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_durations():
    """获取时长配置"""
    config = load_config()
    return config["durations"]


def set_durations(work=None, short_break=None, long_break=None):
    """设置时长配置"""
    config = load_config()
    if work is not None:
        config["durations"]["work"] = work
    if short_break is not None:
        config["durations"]["short_break"] = short_break
    if long_break is not None:
        config["durations"]["long_break"] = long_break
    save_config(config)


def add_completed_pomodoro():
    """记录完成一个番茄"""
    today = date.today().isoformat()
    config = load_config()
    config["statistics"]["total_pomodoros"] += 1
    if today not in config["statistics"]["daily"]:
        config["statistics"]["daily"][today] = 0
    config["statistics"]["daily"][today] += 1
    save_config(config)


def get_today_count():
    """获取今日完成的番茄数"""
    today = date.today().isoformat()
    config = load_config()
    return config["statistics"]["daily"].get(today, 0)


def get_total_count():
    """获取总番茄数"""
    config = load_config()
    return config["statistics"]["total_pomodoros"]
