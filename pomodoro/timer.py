"""番茄钟核心计时器模块 - 纯逻辑层，不依赖任何 UI 库"""
import time
from enum import Enum


class TimerState(Enum):
    IDLE = "idle"
    WORKING = "working"
    PAUSED = "paused"
    SHORT_BREAK = "short_break"
    LONG_BREAK = "long_break"


class PomodoroTimer:
    """番茄钟计时器核心类
    使用回调机制与 UI 层解耦：
    - on_tick(remaining_seconds, total_seconds) - 每秒调用
    - on_state_change(new_state) - 状态切换时调用
    - on_complete(session_type) - 一个阶段完成时调用
    """

    def __init__(self):
        self._durations = {
            "work": 25 * 60,
            "short_break": 5 * 60,
            "long_break": 15 * 60,
            "long_break_interval": 4,
        }

        self._state = TimerState.IDLE
        self._remaining = self._durations["work"]  # 剩余秒数
        self._total_for_session = self._durations["work"]  # 当前阶段总秒数
        self._completed_count = 0  # 本次会话完成的番茄数
        self._pomodoro_count = 0   # 连续番茄数（用于判断长短休息）
        self._last_tick_time = 0
        self._pre_pause_state = None  # 暂停前的状态

        # 回调函数，由 UI 层设置
        self.on_tick = None          # (remaining_sec, total_sec)
        self.on_state_change = None  # (new_state: TimerState)
        self.on_complete = None      # (session_type: TimerState)

    @property
    def state(self):
        return self._state

    @property
    def remaining(self):
        return self._remaining

    @property
    def total_for_session(self):
        return self._total_for_session

    @property
    def completed_count(self):
        return self._completed_count

    def set_durations(self, work=None, short_break=None, long_break=None):
        """设置时长（分钟），更新后立即生效（重置后）"""
        if work is not None:
            self._durations["work"] = work * 60
        if short_break is not None:
            self._durations["short_break"] = short_break * 60
        if long_break is not None:
            self._durations["long_break"] = long_break * 60
        # 如果在空闲状态，同步更新剩余时间
        if self._state == TimerState.IDLE:
            self._remaining = self._durations["work"]
            self._total_for_session = self._durations["work"]

    def get_durations(self):
        """获取当前时长设置（分钟）"""
        return {
            "work": self._durations["work"] // 60,
            "short_break": self._durations["short_break"] // 60,
            "long_break": self._durations["long_break"] // 60,
        }

    def start(self):
        """开始工作计时"""
        if self._state == TimerState.PAUSED:
            return self.resume()

        # 从 IDLE 或休息状态开始工作
        self._remaining = self._durations["work"]
        self._total_for_session = self._durations["work"]
        self._state = TimerState.WORKING
        self._last_tick_time = time.time()
        self._notify_state_change()

    def pause(self):
        """暂停计时"""
        if self._state in (TimerState.WORKING, TimerState.SHORT_BREAK, TimerState.LONG_BREAK):
            self._pre_pause_state = self._state
            self._state = TimerState.PAUSED
            self._notify_state_change()

    def resume(self):
        """恢复计时"""
        if self._state == TimerState.PAUSED and self._pre_pause_state is not None:
            self._state = self._pre_pause_state
            self._last_tick_time = time.time()
            self._pre_pause_state = None
            self._notify_state_change()

    def reset(self):
        """重置计时器到空闲状态"""
        self._state = TimerState.IDLE
        self._remaining = self._durations["work"]
        self._total_for_session = self._durations["work"]
        self._pomodoro_count = 0
        self._notify_state_change()

    def skip(self):
        """跳过当前阶段"""
        # 确定实际状态（暂停时以暂停前的状态为准）
        effective_state = self._state
        if self._state == TimerState.PAUSED and self._pre_pause_state:
            effective_state = self._pre_pause_state
            self._pre_pause_state = None

        if effective_state == TimerState.WORKING:
            # 跳过工作时间 → 进入休息
            self._pomodoro_count += 1
            self._completed_count += 1
            if self.on_complete:
                self.on_complete(TimerState.WORKING)
            if self._pomodoro_count >= self._durations["long_break_interval"]:
                self._start_break(long_break=True)
                self._pomodoro_count = 0
            else:
                self._start_break(long_break=False)

        elif effective_state in (TimerState.SHORT_BREAK, TimerState.LONG_BREAK):
            # 跳过休息 → 开始工作
            if self.on_complete:
                self.on_complete(effective_state)
            self._start_work()

    def tick(self):
        """每秒钟调用一次，执行一秒钟的倒计时
        返回 True 表示仍在运行，False 表示需要停止调用
        推荐由 UI 层用 after() 定时调用
        """
        if self._state in (TimerState.IDLE, TimerState.PAUSED):
            return False

        now = time.time()
        elapsed = now - self._last_tick_time
        self._last_tick_time = now

        # 至少过去 0.9 秒才算一次有效 tick
        if elapsed < 0.9:
            return True

        self._remaining = max(0, self._remaining - 1)

        # 通知 UI 更新显示
        if self.on_tick:
            self.on_tick(self._remaining, self._total_for_session)

        # 计时结束
        if self._remaining <= 0:
            self._handle_completion()
            return False

        return True

    def _handle_completion(self):
        """处理计时完成"""
        if self._state == TimerState.WORKING:
            # 工作完成
            self._pomodoro_count += 1
            self._completed_count += 1

            if self.on_complete:
                self.on_complete(TimerState.WORKING)

            # 判断短休息还是长休息
            if self._pomodoro_count >= self._durations["long_break_interval"]:
                self._start_break(long_break=True)
                self._pomodoro_count = 0
            else:
                self._start_break(long_break=False)

        elif self._state == TimerState.SHORT_BREAK:
            if self.on_complete:
                self.on_complete(TimerState.SHORT_BREAK)
            self._start_work()

        elif self._state == TimerState.LONG_BREAK:
            if self.on_complete:
                self.on_complete(TimerState.LONG_BREAK)
            self._start_work()

    def _start_break(self, long_break=False):
        """进入休息状态"""
        if long_break:
            self._state = TimerState.LONG_BREAK
            self._remaining = self._durations["long_break"]
            self._total_for_session = self._durations["long_break"]
        else:
            self._state = TimerState.SHORT_BREAK
            self._remaining = self._durations["short_break"]
            self._total_for_session = self._durations["short_break"]

        self._last_tick_time = time.time()
        self._notify_state_change()

    def _start_work(self):
        """进入工作状态"""
        self._state = TimerState.WORKING
        self._remaining = self._durations["work"]
        self._total_for_session = self._durations["work"]
        self._last_tick_time = time.time()
        self._notify_state_change()

    def _notify_state_change(self):
        if self.on_state_change:
            self.on_state_change(self._state)
