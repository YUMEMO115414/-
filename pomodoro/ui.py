"""番茄钟主界面模块 - 使用 tkinter 构建极简现代风格界面"""
import tkinter as tk
from tkinter import ttk, messagebox
import math
import winsound
import threading

from .timer import PomodoroTimer, TimerState
from .config import (
    load_config, save_config, get_durations, set_durations,
    add_completed_pomodoro, get_today_count, get_total_count
)


# ============================================================
# 颜色主题 (极简现代风)
# ============================================================
BG_COLOR = "#F5F5F5"           # 主背景
CARD_BG = "#FFFFFF"            # 卡片白
TEXT_PRIMARY = "#2D2D2D"       # 主文字
TEXT_SECONDARY = "#999999"     # 次要文字
WORK_COLOR = "#E74C3C"         # 工作红
WORK_LIGHT = "#FFECEC"          # 工作浅红
BREAK_COLOR = "#27AE60"         # 休息绿
BREAK_LIGHT = "#E8F8EF"         # 休息浅绿
RING_BG = "#E8E8E8"            # 进度环背景
BUTTON_BG = "#FFFFFF"          # 按钮背景
BUTTON_BORDER = "#DDDDDD"      # 按钮边框
HOVER_BLUE = "#3498DB"         # 悬停蓝


# ============================================================
# 自定义圆角按钮 (Canvas 绘制)
# ============================================================
class RoundedButton(tk.Canvas):
    """极简圆角按钮"""

    def __init__(self, parent, text, command=None, width=120, height=42,
                 bg=BUTTON_BG, fg=TEXT_PRIMARY, font=None, accent_color=None, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=BG_COLOR, highlightthickness=0, **kwargs)

        self._text = text
        self._command = command
        self._btn_bg = bg
        self._btn_fg = fg
        self._font = font or ("Microsoft YaHei UI", 12)
        self._accent = accent_color
        self._radius = 21  # 高度的一半
        self._hovered = False
        self._pressed = False

        self._draw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self):
        self.delete("all")
        w = self.winfo_width() or int(self["width"])
        h = self.winfo_height() or int(self["height"])
        r = h // 2

        # 颜色逻辑
        if self._pressed:
            bg = self._accent or HOVER_BLUE
            fg = "#FFFFFF"
        elif self._hovered:
            bg = self._accent or self._btn_bg
            # 微调亮度
            fg = "#FFFFFF" if self._accent else self._btn_fg
        else:
            bg = self._btn_bg
            fg = self._btn_fg

        # 阴影 (仅在未按下时)
        if not self._pressed and not self._hovered:
            self.create_rounded_rect(2, 2, w, h, r, fill="#E0E0E0", outline="")
        elif not self._pressed:
            self.create_rounded_rect(2, 2, w, h, r, fill="#D0D0D0", outline="")

        # 主体
        self.create_rounded_rect(0, -1 if self._pressed else -2, w, h - (3 if self._pressed else 2),
                                 r, fill=bg, outline=self._accent or BUTTON_BORDER if not self._hovered else "")

        # 文字
        self.create_text(w // 2, h // 2 - 1 + (1 if self._pressed else 0),
                         text=self._text, fill=fg, font=self._font, anchor="center")

    def create_rounded_rect(self, x, y, w, h, r, **kwargs):
        """绘制圆角矩形"""
        points = [
            x + r, y,
            x + w - r, y,
            x + w - r, y,
            x + w, y,
            x + w, y + r,
            x + w, y + r,
            x + w, y + h - r,
            x + w, y + h - r,
            x + w, y + h,
            x + w - r, y + h,
            x + w - r, y + h,
            x + r, y + h,
            x + r, y + h,
            x, y + h,
            x, y + h - r,
            x, y + h - r,
            x, y + r,
            x, y + r,
            x, y,
            x + r, y,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _on_enter(self, event):
        self._hovered = True
        self._draw()

    def _on_leave(self, event):
        self._hovered = False
        self._pressed = False
        self._draw()

    def _on_press(self, event):
        self._pressed = True
        self._draw()

    def _on_release(self, event):
        self._pressed = False
        self._draw()
        if self._command:
            self._command()

    def set_text(self, text):
        self._text = text
        self._draw()


# ============================================================
# 设置对话框
# ============================================================
class SettingsDialog(tk.Toplevel):
    """时长设置弹窗"""

    def __init__(self, parent, timer, on_save=None):
        super().__init__(parent)
        self.title("设置")
        self.resizable(False, False)
        self.configure(bg=BG_COLOR)

        self._timer = timer
        self._on_save = on_save

        # 居中于父窗口
        self.geometry("320x340")
        self.transient(parent)
        self.grab_set()

        # 图标/装饰隐藏
        self._build_ui()

        # 居中
        self.update_idletasks()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        px, py = parent.winfo_x(), parent.winfo_y()
        x = px + (pw - 320) // 2
        y = py + (ph - 340) // 2
        self.geometry(f"+{x}+{y}")

    def _build_ui(self):
        # 标题
        title = tk.Label(self, text="⏱ 番茄钟设置", font=("Microsoft YaHei UI", 16, "bold"),
                         fg=TEXT_PRIMARY, bg=BG_COLOR)
        title.pack(pady=(24, 20))

        durations = self._timer.get_durations()

        # 工作时长
        self._work_var = tk.IntVar(value=durations["work"])
        self._add_slider("🍅 工作时长", self._work_var, 5, 60)

        # 短休息时长
        self._short_var = tk.IntVar(value=durations["short_break"])
        self._add_slider("☕ 短休息", self._short_var, 1, 30)

        # 长休息时长
        self._long_var = tk.IntVar(value=durations["long_break"])
        self._add_slider("🌴 长休息", self._long_var, 5, 60)

        # 保存按钮
        btn_frame = tk.Frame(self, bg=BG_COLOR)
        btn_frame.pack(pady=(16, 20))

        save_btn = RoundedButton(btn_frame, "保存设置", command=self._save,
                                 width=140, height=40, accent_color=HOVER_BLUE)
        save_btn.pack()

    def _add_slider(self, label_text, var, min_val, max_val):
        """添加带滑动条和数值显示的设置行"""
        frame = tk.Frame(self, bg=BG_COLOR)
        frame.pack(fill="x", padx=32, pady=6)

        lbl = tk.Label(frame, text=label_text, font=("Microsoft YaHei UI", 11),
                       fg=TEXT_PRIMARY, bg=BG_COLOR)
        lbl.pack(side="left")

        val_lbl = tk.Label(frame, textvariable=var, font=("Microsoft YaHei UI", 13, "bold"),
                           fg=HOVER_BLUE, bg=BG_COLOR, width=2)
        val_lbl.pack(side="right")

        unit = tk.Label(frame, text="分钟", font=("Microsoft YaHei UI", 10),
                        fg=TEXT_SECONDARY, bg=BG_COLOR)
        unit.pack(side="right", padx=(0, 4))

        # 滑动条
        slider = tk.Scale(frame, from_=min_val, to=max_val, variable=var,
                          orient="horizontal", bg=BG_COLOR, fg=TEXT_PRIMARY,
                          highlightthickness=0, borderwidth=0, length=160,
                          sliderlength=16, troughcolor=RING_BG,
                          activebackground=HOVER_BLUE,
                          font=("Microsoft YaHei UI", 8))
        slider.pack(side="right", padx=(0, 8))

    def _save(self):
        w = self._work_var.get()
        s = self._short_var.get()
        l = self._long_var.get()
        self._timer.set_durations(work=w, short_break=s, long_break=l)
        set_durations(work=w, short_break=s, long_break=l)
        if self._on_save:
            self._on_save()
        self.destroy()


# ============================================================
# 主应用
# ============================================================
class PomodoroApp:
    """番茄钟主应用"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("番茄钟")
        self.root.geometry("400x560")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_COLOR)

        # 设置窗口图标 (尝试)
        try:
            self.root.iconbitmap(default="")
        except Exception:
            pass

        # 初始化计时器
        self.timer = PomodoroTimer()
        self.timer.on_tick = self._on_tick
        self.timer.on_state_change = self._on_state_change
        self.timer.on_complete = self._on_complete

        # 同步配置中的时长到计时器
        cfg_durations = get_durations()
        self.timer.set_durations(
            work=cfg_durations["work"],
            short_break=cfg_durations["short_break"],
            long_break=cfg_durations["long_break"],
        )

        # tick 调度
        self._tick_job = None

        # 构建界面
        self._build_ui()

        # 初始化显示
        self._update_display(force=True)

        # 居中窗口
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - 400) // 2
        y = (sh - 560) // 2
        self.root.geometry(f"+{x}+{y}")

        # 关闭窗口时停止 tick
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI 构建 ----------

    def _build_ui(self):
        """构建全部 UI 组件"""

        # -- 顶部栏 --
        top_bar = tk.Frame(self.root, bg=BG_COLOR, height=44)
        top_bar.pack(fill="x", padx=20, pady=(12, 0))
        top_bar.pack_propagate(False)

        # 标题
        title = tk.Label(top_bar, text="🍅 番茄钟", font=("Microsoft YaHei UI", 15, "bold"),
                         fg=TEXT_PRIMARY, bg=BG_COLOR)
        title.pack(side="left")

        # 设置按钮
        settings_btn = tk.Label(top_bar, text="⚙", font=("Segoe UI Symbol", 18),
                                fg=TEXT_SECONDARY, bg=BG_COLOR, cursor="hand2")
        settings_btn.pack(side="right")
        settings_btn.bind("<Button-1>", lambda e: self._open_settings())

        # -- 进度环区域 --
        ring_frame = tk.Frame(self.root, bg=BG_COLOR)
        ring_frame.pack(pady=(16, 4))

        self.ring_canvas = tk.Canvas(ring_frame, width=260, height=260,
                                     bg=BG_COLOR, highlightthickness=0)
        self.ring_canvas.pack()

        # 倒计时文字 (覆盖在 Canvas 上方)
        self.time_label = tk.Label(ring_frame, text="25:00",
                                   font=("Consolas", 48, "bold"),
                                   fg=WORK_COLOR, bg=BG_COLOR)
        self.time_label.place(in_=self.ring_canvas, relx=0.5, rely=0.5, anchor="center")

        # -- 状态标签 --
        self.state_label = tk.Label(self.root, text="准备开始",
                                    font=("Microsoft YaHei UI", 14),
                                    fg=TEXT_SECONDARY, bg=BG_COLOR)
        self.state_label.pack(pady=(2, 12))

        # -- 按钮栏 --
        btn_frame = tk.Frame(self.root, bg=BG_COLOR)
        btn_frame.pack(pady=(0, 16))

        self.start_btn = RoundedButton(btn_frame, "▶  开始", command=self._on_start,
                                       width=140, height=48, accent_color=WORK_COLOR,
                                       font=("Microsoft YaHei UI", 14, "bold"))
        self.start_btn.pack(side="left", padx=6)

        self.reset_btn = RoundedButton(btn_frame, "↺  重置", command=self._on_reset,
                                       width=95, height=48,
                                       font=("Microsoft YaHei UI", 12))
        self.reset_btn.pack(side="left", padx=6)

        self.skip_btn = RoundedButton(btn_frame, "⏭  跳过", command=self._on_skip,
                                      width=95, height=48,
                                      font=("Microsoft YaHei UI", 12))
        self.skip_btn.pack(side="left", padx=6)

        # -- 统计栏 --
        stats_frame = tk.Frame(self.root, bg=BG_COLOR)
        stats_frame.pack(pady=(4, 0))

        # 今日
        today_icon = tk.Label(stats_frame, text="🍅", font=("Segoe UI Emoji", 16),
                              fg=TEXT_PRIMARY, bg=BG_COLOR)
        today_icon.pack(side="left", padx=(0, 2))

        self.today_label = tk.Label(stats_frame, text="今日: 0",
                                    font=("Microsoft YaHei UI", 12),
                                    fg=TEXT_PRIMARY, bg=BG_COLOR)
        self.today_label.pack(side="left", padx=(0, 16))

        # 总计
        total_icon = tk.Label(stats_frame, text="📊", font=("Segoe UI Emoji", 16),
                              fg=TEXT_PRIMARY, bg=BG_COLOR)
        total_icon.pack(side="left", padx=(0, 2))

        self.total_label = tk.Label(stats_frame, text="总计: 0",
                                    font=("Microsoft YaHei UI", 12),
                                    fg=TEXT_PRIMARY, bg=BG_COLOR)
        self.total_label.pack(side="left")

        # 刷新统计
        self._update_stats()

    # ---------- 进度环绘制 ----------

    def _draw_ring(self, progress, color):
        """绘制圆形进度环
        Args:
            progress: 0.0 ~ 1.0
            color: 环的颜色
        """
        canvas = self.ring_canvas
        canvas.delete("ring")

        w, h = 260, 260
        pad = 20
        x1, y1 = pad, pad
        x2, y2 = w - pad, h - pad
        extent_angle = progress * 360

        # 背景环
        canvas.create_arc(x1, y1, x2, y2, start=90, extent=-359.99,
                          style="arc", outline=RING_BG, width=14,
                          tags="ring")

        if progress > 0:
            # 进度环
            canvas.create_arc(x1, y1, x2, y2, start=90, extent=-extent_angle,
                              style="arc", outline=color, width=14,
                              tags="ring")

        # 小圆点作为进度指示器
        if 0 < progress < 1:
            angle_rad = math.radians(90 - extent_angle)
            cx, cy = w // 2, h // 2
            r = (w - 2 * pad) // 2
            dot_x = cx + r * math.cos(angle_rad)
            dot_y = cy - r * math.sin(angle_rad)
            canvas.create_oval(dot_x - 7, dot_y - 7, dot_x + 7, dot_y + 7,
                               fill=color, outline="", tags="ring")

    # ---------- 显示更新 ----------

    def _update_display(self, force=False):
        """更新显示的数字和进度环"""
        state = self.timer.state
        remaining = self.timer.remaining
        total = self.timer.total_for_session
        progress = 1.0 - (remaining / total) if total > 0 else 0

        # 确定颜色
        if state in (TimerState.WORKING, TimerState.IDLE):
            color = WORK_COLOR
        elif state == TimerState.PAUSED:
            # 保留当前阶段的颜色 - 从计时器获取
            color = WORK_COLOR  # 默认
        else:
            color = BREAK_COLOR

        # 绘制进度环
        self._draw_ring(progress, color)

        # 更新时间数字
        mins, secs = divmod(remaining, 60)
        time_str = f"{mins:02d}:{secs:02d}"
        self.time_label.config(text=time_str, fg=color)

        # 更新状态标签
        state_texts = {
            TimerState.IDLE: "准备开始",
            TimerState.WORKING: "🍅 专注工作中...",
            TimerState.PAUSED: "⏸ 已暂停",
            TimerState.SHORT_BREAK: "☕ 短休息",
            TimerState.LONG_BREAK: "🌴 长休息",
        }
        self.state_label.config(text=state_texts.get(state, ""))

        # 更新按钮文字
        if state == TimerState.WORKING:
            self.start_btn.set_text("⏸  暂停")
        elif state == TimerState.PAUSED:
            self.start_btn.set_text("▶  继续")
        elif state in (TimerState.SHORT_BREAK, TimerState.LONG_BREAK):
            self.start_btn.set_text("⏸  暂停")
        else:
            self.start_btn.set_text("▶  开始")

        # 调整主按钮颜色
        if state in (TimerState.WORKING, TimerState.PAUSED) or \
           (state in (TimerState.SHORT_BREAK, TimerState.LONG_BREAK)):
            self.start_btn._accent = color
        else:
            self.start_btn._accent = WORK_COLOR

        self.start_btn._draw()

    # ---------- 回调 ----------

    def _on_tick(self, remaining, total):
        """计时器每秒回调"""
        self._update_display()

    def _on_state_change(self, new_state):
        """状态变化回调"""
        self._update_display()

    def _on_complete(self, session_type):
        """一个阶段完成回调 - tick 循环由 _do_tick 自动管理"""
        if session_type == TimerState.WORKING:
            add_completed_pomodoro()
            self._update_stats()
            self._play_completion_sound()

        elif session_type in (TimerState.SHORT_BREAK, TimerState.LONG_BREAK):
            self._play_break_end_sound()

    def _play_completion_sound(self):
        """番茄完成提示音"""
        def play():
            try:
                winsound.Beep(880, 200)
                winsound.Beep(1100, 250)
                winsound.Beep(1320, 300)
            except Exception:
                pass
        threading.Thread(target=play, daemon=True).start()

        # 闪烁任务栏
        self._flash_window()

    def _play_break_end_sound(self):
        """休息结束提示音"""
        def play():
            try:
                winsound.Beep(660, 200)
                winsound.Beep(880, 300)
            except Exception:
                pass
        threading.Thread(target=play, daemon=True).start()

        # 闪烁任务栏
        self._flash_window()

    def _flash_window(self):
        """闪烁任务栏提醒"""
        try:
            self.root.attributes('-topmost', True)
            self.root.update()
            self.root.after(500, lambda: self.root.attributes('-topmost', False))
        except Exception:
            pass

    # ---------- 按钮动作 ----------

    def _on_start(self):
        """开始/暂停/继续按钮"""
        state = self.timer.state

        if state == TimerState.IDLE:
            self.timer.start()
            self._start_tick()

        elif state == TimerState.WORKING:
            self.timer.pause()
            self._stop_tick()

        elif state == TimerState.PAUSED:
            self.timer.resume()
            self._start_tick()

        elif state in (TimerState.SHORT_BREAK, TimerState.LONG_BREAK):
            self.timer.pause()
            self._stop_tick()

        self._update_display()

    def _on_reset(self):
        """重置按钮"""
        if self.timer.state != TimerState.IDLE:
            if messagebox.askyesno("重置", "确定要重置当前计时吗？", parent=self.root):
                self._stop_tick()
                self.timer.reset()
                self._update_display()
        else:
            # 仅在空闲时也支持重置（重新加载设置）
            cfg_durations = get_durations()
            self.timer.set_durations(**cfg_durations)
            self._update_display()

    def _on_skip(self):
        """跳过按钮"""
        state = self.timer.state
        if state == TimerState.IDLE:
            return

        skip_text = "确定要跳过当前阶段吗？"
        if state == TimerState.PAUSED:
            skip_text = "确定要跳过暂停的阶段吗？"

        if messagebox.askyesno("跳过", skip_text, parent=self.root):
            self._stop_tick()
            self.timer.skip()
            self._update_display()
            # 跳过休息完成后自动开始工作
            if self.timer.state in (TimerState.WORKING, TimerState.SHORT_BREAK, TimerState.LONG_BREAK):
                self._start_tick()

    def _open_settings(self):
        """打开设置窗口"""
        def on_save():
            self.timer.reset()
            self._update_display()

        SettingsDialog(self.root, self.timer, on_save=on_save)

    # ---------- Tick 调度 ----------

    def _start_tick(self):
        """启动每秒 tick 循环"""
        self._stop_tick()
        self._do_tick()

    def _stop_tick(self):
        """停止 tick 循环"""
        if self._tick_job:
            self.root.after_cancel(self._tick_job)
            self._tick_job = None

    def _do_tick(self):
        """执行一次 tick，并调度下一次"""
        running = self.timer.tick()

        if running:
            self._tick_job = self.root.after(1000, self._do_tick)
        elif self.timer.state not in (TimerState.IDLE, TimerState.PAUSED):
            # 计时完成并自动过渡到下一阶段（如 工作→休息 或 休息→工作）
            # _on_complete 已被 tick() 内部调用，这里只需重启 tick 循环
            self._update_display()
            self._tick_job = self.root.after(1000, self._do_tick)
        else:
            self._tick_job = None
            self._update_display()

    # ---------- 统计 ----------

    def _update_stats(self):
        """更新统计显示"""
        today = get_today_count()
        total = get_total_count()
        self.today_label.config(text=f"今日: {today}")
        self.total_label.config(text=f"总计: {total}")

    # ---------- 生命周期 ----------

    def _on_close(self):
        """关闭窗口"""
        self._stop_tick()
        self.root.destroy()

    def run(self):
        """启动应用"""
        self.root.mainloop()
