"""
RedAIBot: 红方规则 AI 对手 (含导弹决策)

策略:
    - "straight"    : 直飞平飞
    - "pursuit"     : 经典追踪
    - "energy"      : 能量战术
    - "aggressive"  : 激进追踪 + 导弹
    - "evasive"     : 防御机动 + 反导

动作输出: [aileron, elevator, rudder, throttle, weapon]  (5维)

作者: NetherWind Studio
"""

import math
import numpy as np


def norm_deg(a):
    while a > 180:
        a -= 360
    while a <= -180:
        a += 360
    return a


class RedAIBot:
    """红方规则 AI 控制器 (含导弹)"""

    def __init__(self, policy_name="pursuit"):
        self.policy_name = policy_name
        self._fn = self._get_policy(policy_name)
        self.step_count = 0

    def _get_policy(self, name):
        policies = {
            "straight": self._straight,
            "pursuit": self._pursuit,
            "energy": self._energy,
            "aggressive": self._aggressive,
            "evasive": self._evasive,
        }
        if name not in policies:
            raise ValueError(f"Unknown policy: {name}, available: {list(policies.keys())}")
        return policies[name]

    def act(self, obs):
        """
        obs 字典包含:
            alt, speed_kts, roll, pitch, yaw, dist_nm, dist_ft,
            ata_deg, aa_deg, d_alt_ft, p, nzi,
            missiles_left, incoming_missiles, can_fire

        返回: [aileron, elevator, rudder, throttle, weapon]  (5维)
        """
        self.step_count += 1
        action4 = self._fn(obs)
        # 导弹决策
        weapon = self._weapon_decision(obs)
        return np.array([*action4, weapon], dtype=np.float32)

    def _weapon_decision(self, obs):
        """
        导弹发射决策。
        返回: -1.0(不发射) 或 1.0(发射)
        """
        if not obs.get("can_fire", False):
            return -1.0
        if obs["missiles_left"] <= 0:
            return -1.0

        ata = abs(obs["ata_deg"])
        dist = obs["dist_nm"]

        # 发射条件: ATA < 35度, 距离 1~15 NM
        if ata < 35.0 and 1.0 < dist < 15.0:
            return 1.0
        return -1.0

    # ------------------------------------------------------------------
    # 飞行策略
    # ------------------------------------------------------------------

    def _straight(self, obs):
        e = self._hold_alt(obs["alt"], 20000.0)
        a = self._hold_roll(obs["roll"], 0.0)
        return np.array([a, e, 0.0, 0.6], dtype=np.float32)

    def _pursuit(self, obs):
        """经典追踪: 向敌机方向大角度滚转转弯"""
        ata = obs["ata_deg"]
        dist = obs["dist_nm"]

        # 大滚转追踪 (最大 75度)
        if abs(ata) > 8:
            target_roll = np.clip(ata * 2.5, -75.0, 75.0)
        else:
            target_roll = 0.0

        aileron = self._hold_roll(obs["roll"], target_roll, kp=0.08)

        # 高度跟踪 + 适度拉杆提高转弯率
        target_alt = 20000.0 + obs["d_alt_ft"] * 0.3
        elevator = self._hold_alt(obs["alt"], target_alt)

        # 滚转时配合拉杆 (协调转弯)
        if abs(obs["roll"]) > 15:
            elevator -= 0.15  # 拉杆辅助转弯
            elevator = np.clip(elevator, -0.6, 0.3)

        # 方向舵配合
        rudder = -aileron * 0.5

        # 油门
        throttle = 0.9 if dist > 5 else (0.7 if dist > 2 else 0.6)

        return np.array([aileron, elevator, rudder, throttle], dtype=np.float32)

    def _energy(self, obs):
        """能量战术: 柔和滚转，优先保高度和速度"""
        ata = obs["ata_deg"]
        speed = obs["speed_kts"]
        alt = obs["alt"]

        target_roll = np.clip(ata * 1.2, -35.0, 35.0)
        aileron = self._hold_roll(obs["roll"], target_roll, kp=0.06)

        target_alt = max(22000.0, 20000.0 + obs["d_alt_ft"])
        elevator = self._hold_alt(alt, target_alt)

        rudder = -aileron * 0.2

        if speed < 400:
            throttle = 1.0
        elif speed < 450:
            throttle = 0.8
        else:
            throttle = 0.6

        return np.array([aileron, elevator, rudder, throttle], dtype=np.float32)

    def _aggressive(self, obs):
        """激进追踪: 最大滚转 + 高G转弯 + 积极发射导弹"""
        ata = obs["ata_deg"]
        dist = obs["dist_nm"]

        # 最大努力转弯 (最大 80度滚转)
        if abs(ata) > 5:
            target_roll = np.clip(ata * 3.5, -80.0, 80.0)
        else:
            target_roll = 0.0

        aileron = self._hold_roll(obs["roll"], target_roll, kp=0.1)

        # 近距追踪时大拉杆
        if abs(ata) < 45 and dist < 3.0:
            elevator = -0.4  # 大拉杆做高G转弯
        else:
            elevator = self._hold_alt(obs["alt"], 20000.0 + obs["d_alt_ft"] * 0.5)

        # 强滚转时大拉杆
        if abs(obs["roll"]) > 30:
            elevator = min(elevator, -0.2)

        rudder = -aileron * 0.5
        throttle = 0.95 if dist < 3.0 else 0.85

        return np.array([aileron, elevator, rudder, throttle], dtype=np.float32)

    def _evasive(self, obs):
        """防御机动: 规避来袭导弹 + 被咬尾时急转"""
        ata = obs["ata_deg"]
        aa = obs["aa_deg"]
        dist = obs["dist_nm"]
        incoming = obs.get("incoming_missiles", 0)

        is_pursued = (abs(aa) < 45 and dist < 5.0)

        if incoming > 0 or is_pursued:
            # 急转: 向敌机反方向最大滚转
            target_roll = -75.0 if ata > 0 else 75.0
            aileron = self._hold_roll(obs["roll"], target_roll, kp=0.12)
            elevator = -0.5  # 大拉杆
            throttle = 1.0   # 全加力
            rudder = -aileron * 0.6
        else:
            return self._energy(obs)

        return np.array([aileron, elevator, rudder, throttle], dtype=np.float32)

    # ------------------------------------------------------------------
    # PID 辅助
    # ------------------------------------------------------------------

    def _hold_roll(self, current, target, kp=0.08):
        return np.clip((target - current) * kp, -1.0, 1.0)

    def _hold_alt(self, current, target, kp=0.0003):
        return np.clip(-(target - current) * kp, -0.6, 0.4)
