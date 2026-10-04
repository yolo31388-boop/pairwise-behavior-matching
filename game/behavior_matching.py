"""
玩家行为分析与匹配系统 - 核心模块

功能：
- calculate_skill: 多维度技能评估（段位/胜率/KDA/命中率/爆头率），
  近期表现指数衰减加权，位置（角色）差异化权重
- matchmaking: 综合考虑团队平衡、角色互补与网络延迟的匹配
- team_balance: 蛇形分边 + 角色再平衡，支持替补补位
- behavior_analysis: 游戏风格识别、异常行为检测、匹配质量学习
- match_timeout: 匹配超时梯度放宽与等待通知
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


RANK_SCORES = {
    "bronze": 100.0,
    "silver": 200.0,
    "gold": 350.0,
    "platinum": 500.0,
    "diamond": 700.0,
    "master": 900.0,
}

# 各位置的五维权重：(段位, 胜率, KDA, 命中率, 爆头率)
ROLE_WEIGHTS = {
    "carry": (0.30, 0.15, 0.20, 0.20, 0.15),
    "support": (0.30, 0.30, 0.15, 0.15, 0.10),
    "tank": (0.30, 0.20, 0.20, 0.22, 0.08),
}
DEFAULT_ROLE_WEIGHTS = (0.30, 0.20, 0.20, 0.18, 0.12)

RECENCY_HALFLIFE_DAYS = 7.0
RECENCY_MAX_BLEND = 0.4

# 超时梯度：(等待秒数阈值, 技能容差, 延迟容差ms)
TIMEOUT_TIERS = [
    (0, 50.0, 30.0),
    (30, 100.0, 60.0),
    (60, 200.0, 100.0),
    (120, 350.0, 150.0),
    (300, 1000.0, 300.0),
]

ELO_K = 32.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class Config:
    team_size: int = 5


class BehaviorMatching:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []

    def update(self, dt: float):
        self._state["wait_time"] = self._state.get("wait_time", 0.0) + dt

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 技能计算：多维度 + 近期表现 + 位置
    # ------------------------------------------------------------------
    def calculate_skill(self, player: Dict, now: Optional[float] = None) -> float:
        rank = str(player.get("rank", "bronze")).lower()
        rank_score = RANK_SCORES.get(rank, RANK_SCORES["bronze"])
        win_rate = _clamp01(player.get("win_rate", 0.5)) * 1000.0
        kda_score = min(max(float(player.get("kda", 1.0)), 0.0), 8.0) / 8.0 * 1000.0
        accuracy = _clamp01(player.get("accuracy", 0.5)) * 1000.0
        headshot = _clamp01(player.get("headshot_rate", 0.1)) * 1000.0

        role = str(player.get("role", "carry")).lower()
        w_rank, w_wr, w_kda, w_acc, w_hs = ROLE_WEIGHTS.get(role, DEFAULT_ROLE_WEIGHTS)
        base = (rank_score * w_rank + win_rate * w_wr + kda_score * w_kda
                + accuracy * w_acc + headshot * w_hs)

        skill = base
        recent = player.get("recent_matches") or []
        if recent:
            now = time.time() if now is None else now
            total_weight = 0.0
            weighted_perf = 0.0
            for match in recent:
                age_days = max(0.0, (now - float(match.get("timestamp", now))) / 86400.0)
                weight = 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)
                perf = (600.0 if match.get("win") else 300.0) \
                    + min(max(float(match.get("kda", 1.0)), 0.0), 8.0) * 25.0
                weighted_perf += weight * perf
                total_weight += weight
            if total_weight > 0:
                recent_perf = weighted_perf / total_weight
                # 比赛越久远总权重越小，对当前技能的影响也越小
                blend = RECENCY_MAX_BLEND * min(1.0, total_weight / 2.0)
                skill = (1.0 - blend) * base + blend * recent_perf

        adjustments = self._state.get("skill_adjust", {})
        skill += adjustments.get(player.get("id"), 0.0)
        return skill

    # ------------------------------------------------------------------
    # 匹配：团队平衡 + 角色互补 + 网络优化
    # ------------------------------------------------------------------
    def matchmaking(self, players: List[Dict], team_size: int = 5,
                    elapsed: float = 0.0) -> Dict:
        timeout = self.match_timeout(elapsed)
        candidates = list(players)

        # 网络优化：候选超过需求时，选取延迟离散度最小的一组
        needed = team_size * 2
        if len(candidates) > needed:
            candidates.sort(key=lambda p: p.get("ping", 50))
            best_start = min(
                range(len(candidates) - needed + 1),
                key=lambda i: candidates[i + needed - 1].get("ping", 50)
                - candidates[i].get("ping", 50),
            )
            candidates = candidates[best_start:best_start + needed]

        team_a, team_b = self.team_balance(candidates, team_size)
        pings = [p.get("ping", 50) for p in candidates] or [0]
        return {
            "team_a": team_a,
            "team_b": team_b,
            "quality": self._match_quality(team_a, team_b, team_size),
            "ping_spread": max(pings) - min(pings),
            "timeout": timeout,
        }

    def _match_quality(self, team_a: List[Dict], team_b: List[Dict],
                       team_size: int) -> float:
        skill_a = sum(self.calculate_skill(p) for p in team_a)
        skill_b = sum(self.calculate_skill(p) for p in team_b)
        balance = 1.0 - abs(skill_a - skill_b) / max(skill_a, skill_b, 1.0)
        roles_a = len({p.get("role", "carry") for p in team_a})
        roles_b = len({p.get("role", "carry") for p in team_b})
        coverage = (min(roles_a, team_size) + min(roles_b, team_size)) / (2.0 * team_size)
        role_weight = self._state.get("role_weight", 0.3)
        quality = balance * (1.0 - role_weight) + coverage * role_weight
        return max(0.0, min(1.0, quality))

    # ------------------------------------------------------------------
    # 团队平衡：蛇形分边 + 角色再平衡 + 替补
    # ------------------------------------------------------------------
    def team_balance(self, players: List[Dict],
                     team_size: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        team_size = team_size or max(1, len(players) // 2)
        ranked = sorted(players, key=lambda p: self.calculate_skill(p), reverse=True)
        team_a: List[Dict] = []
        team_b: List[Dict] = []
        for index, player in enumerate(ranked):
            # 蛇形分边：A B B A A B B A ...
            if index % 4 in (0, 3):
                team_a.append(player)
            else:
                team_b.append(player)
        self._rebalance_roles(team_a, team_b, team_size)
        return team_a, team_b

    def _rebalance_roles(self, team_a: List[Dict], team_b: List[Dict],
                         team_size: int) -> None:
        limit = team_size // 2 + 1
        for _ in range(team_size * 2):
            swapped = False
            for team, other in ((team_a, team_b), (team_b, team_a)):
                counts = Counter(p.get("role", "carry") for p in team)
                other_counts = Counter(p.get("role", "carry") for p in other)
                for role, count in counts.items():
                    if count <= limit:
                        continue
                    best_pair = None
                    best_cost = None
                    for out_player in team:
                        if out_player.get("role", "carry") != role:
                            continue
                        for in_player in other:
                            in_role = in_player.get("role", "carry")
                            if in_role == role or other_counts.get(in_role, 0) <= 1:
                                continue
                            if counts.get(in_role, 0) >= count - 1:
                                continue
                            cost = abs(self.calculate_skill(out_player)
                                       - self.calculate_skill(in_player))
                            if best_cost is None or cost < best_cost:
                                best_cost = cost
                                best_pair = (out_player, in_player)
                    if best_pair:
                        out_player, in_player = best_pair
                        team.remove(out_player)
                        other.remove(in_player)
                        team.append(in_player)
                        other.append(out_player)
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                return

    def substitute_player(self, team: List[Dict], leaver: Dict,
                          pool: List[Dict]) -> Dict:
        """玩家退出后从候选池补人：优先同位置，其次技能最接近。"""
        if not pool:
            raise ValueError("候选池为空，无法补人")
        leaver_role = leaver.get("role", "carry")
        leaver_skill = self.calculate_skill(leaver)
        substitute = min(
            pool,
            key=lambda p: (p.get("role", "carry") != leaver_role,
                           abs(self.calculate_skill(p) - leaver_skill)),
        )
        if leaver in team:
            team.remove(leaver)
        team.append(substitute)
        pool.remove(substitute)
        return substitute

    # ------------------------------------------------------------------
    # 行为分析：风格识别 + 异常检测 + 学习
    # ------------------------------------------------------------------
    def behavior_analysis(self, player: Dict) -> Dict:
        kda = max(0.0, float(player.get("kda", 1.0)))
        accuracy = _clamp01(player.get("accuracy", 0.5))
        headshot = _clamp01(player.get("headshot_rate", 0.1))
        win_rate = _clamp01(player.get("win_rate", 0.5))
        assists = max(0.0, float(player.get("assists", 0.0)))
        reports = max(0, int(player.get("reports", 0)))

        if kda >= 3.0 and headshot >= 0.3:
            style = "aggressive"
        elif assists >= 8.0 or str(player.get("role", "")).lower() == "support":
            style = "supportive"
        elif kda < 1.5 and accuracy < 0.4:
            style = "passive"
        else:
            style = "balanced"

        flags = []
        anomaly_score = 0.0
        if accuracy > 0.85:
            anomaly_score += 0.4
            flags.append("accuracy_abnormal")
        if headshot > 0.7:
            anomaly_score += 0.4
            flags.append("headshot_abnormal")
        if kda > 8.0:
            anomaly_score += 0.3
            flags.append("kda_abnormal")
        if win_rate > 0.9:
            anomaly_score += 0.2
            flags.append("win_rate_abnormal")
        if reports > 5:
            anomaly_score += 0.3
            flags.append("frequently_reported")
        anomaly_score = min(1.0, anomaly_score)

        return {
            "style": style,
            "anomaly_score": anomaly_score,
            "is_anomalous": anomaly_score >= 0.6,
            "flags": flags,
        }

    def learn(self, feedback: Dict) -> None:
        """从比赛结果学习（ELO 式修正），使技能评估与匹配质量随时间提升。"""
        player_id = feedback.get("player_id")
        if player_id is None:
            return
        own_skill = float(feedback.get("own_skill", 500.0))
        opponent_skill = float(feedback.get("opponent_skill", own_skill))
        won = bool(feedback.get("won", False))
        expected = 1.0 / (1.0 + 10.0 ** (-(own_skill - opponent_skill) / 400.0))
        actual = 1.0 if won else 0.0
        adjustments = self._state.setdefault("skill_adjust", {})
        adjustments[player_id] = adjustments.get(player_id, 0.0) + ELO_K * (actual - expected)
        self._history.append({
            "type": "learn",
            "player_id": player_id,
            "won": won,
            "adjustment": adjustments[player_id],
        })

    # ------------------------------------------------------------------
    # 匹配超时：梯度放宽 + 通知
    # ------------------------------------------------------------------
    def match_timeout(self, elapsed: float) -> Dict:
        elapsed = max(0.0, float(elapsed))
        tier = 0
        for index, (threshold, _, _) in enumerate(TIMEOUT_TIERS):
            if elapsed >= threshold:
                tier = index
        _, skill_tolerance, ping_tolerance = TIMEOUT_TIERS[tier]
        return {
            "elapsed": elapsed,
            "tier": tier,
            "skill_tolerance": skill_tolerance,
            "ping_tolerance": ping_tolerance,
            "should_notify": True,
            "notification": "匹配等待中：已等待 {} 秒，匹配范围第 {} 档".format(
                int(elapsed), tier + 1),
        }
