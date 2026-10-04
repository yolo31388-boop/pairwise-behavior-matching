"""
玩家行为分析与匹配系统 - 核心模块

功能：
- calculate_skill:   多维度技能计算（段位 + 胜率/KDA/命中率/爆头率 + 最近表现 + 位置）
- matchmaking:       匹配算法（团队平衡 + 角色互补 + 网络优化 + 超时梯度）
- team_balance:      团队平衡（蛇形选秀 + 角色修正）
- substitute_player: 替补机制（玩家退出后按位置和技能补人）
- behavior_analysis: 行为分析（游戏风格 + 异常检测 + 增量学习）
- match_timeout:     匹配超时梯度处理与通知
"""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional
import time


RANK_SCORES = {
    "bronze": 1000.0,
    "silver": 1300.0,
    "gold": 1600.0,
    "platinum": 1900.0,
    "diamond": 2200.0,
    "master": 2500.0,
}

# 不同位置对各数据维度的权重（每行之和为 1）
ROLE_WEIGHTS = {
    "carry":   {"win_rate": 0.20, "kda": 0.35, "accuracy": 0.30, "headshot": 0.15},
    "mid":     {"win_rate": 0.25, "kda": 0.30, "accuracy": 0.25, "headshot": 0.20},
    "jungle":  {"win_rate": 0.30, "kda": 0.30, "accuracy": 0.20, "headshot": 0.20},
    "tank":    {"win_rate": 0.35, "kda": 0.25, "accuracy": 0.25, "headshot": 0.15},
    "support": {"win_rate": 0.40, "kda": 0.25, "accuracy": 0.20, "headshot": 0.15},
}

MAX_SAME_ROLE_PER_TEAM = 2

# 超时梯度：(等待上限秒数, 可接受技能差, 档位名)
TIMEOUT_TIERS = (
    (30.0, 100.0, "strict"),
    (60.0, 200.0, "relaxed"),
    (120.0, 400.0, "loose"),
    (float("inf"), 800.0, "maximum"),
)

ABNORMAL_THRESHOLDS = {
    "headshot_rate": (0.85, "爆头率异常偏高"),
    "accuracy": (0.95, "命中率异常偏高"),
    "kda": (8.0, "KDA异常偏高"),
}


def _clamp(value, low, high):
    return max(low, min(high, value))


@dataclass
class Config:
    pass


class BehaviorMatching:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._learned = {}
        self._behavior_profiles = {}

    def update(self, dt: float):
        self._state["elapsed"] = self._state.get("elapsed", 0.0) + dt

    def reset(self):
        self._state = {}
        self._history = []
        self._learned = {}
        self._behavior_profiles = {}

    # ------------------------------------------------------------------
    # 技能计算：多维度 + 最近表现 + 位置
    # ------------------------------------------------------------------
    def calculate_skill(self, player: Dict, now: Optional[float] = None) -> float:
        rank = str(player.get("rank", "bronze")).lower()
        base = RANK_SCORES.get(rank, RANK_SCORES["bronze"])

        win_rate = _clamp(float(player.get("win_rate", 0.5)), 0.0, 1.0)
        kda = max(0.0, float(player.get("kda", 2.0)))
        accuracy = _clamp(float(player.get("accuracy", 0.5)), 0.0, 1.0)
        headshot = _clamp(float(player.get("headshot_rate", 0.2)), 0.0, 1.0)

        factors = {
            "win_rate": (win_rate - 0.5) * 2.0,
            "kda": _clamp((kda - 2.0) / 3.0, -1.0, 1.0),
            "accuracy": (accuracy - 0.5) * 2.0,
            "headshot": _clamp((headshot - 0.2) / 0.3, -1.0, 1.0),
        }
        role = str(player.get("role", "carry")).lower()
        weights = ROLE_WEIGHTS.get(role, ROLE_WEIGHTS["carry"])
        adjust = sum(weights[key] * factors[key] for key in weights)

        recent = self._recent_performance(player, now)
        recency_weight = float(self.config.get("recency_weight", 0.15))

        learned = self._learned.get(player.get("player_id"), 0.0)

        skill = base * (1.0 + adjust) * (1.0 + recency_weight * recent) * (1.0 + learned)
        return max(1.0, skill)

    def _recent_performance(self, player: Dict, now: Optional[float] = None) -> float:
        """最近表现因子 [-1, 1]，按指数衰减加权，陈旧数据影响趋近于 0。"""
        matches = player.get("recent_matches") or []
        if not matches:
            return 0.0
        now = time.time() if now is None else now
        half_life = float(self.config.get("recent_half_life_days", 7.0)) * 86400.0
        total = 0.0
        for match in matches:
            age = max(0.0, now - float(match.get("timestamp", now)))
            weight = 0.5 ** (age / half_life)
            score = 1.0 if match.get("won") else -1.0
            if "kda" in match:
                score += _clamp((float(match["kda"]) - 2.0) / 3.0, -1.0, 1.0)
            total += weight * score
        return _clamp(total / (2.0 * len(matches)), -1.0, 1.0)

    # ------------------------------------------------------------------
    # 匹配算法：团队平衡 + 角色互补 + 网络优化 + 超时梯度
    # ------------------------------------------------------------------
    def matchmaking(self, players: List[Dict], team_size: int = 5,
                    wait_seconds: float = 0.0) -> Optional[Dict]:
        players = list(players or [])
        if len(players) < team_size * 2:
            return None
        timeout = self.match_timeout(wait_seconds)
        selected = self._select_players(players, team_size, timeout["max_skill_diff"])
        result = self.team_balance(selected, team_size=team_size)

        latencies = [p.get("ping", 50) for p in selected]
        latency_spread = (max(latencies) - min(latencies)) if latencies else 0.0
        total_skill = result["team_a_skill"] + result["team_b_skill"]
        quality = 1.0
        if total_skill > 0:
            quality -= 0.5 * min(1.0, result["skill_diff"] / total_skill)
        quality -= 0.1 * self._role_penalty([result["team_a"], result["team_b"]])
        quality -= 0.2 * min(1.0, latency_spread / 300.0)
        quality = _clamp(quality, 0.0, 1.0)

        result.update({
            "quality": quality,
            "latency_spread": latency_spread,
            "tier": timeout["tier"],
            "max_skill_diff": timeout["max_skill_diff"],
            "notification": timeout["notification"],
        })
        self._history.append({
            "event": "matchmaking",
            "quality": quality,
            "timestamp": time.time(),
        })
        return result

    def _select_players(self, players: List[Dict], team_size: int,
                        max_skill_diff: float) -> List[Dict]:
        """从候选中选出 2*team_size 人：技能差在梯度范围内，且延迟差最小。"""
        need = team_size * 2
        if len(players) <= need:
            return list(players)
        ordered = sorted(players,
                         key=lambda p: (self.calculate_skill(p), p.get("ping", 50)))
        windows = []
        for start in range(len(ordered) - need + 1):
            window = ordered[start:start + need]
            skill_spread = (self.calculate_skill(window[-1])
                            - self.calculate_skill(window[0]))
            pings = [p.get("ping", 50) for p in window]
            windows.append((skill_spread, max(pings) - min(pings), window))
        feasible = [w for w in windows if w[0] <= max_skill_diff]
        pool = feasible if feasible else windows
        pool.sort(key=lambda w: (w[1], w[0]))
        return list(pool[0][2])

    # ------------------------------------------------------------------
    # 团队平衡：蛇形选秀 + 角色修正
    # ------------------------------------------------------------------
    def team_balance(self, players: List[Dict], team_size: int = 5) -> Optional[Dict]:
        players = list(players or [])
        if len(players) < team_size * 2:
            return None
        ordered = sorted(players, key=lambda p: self.calculate_skill(p),
                         reverse=True)
        team_a, team_b = [], []
        for idx, player in enumerate(ordered[:team_size * 2]):
            if idx % 4 in (0, 3):
                team_a.append(player)
            else:
                team_b.append(player)
        self._fix_roles(team_a, team_b)
        skill_a = sum(self.calculate_skill(p) for p in team_a)
        skill_b = sum(self.calculate_skill(p) for p in team_b)
        return {
            "team_a": team_a,
            "team_b": team_b,
            "team_a_skill": skill_a,
            "team_b_skill": skill_b,
            "skill_diff": abs(skill_a - skill_b),
        }

    def _role_penalty(self, teams: List[List[Dict]]) -> int:
        penalty = 0
        for team in teams:
            counts = Counter(p.get("role", "carry") for p in team)
            penalty += sum(max(0, c - MAX_SAME_ROLE_PER_TEAM)
                           for c in counts.values())
        return penalty

    def _fix_roles(self, team_a: List[Dict], team_b: List[Dict]) -> None:
        """交换双方队员，消除同队 3 个及以上同位置的情况。"""
        for _ in range(50):
            penalty = self._role_penalty([team_a, team_b])
            if penalty == 0:
                return
            best = None
            for i, pa in enumerate(team_a):
                for j, pb in enumerate(team_b):
                    if pa.get("role", "carry") == pb.get("role", "carry"):
                        continue
                    team_a[i], team_b[j] = pb, pa
                    new_penalty = self._role_penalty([team_a, team_b])
                    team_a[i], team_b[j] = pa, pb
                    if new_penalty >= penalty:
                        continue
                    delta = abs(self.calculate_skill(pa) - self.calculate_skill(pb))
                    if best is None or (new_penalty, delta) < (best[0], best[1]):
                        best = (new_penalty, delta, i, j)
            if best is None:
                return
            _, _, i, j = best
            team_a[i], team_b[j] = team_b[j], team_a[i]

    def substitute_player(self, team: List[Dict], leaving_player_id,
                          candidates: List[Dict]) -> Optional[Dict]:
        """玩家退出后，从候选池中选择位置相同、技能最接近的替补。"""
        leaving = None
        for player in team:
            if player.get("player_id") == leaving_player_id:
                leaving = player
                break
        if leaving is None:
            return None
        team.remove(leaving)
        candidates = [c for c in (candidates or [])
                      if c.get("player_id") != leaving_player_id]
        if not candidates:
            return None
        leaving_skill = self.calculate_skill(leaving)
        leaving_role = leaving.get("role", "carry")
        best = min(candidates, key=lambda c: (
            0 if c.get("role", "carry") == leaving_role else 1,
            abs(self.calculate_skill(c) - leaving_skill),
        ))
        team.append(best)
        return best

    # ------------------------------------------------------------------
    # 行为分析：游戏风格 + 异常检测 + 增量学习
    # ------------------------------------------------------------------
    def behavior_analysis(self, player: Dict) -> Dict:
        player_id = player.get("player_id")
        role = str(player.get("role", "carry")).lower()
        kda = float(player.get("kda", 2.0))
        assists = float(player.get("assists", 0.0))
        deaths = float(player.get("deaths", 5.0))

        if role == "support" or assists >= 8.0:
            style = "supportive"
        elif kda >= 3.0:
            style = "aggressive"
        elif deaths <= 3.0:
            style = "defensive"
        else:
            style = "balanced"

        reasons = []
        for key, (threshold, reason) in ABNORMAL_THRESHOLDS.items():
            if float(player.get(key, 0.0)) > threshold:
                reasons.append(reason)

        profile = self._behavior_profiles.setdefault(
            player_id, {"observations": 0})
        profile["observations"] += 1
        profile["style"] = style
        confidence = min(1.0, profile["observations"] / 10.0)

        return {
            "player_id": player_id,
            "style": style,
            "is_abnormal": bool(reasons),
            "abnormal_reasons": reasons,
            "confidence": confidence,
            "observations": profile["observations"],
        }

    def record_match_result(self, winner_ids: List, loser_ids: List) -> None:
        """从比赛结果中学习，修正玩家技能评估，匹配质量随时间提升。"""
        rate = float(self.config.get("learning_rate", 0.02))
        for pid in winner_ids or []:
            self._learned[pid] = _clamp(self._learned.get(pid, 0.0) + rate,
                                        -0.2, 0.2)
        for pid in loser_ids or []:
            self._learned[pid] = _clamp(self._learned.get(pid, 0.0) - rate,
                                        -0.2, 0.2)
        self._history.append({
            "event": "match_result",
            "winners": list(winner_ids or []),
            "losers": list(loser_ids or []),
            "timestamp": time.time(),
        })

    # ------------------------------------------------------------------
    # 匹配超时：梯度扩大匹配范围 + 等待通知
    # ------------------------------------------------------------------
    def match_timeout(self, wait_seconds: float) -> Dict:
        wait = max(0.0, float(wait_seconds))
        for limit, max_diff, tier in TIMEOUT_TIERS:
            if wait < limit:
                return {
                    "wait_seconds": wait,
                    "tier": tier,
                    "max_skill_diff": max_diff,
                    "notification": (
                        "匹配中：已等待 {} 秒，当前可接受技能差 ±{}".format(
                            int(wait), int(max_diff))
                    ),
                }
