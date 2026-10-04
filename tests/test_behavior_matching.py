"""
玩家行为分析与匹配系统 - 验收测试
运行方式：python -m pytest tests/test_behavior_matching.py -q
共 16 个测试用例
"""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.behavior_matching import BehaviorMatching


def make_player(pid, rank="gold", win_rate=0.5, kda=1.5, accuracy=0.5,
                headshot_rate=0.1, role="carry", ping=30, **extra):
    player = {
        "id": pid,
        "rank": rank,
        "win_rate": win_rate,
        "kda": kda,
        "accuracy": accuracy,
        "headshot_rate": headshot_rate,
        "role": role,
        "ping": ping,
    }
    player.update(extra)
    return player


class TestBehaviorMatching:
    def setup_method(self):
        self.system = BehaviorMatching()

    def test_case_01(self):
        # 技能计算是多维度的：同段位下胜率/KDA/命中率/爆头率更高者技能更高
        weak = make_player(1, win_rate=0.4, kda=1.0, accuracy=0.3,
                           headshot_rate=0.05)
        strong = make_player(2, win_rate=0.6, kda=3.0, accuracy=0.7,
                             headshot_rate=0.3)
        assert self.system.calculate_skill(strong) > self.system.calculate_skill(weak)

    def test_case_02(self):
        # 每个维度都独立影响技能分
        base = make_player(1)
        base_skill = self.system.calculate_skill(base)
        for key, value in (("win_rate", 0.7), ("kda", 4.0),
                           ("accuracy", 0.8), ("headshot_rate", 0.4)):
            boosted = make_player(1, **{key: value})
            assert self.system.calculate_skill(boosted) > base_skill

    def test_case_03(self):
        # 近期表现权重更高：同样的战绩，越近影响越大
        now = time.time()
        wins = [{"win": True, "kda": 3.0} for _ in range(5)]
        recent = make_player(1, recent_matches=[
            dict(m, timestamp=now) for m in wins])
        stale = make_player(2, recent_matches=[
            dict(m, timestamp=now - 30 * 86400) for m in wins])
        baseline = make_player(3)
        skill_recent = self.system.calculate_skill(recent, now=now)
        skill_stale = self.system.calculate_skill(stale, now=now)
        skill_base = self.system.calculate_skill(baseline, now=now)
        assert skill_recent > skill_stale
        assert abs(skill_stale - skill_base) < 0.2 * abs(skill_recent - skill_base)

    def test_case_04(self):
        # 近期连胜与近期连败的技能分应明显拉开
        now = time.time()
        streak_win = make_player(1, recent_matches=[
            {"win": True, "kda": 3.0, "timestamp": now} for _ in range(5)])
        streak_lose = make_player(2, recent_matches=[
            {"win": False, "kda": 0.8, "timestamp": now} for _ in range(5)])
        assert (self.system.calculate_skill(streak_win, now=now)
                > self.system.calculate_skill(streak_lose, now=now))

    def test_case_05(self):
        # 技能计算区分位置：相同数据下不同位置的评分不同
        carry = make_player(1, role="carry")
        support = make_player(2, role="support")
        tank = make_player(3, role="tank")
        skills = {self.system.calculate_skill(p) for p in (carry, support, tank)}
        assert len(skills) == 3

    def test_case_06(self):
        # 匹配结果两队总技能应接近（团队平衡生效）
        players = [make_player(i, rank=r, win_rate=0.5 + i * 0.01)
                   for i, r in enumerate(
                       ["bronze", "silver", "gold", "gold", "platinum",
                        "platinum", "diamond", "diamond", "master", "silver"])]
        result = self.system.matchmaking(players)
        skill_a = sum(self.system.calculate_skill(p) for p in result["team_a"])
        skill_b = sum(self.system.calculate_skill(p) for p in result["team_b"])
        assert abs(skill_a - skill_b) <= 0.1 * max(skill_a, skill_b)

    def test_case_07(self):
        # 匹配保证角色互补：任何一队都不应全是同一位置
        players = [make_player(i, role="carry") for i in range(6)]
        players += [make_player(6, role="support"), make_player(7, role="support"),
                    make_player(8, role="tank"), make_player(9, role="tank")]
        result = self.system.matchmaking(players)
        for team in (result["team_a"], result["team_b"]):
            roles = {p["role"] for p in team}
            assert len(roles) >= 2

    def test_case_08(self):
        # 网络优化：候选充足时优先把延迟相近的玩家匹进同一场
        low_ping = [make_player(i, ping=20 + i) for i in range(10)]
        high_ping = [make_player(10 + i, ping=200 + i) for i in range(10)]
        result = self.system.matchmaking(low_ping + high_ping)
        assert result["ping_spread"] <= 50

    def test_case_09(self):
        # team_balance 实际生效：优于简单的强弱对半切分
        players = [make_player(i, win_rate=0.3 + i * 0.05) for i in range(10)]
        team_a, team_b = self.system.team_balance(players)
        skill_a = sum(self.system.calculate_skill(p) for p in team_a)
        skill_b = sum(self.system.calculate_skill(p) for p in team_b)
        naive = sorted(players, key=lambda p: p["win_rate"])
        naive_diff = abs(sum(self.system.calculate_skill(p) for p in naive[:5])
                         - sum(self.system.calculate_skill(p) for p in naive[5:]))
        assert abs(skill_a - skill_b) < naive_diff

    def test_case_10(self):
        # team_balance 的角色分配：双方都应具备位置多样性
        players = ([make_player(i, role="carry") for i in range(4)]
                   + [make_player(4 + i, role="support") for i in range(4)]
                   + [make_player(8, role="tank"), make_player(9, role="tank")])
        team_a, team_b = self.system.team_balance(players)
        for team in (team_a, team_b):
            assert len({p["role"] for p in team}) >= 2

    def test_case_11(self):
        # 替补机制：有人退出后从候选池补入同位置且技能接近的玩家
        team = [make_player(i, role="carry") for i in range(4)]
        leaver = make_player(4, role="support", win_rate=0.55)
        team.append(leaver)
        pool = [make_player(5, role="carry", win_rate=0.9),
                make_player(6, role="support", win_rate=0.55)]
        substitute = self.system.substitute_player(team, leaver, pool)
        assert leaver not in team
        assert substitute in team and len(team) == 5
        assert substitute["role"] == "support"

    def test_case_12(self):
        # 行为分析识别游戏风格
        aggressive = make_player(1, kda=4.5, headshot_rate=0.4)
        supportive = make_player(2, role="support", assists=12.0)
        passive = make_player(3, kda=0.8, accuracy=0.2)
        assert self.system.behavior_analysis(aggressive)["style"] == "aggressive"
        assert self.system.behavior_analysis(supportive)["style"] == "supportive"
        assert self.system.behavior_analysis(passive)["style"] == "passive"

    def test_case_13(self):
        # 异常行为检测：疑似开挂者被标记，正常玩家不被误伤
        cheater = make_player(1, accuracy=0.95, headshot_rate=0.9, kda=12.0)
        normal = make_player(2, accuracy=0.5, headshot_rate=0.2, kda=2.0)
        cheater_result = self.system.behavior_analysis(cheater)
        normal_result = self.system.behavior_analysis(normal)
        assert cheater_result["is_anomalous"]
        assert cheater_result["anomaly_score"] > normal_result["anomaly_score"]
        assert not normal_result["is_anomalous"]

    def test_case_14(self):
        # 学习能力：持续战胜更强对手后，系统应上调该玩家的技能评估
        player = make_player(1, rank="gold")
        before = self.system.calculate_skill(player)
        for _ in range(10):
            self.system.learn({"player_id": 1, "won": True,
                               "own_skill": before, "opponent_skill": before + 300})
        after = self.system.calculate_skill(player)
        assert after > before
        assert len(self.system._history) == 10

    def test_case_15(self):
        # 超时梯度：等待越久匹配范围越宽，且单调放宽
        tiers = [self.system.match_timeout(t)
                 for t in (0, 30, 60, 120, 300)]
        assert tiers[0]["tier"] != tiers[-1]["tier"]
        for prev, curr in zip(tiers, tiers[1:]):
            assert curr["skill_tolerance"] >= prev["skill_tolerance"]
            assert curr["ping_tolerance"] >= prev["ping_tolerance"]
        assert tiers[-1]["skill_tolerance"] > tiers[0]["skill_tolerance"]

    def test_case_16(self):
        # 超时通知：告知玩家已等待时长；完整匹配流程可用
        timeout = self.system.match_timeout(45)
        assert timeout["should_notify"]
        assert "45" in timeout["notification"]
        players = [make_player(i) for i in range(10)]
        result = self.system.matchmaking(players, elapsed=45)
        assert len(result["team_a"]) == 5 and len(result["team_b"]) == 5
        assert 0.0 <= result["quality"] <= 1.0
        assert result["timeout"]["notification"] == timeout["notification"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
