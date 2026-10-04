"""
玩家行为分析与匹配系统 - 验收测试
运行方式：python -m pytest tests/test_behavior_matching.py -q
共 16 个测试用例
"""
import pytest
import sys
import os
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.behavior_matching import BehaviorMatching


def make_player(pid, rank="gold", win_rate=0.5, kda=2.0, accuracy=0.5,
                headshot_rate=0.2, role="carry", ping=30, recent_matches=None,
                assists=0.0, deaths=5.0):
    return {
        "player_id": pid,
        "rank": rank,
        "win_rate": win_rate,
        "kda": kda,
        "accuracy": accuracy,
        "headshot_rate": headshot_rate,
        "role": role,
        "ping": ping,
        "recent_matches": recent_matches or [],
        "assists": assists,
        "deaths": deaths,
    }


class TestBehaviorMatching:
    def setup_method(self):
        self.system = BehaviorMatching()

    # ---- 技能计算：多维度 ----
    def test_case_01(self):
        """同段位下，胜率/KDA/命中率/爆头率越高技能越高。"""
        base = self.system.calculate_skill(make_player("base"))
        for field_name, value in (("win_rate", 0.8), ("kda", 5.0),
                                  ("accuracy", 0.8), ("headshot_rate", 0.5)):
            player = make_player(field_name, **{field_name: value})
            assert self.system.calculate_skill(player) > base
        strong = make_player("strong", win_rate=0.7, kda=4.0,
                             accuracy=0.7, headshot_rate=0.4)
        weak = make_player("weak", win_rate=0.4, kda=1.0,
                           accuracy=0.3, headshot_rate=0.1)
        assert self.system.calculate_skill(strong) > \
            self.system.calculate_skill(weak)

    # ---- 技能计算：最近表现 ----
    def test_case_02(self):
        """最近比赛按时间衰减加权，一个月前的表现几乎不影响技能。"""
        now = time.time()
        recent_wins = [{"timestamp": now, "won": True, "kda": 3.0}
                       for _ in range(5)]
        old_wins = [{"timestamp": now - 30 * 86400, "won": True, "kda": 3.0}
                    for _ in range(5)]
        recent_losses = [{"timestamp": now, "won": False, "kda": 1.0}
                         for _ in range(5)]
        s_recent = self.system.calculate_skill(
            make_player("recent", recent_matches=recent_wins))
        s_old = self.system.calculate_skill(
            make_player("old", recent_matches=old_wins))
        s_none = self.system.calculate_skill(make_player("none"))
        s_loss = self.system.calculate_skill(
            make_player("loss", recent_matches=recent_losses))
        assert s_recent > s_old
        assert s_old == pytest.approx(s_none, rel=0.02)
        assert s_loss < s_none

    # ---- 技能计算：位置差异 ----
    def test_case_03(self):
        """不同位置权重不同：C位更看KDA，辅助更看胜率。"""
        carry = self.system.calculate_skill(
            make_player("carry_kda", role="carry", kda=5.0))
        support = self.system.calculate_skill(
            make_player("support_kda", role="support", kda=5.0))
        assert carry > support
        support_wr = self.system.calculate_skill(
            make_player("support_wr", role="support", win_rate=0.8))
        carry_wr = self.system.calculate_skill(
            make_player("carry_wr", role="carry", win_rate=0.8))
        assert support_wr > carry_wr

    # ---- 技能计算：段位仍然有效 ----
    def test_case_04(self):
        """钻石技能远高于青铜，差距超出严格匹配的可接受范围。"""
        diamond = self.system.calculate_skill(make_player("d", rank="diamond"))
        bronze = self.system.calculate_skill(make_player("b", rank="bronze"))
        assert diamond > bronze
        assert diamond - bronze > \
            self.system.match_timeout(0)["max_skill_diff"]

    # ---- 匹配算法：团队平衡 ----
    def test_case_05(self):
        """6强4弱时，每队都是3强2弱而不是5强对5弱。"""
        roles = ["carry", "carry", "support", "support", "tank",
                 "tank", "mid", "mid", "jungle", "jungle"]
        players = [
            make_player(f"p{i}", rank="diamond", win_rate=0.65, kda=4.0,
                        role=roles[i]) if i < 6
            else make_player(f"p{i}", rank="bronze", win_rate=0.4, kda=1.0,
                             role=roles[i])
            for i in range(10)
        ]
        result = self.system.matchmaking(players)
        strong_a = sum(1 for p in result["team_a"] if p["rank"] == "diamond")
        strong_b = sum(1 for p in result["team_b"] if p["rank"] == "diamond")
        assert strong_a == 3 and strong_b == 3
        total = result["team_a_skill"] + result["team_b_skill"]
        assert result["skill_diff"] <= 0.1 * total

    # ---- 匹配算法：角色互补 ----
    def test_case_06(self):
        """每队同位置不超过2人，不会出现5个C位同队。"""
        roles = ["carry"] * 4 + ["support"] * 3 + ["tank"] * 3
        players = [
            make_player(f"p{i}", kda=1.0 + i * 0.3, role=roles[i])
            for i in range(10)
        ]
        result = self.system.matchmaking(players)
        for team in (result["team_a"], result["team_b"]):
            counts = Counter(p["role"] for p in team)
            assert max(counts.values()) <= 2

    # ---- 匹配算法：网络优化 ----
    def test_case_07(self):
        """人数充足时优先把延迟相近的玩家匹配到一起。"""
        players = [make_player(f"low{i}", ping=20 + i) for i in range(10)]
        players.append(make_player("high1", ping=300))
        players.append(make_player("high2", ping=320))
        result = self.system.matchmaking(players)
        selected = result["team_a"] + result["team_b"]
        assert all(p["ping"] < 100 for p in selected)
        assert result["latency_spread"] < 100

    # ---- 匹配算法：返回结构 ----
    def test_case_08(self):
        """匹配结果结构完整，两队各5人且无重叠。"""
        players = [make_player(f"p{i}", kda=1.0 + i * 0.2, ping=30 + i)
                   for i in range(10)]
        result = self.system.matchmaking(players)
        assert result is not None
        assert len(result["team_a"]) == 5
        assert len(result["team_b"]) == 5
        ids_a = {p["player_id"] for p in result["team_a"]}
        ids_b = {p["player_id"] for p in result["team_b"]}
        assert not ids_a & ids_b
        assert 0.0 <= result["quality"] <= 1.0
        assert self.system.matchmaking(players[:5]) is None

    # ---- 团队平衡：实际生效 ----
    def test_case_09(self):
        """team_balance 直接保证双方技能接近、强弱分布均匀。"""
        roles = ["carry", "carry", "support", "support", "tank",
                 "tank", "mid", "mid", "jungle", "jungle"]
        players = [
            make_player(f"p{i}", rank="diamond", win_rate=0.65, kda=4.0,
                        role=roles[i]) if i < 6
            else make_player(f"p{i}", rank="bronze", win_rate=0.4, kda=1.0,
                             role=roles[i])
            for i in range(10)
        ]
        result = self.system.team_balance(players)
        strong_a = sum(1 for p in result["team_a"] if p["rank"] == "diamond")
        strong_b = sum(1 for p in result["team_b"] if p["rank"] == "diamond")
        assert strong_a == 3 and strong_b == 3
        total = result["team_a_skill"] + result["team_b_skill"]
        assert result["skill_diff"] <= 0.1 * total

    # ---- 团队平衡：角色分配 ----
    def test_case_10(self):
        """team_balance 保证每队角色分配合理。"""
        roles = ["carry"] * 4 + ["support"] * 3 + ["tank"] * 3
        players = [
            make_player(f"p{i}", kda=1.0 + i * 0.3, role=roles[i])
            for i in range(10)
        ]
        result = self.system.team_balance(players)
        for team in (result["team_a"], result["team_b"]):
            counts = Counter(p["role"] for p in team)
            assert max(counts.values()) <= 2

    # ---- 团队平衡：替补机制 ----
    def test_case_11(self):
        """有人退出后，优先补同位置且技能最接近的替补。"""
        team = [
            make_player("leaving", role="carry", kda=3.0),
            make_player("c1", role="support", kda=2.5),
            make_player("c2", role="tank", kda=2.8),
            make_player("c3", role="mid", kda=2.0),
            make_player("c4", role="jungle", kda=2.2),
        ]
        candidates = [
            make_player("sub_close", role="carry", kda=3.1),
            make_player("sub_role", role="support", kda=3.0),
            make_player("sub_far", role="carry", kda=8.0),
        ]
        substitute = self.system.substitute_player(team, "leaving", candidates)
        assert substitute["player_id"] == "sub_close"
        assert len(team) == 5
        assert all(p["player_id"] != "leaving" for p in team)
        assert self.system.substitute_player(team, "missing", candidates) is None

    # ---- 行为分析：游戏风格 ----
    def test_case_12(self):
        """分析玩家游戏风格：激进型与辅助型可识别。"""
        aggressive = make_player("aggro", kda=5.0, accuracy=0.7,
                                 headshot_rate=0.5)
        assert self.system.behavior_analysis(aggressive)["style"] == \
            "aggressive"
        supportive = make_player("supp", role="support", assists=12.0)
        assert self.system.behavior_analysis(supportive)["style"] == \
            "supportive"

    # ---- 行为分析：异常检测 ----
    def test_case_13(self):
        """疑似开挂数据被标记，正常玩家不被标记。"""
        cheater = make_player("cheater", kda=12.0, accuracy=0.98,
                              headshot_rate=0.95)
        result = self.system.behavior_analysis(cheater)
        assert result["is_abnormal"] is True
        assert len(result["abnormal_reasons"]) >= 2
        normal = self.system.behavior_analysis(make_player("normal"))
        assert normal["is_abnormal"] is False
        assert normal["abnormal_reasons"] == []

    # ---- 行为分析：学习能力 ----
    def test_case_14(self):
        """观察越多置信度越高；比赛结果会修正技能评估。"""
        player = make_player("learner")
        first = self.system.behavior_analysis(player)
        last = first
        for _ in range(9):
            last = self.system.behavior_analysis(player)
        assert last["confidence"] > first["confidence"]
        assert last["observations"] == 10

        before = self.system.calculate_skill(player)
        for _ in range(5):
            self.system.record_match_result(["learner"], ["enemy"])
        after = self.system.calculate_skill(player)
        assert after > before

    # ---- 匹配超时：梯度 ----
    def test_case_15(self):
        """等待越久可接受技能差越大，且分多个档位。"""
        waits = (5, 45, 90, 200)
        info = [self.system.match_timeout(w) for w in waits]
        diffs = [item["max_skill_diff"] for item in info]
        assert diffs[0] < diffs[1] < diffs[2] < diffs[3]
        assert len({item["tier"] for item in info}) == 4

    # ---- 匹配超时：通知与上限 ----
    def test_case_16(self):
        """超时结果带等待通知；等再久也有上限，不会随便匹配。"""
        result = self.system.match_timeout(30)
        assert result["notification"]
        assert "30" in result["notification"]
        assert self.system.match_timeout(10000)["max_skill_diff"] <= 800.0

        players = [make_player(f"p{i}") for i in range(10)]
        match = self.system.matchmaking(players, wait_seconds=30)
        assert match["notification"]
        assert match["tier"] == "relaxed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
