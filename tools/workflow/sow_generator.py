#!/usr/bin/env python3
"""
SOW Generator - Gate通過結果のSOW追記機能
KIRO-025: /pre-implementation-check 結果をSOWに自動追記する

このモジュールは、/pre-implementation-check の実行結果を受け取り、
SOWドキュメントに追記する機能を提供します。
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """個別Gateの結果"""

    gate_id: int
    gate_name: str
    passed: bool
    details: Optional[str] = None


@dataclass
class PreImplementationCheckResult:
    """Pre-Implementation Check全体の結果"""

    gates: List[GateResult]
    all_passed: bool
    execution_time: datetime
    notes: Optional[str] = None

    @property
    def passed_count(self) -> int:
        return sum(1 for g in self.gates if g.passed)

    @property
    def total_count(self) -> int:
        return len(self.gates)


class SOWGenerator:
    """SOWドキュメントへのGate結果追記を管理するクラス"""

    def __init__(self, workspace_base: str = None):
        """
        Args:
            workspace_base: ワークスペースのベースパス。
                           Noneの場合は WorkspaceConfig から取得。
        """
        if workspace_base is None:
            try:
                from config.workspace_config import WorkspaceConfig

                self.workspace_base = WorkspaceConfig.get_workspace_base()
            except ImportError:
                # フォールバック
                self.workspace_base = "/mnt/c/AItools/lora/train/yado/tracker-workspace"
        else:
            self.workspace_base = workspace_base

        logger.info(f"SOWGenerator initialized with workspace: {self.workspace_base}")

    def get_sow_path(self, tracker_id: str) -> str:
        """SOWドキュメントのパスを取得"""
        return os.path.join(self.workspace_base, tracker_id, "sow_document.md")

    def append_gate_results(
        self, tracker_id: str, result: PreImplementationCheckResult
    ) -> bool:
        """
        Gate通過結果をSOWドキュメントに追記する。

        Args:
            tracker_id: トラッカーID
            result: Pre-Implementation Check の結果

        Returns:
            追記成功時True、失敗時False
        """
        sow_path = self.get_sow_path(tracker_id)

        # SOWファイルの存在確認
        if not os.path.exists(sow_path):
            logger.error(f"SOW document not found: {sow_path}")
            return False

        # 追記内容を生成
        content = self._generate_gate_results_section(result)

        try:
            # 既存の内容を読み込み
            with open(sow_path, "r", encoding="utf-8") as f:
                existing_content = f.read()

            # 既に追記済みかチェック
            if "## Pre-Implementation Check Results" in existing_content:
                logger.warning(
                    f"Pre-Implementation Check Results already exists in {sow_path}"
                )
                # 既存のセクションを更新
                return self._update_existing_section(sow_path, existing_content, content)

            # 末尾に追記
            with open(sow_path, "a", encoding="utf-8") as f:
                f.write("\n\n" + content)

            logger.info(f"Gate results appended to SOW: {sow_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to append gate results to SOW: {e}")
            return False

    def _generate_gate_results_section(
        self, result: PreImplementationCheckResult
    ) -> str:
        """Gate結果セクションのMarkdownを生成"""
        lines = []
        lines.append("## Pre-Implementation Check Results")
        lines.append("")
        lines.append(f"**実行日時**: {result.execution_time.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append("### Gate通過状況")

        for gate in result.gates:
            checkbox = "[x]" if gate.passed else "[ ]"
            status = "✅" if gate.passed else "❌"
            lines.append(f"- {checkbox} Gate {gate.gate_id}: {gate.gate_name} {status}")
            if gate.details:
                lines.append(f"  - {gate.details}")

        lines.append("")
        lines.append("### 実装開始承認")

        if result.all_passed:
            lines.append("✅ 全Gate通過 - 実装開始可")
        else:
            lines.append(
                f"❌ 未通過Gateあり ({result.passed_count}/{result.total_count}) - 実装開始禁止"
            )
            if result.notes:
                lines.append(f"  理由: {result.notes}")

        return "\n".join(lines)

    def _update_existing_section(
        self, sow_path: str, existing_content: str, new_content: str
    ) -> bool:
        """既存のPre-Implementation Check Resultsセクションを更新"""
        try:
            # セクションの開始位置を見つける
            start_marker = "## Pre-Implementation Check Results"
            start_idx = existing_content.find(start_marker)

            if start_idx == -1:
                return False

            # 次のセクション（## で始まる行）を探す
            next_section_idx = existing_content.find("\n## ", start_idx + len(start_marker))

            if next_section_idx == -1:
                # 最後のセクションの場合
                updated_content = existing_content[:start_idx] + new_content
            else:
                # 途中のセクションの場合
                updated_content = (
                    existing_content[:start_idx]
                    + new_content
                    + "\n"
                    + existing_content[next_section_idx:]
                )

            with open(sow_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            logger.info(f"Updated existing Pre-Implementation Check Results in {sow_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to update existing section: {e}")
            return False

    def create_result_from_gates(
        self,
        gate_results: Dict[int, bool],
        notes: Optional[str] = None,
    ) -> PreImplementationCheckResult:
        """
        Gate結果の辞書からPreImplementationCheckResultを作成する。

        Args:
            gate_results: Gate番号とPass/Failの辞書 {1: True, 2: True, ...}
            notes: 追加メモ

        Returns:
            PreImplementationCheckResult インスタンス
        """
        gate_names = {
            1: "要件理解",
            2: "影響度調査",
            3: "アーキテクチャ整合性",
            4: "テスト先行",
            5: "既存コード理解",
        }

        gates = []
        for gate_id, passed in sorted(gate_results.items()):
            gates.append(
                GateResult(
                    gate_id=gate_id,
                    gate_name=gate_names.get(gate_id, f"Gate {gate_id}"),
                    passed=passed,
                )
            )

        all_passed = all(g.passed for g in gates)

        return PreImplementationCheckResult(
            gates=gates,
            all_passed=all_passed,
            execution_time=datetime.now(),
            notes=notes,
        )


# Singleton instance
_sow_generator_instance = None


def get_sow_generator() -> SOWGenerator:
    """Get the global SOW generator instance"""
    global _sow_generator_instance
    if _sow_generator_instance is None:
        _sow_generator_instance = SOWGenerator()
    return _sow_generator_instance


# CLI用のヘルパー関数
def append_check_results_to_sow(
    tracker_id: str,
    gate_results: Dict[int, bool],
    notes: Optional[str] = None,
) -> bool:
    """
    CLI等から呼び出すためのヘルパー関数。

    Args:
        tracker_id: トラッカーID
        gate_results: Gate結果の辞書 {1: True, 2: True, ...}
        notes: 追加メモ

    Returns:
        追記成功時True
    """
    generator = get_sow_generator()
    result = generator.create_result_from_gates(gate_results, notes)
    return generator.append_gate_results(tracker_id, result)


if __name__ == "__main__":
    # テスト用
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sow_generator.py <tracker_id>")
        print("Example: python sow_generator.py KIRO-025")
        sys.exit(1)

    tracker_id = sys.argv[1]

    # テスト用のGate結果
    test_results = {1: True, 2: True, 3: True, 4: True, 5: True}

    success = append_check_results_to_sow(tracker_id, test_results)
    if success:
        print(f"✅ Gate results appended to SOW for {tracker_id}")
    else:
        print(f"❌ Failed to append gate results for {tracker_id}")
