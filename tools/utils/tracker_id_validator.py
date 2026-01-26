#!/usr/bin/env python3
"""
トラッカーID検証ユーティリティ

有効なトラッカーID形式の検証を行う
サポート形式: TEST-001, QUAL-042, INTG-003, OPTM-001等
"""

import re
from typing import Optional, Tuple


class TrackerIdValidationError(Exception):
    """トラッカーID検証エラー"""

    pass


class TrackerIdValidator:
    """トラッカーID検証クラス"""

    # 有効なトラッカーIDパターン
    VALID_PATTERNS = [
        r"^TEST-\d{3}$",  # TEST-001, TEST-042等
        r"^QUAL-\d{3}$",  # QUAL-001, QUAL-042等
        r"^INTG-\d{3}$",  # INTG-001, INTG-042等
        r"^OPTM-\d{3}$",  # OPTM-001, OPTM-042等
        r"^INCI-\d{3}$",  # INCI-001, INCI-042等
        r"^P1-[A-Z]\d{3}$",  # P1-A001, P1-B042等
        r"^PH\d+-\d{3}$",  # PH1-001, PH2-042等
        r"^QCC-\d{3}$",  # QCC-001, QCC-042等
    ]

    @staticmethod
    def validate_tracker_id(tracker_id: str) -> Tuple[bool, Optional[str]]:
        """
        トラッカーID検証

        Args:
            tracker_id: 検証対象のトラッカーID

        Returns:
            (検証結果, エラーメッセージ)

        Raises:
            TrackerIdValidationError: 無効なトラッカーID形式の場合
        """
        if not tracker_id:
            error_msg = TrackerIdValidator._generate_error_message(tracker_id, "空のトラッカーID")
            raise TrackerIdValidationError(error_msg)

        # パターンマッチング
        for pattern in TrackerIdValidator.VALID_PATTERNS:
            if re.match(pattern, tracker_id):
                return True, None

        # 無効な形式
        error_msg = TrackerIdValidator._generate_error_message(tracker_id, "無効な形式")
        raise TrackerIdValidationError(error_msg)

    @staticmethod
    def _generate_error_message(tracker_id: str, reason: str) -> str:
        """
        統一エラーメッセージ生成

        Args:
            tracker_id: 無効なトラッカーID
            reason: エラー理由

        Returns:
            統一フォーマットのエラーメッセージ
        """
        return f"""❌ エラー: 無効なトラッカーID形式です
   トラッカーID: {tracker_id}
   理由: {reason}

🔧 有効な形式:
   - TEST-XXX: TEST-001, TEST-042
   - QUAL-XXX: QUAL-001, QUAL-042  
   - INTG-XXX: INTG-001, INTG-042
   - OPTM-XXX: OPTM-001, OPTM-042
   - INCI-XXX: INCI-001, INCI-042
   - P1-XXXX: P1-A001, P1-B042
   - PHXX-XXX: PH1-001, PH2-042
   - QCC-XXX: QCC-001, QCC-042

⚠️ 注意: 無効なトラッカーIDでの処理実行は品質保証違反です"""

    @staticmethod
    def validate_and_exit_on_error(tracker_id: str) -> None:
        """
        トラッカーID検証実行（エラー時は即座終了）

        Args:
            tracker_id: 検証対象のトラッカーID
        """
        try:
            TrackerIdValidator.validate_tracker_id(tracker_id)
        except TrackerIdValidationError as e:
            print(str(e))
            import sys

            sys.exit(1)

    @staticmethod
    def extract_prefix(tracker_id: str) -> str:
        """
        トラッカーIDからプレフィックス抽出

        Args:
            tracker_id: トラッカーID

        Returns:
            プレフィックス（例：TEST-001 → TEST）
        """
        # 検証を先に実行
        TrackerIdValidator.validate_tracker_id(tracker_id)

        # プレフィックス抽出
        if "-" in tracker_id:
            return tracker_id.split("-")[0]
        else:
            # PH1-001のような形式の場合
            match = re.match(r"^(PH\d+)", tracker_id)
            if match:
                return match.group(1)

        return tracker_id


def main():
    """CLI実行用メイン関数"""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python tracker_id_validator.py <tracker_id>")
        sys.exit(1)

    tracker_id = sys.argv[1]

    try:
        is_valid, error_msg = TrackerIdValidator.validate_tracker_id(tracker_id)
        if is_valid:
            prefix = TrackerIdValidator.extract_prefix(tracker_id)
            print(f"✅ トラッカーID検証成功: {tracker_id}")
            print(f"📋 プレフィックス: {prefix}")
        else:
            print(error_msg)
            sys.exit(1)
    except TrackerIdValidationError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
