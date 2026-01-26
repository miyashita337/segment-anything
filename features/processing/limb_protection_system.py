#!/usr/bin/env python3
"""
Limb Protection System - 手足保護システム
人体骨格推定と部分欠損の自動検出・補完による手足切断問題の解決
"""

import numpy as np
import cv2

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LimbProtectionSystem:
    """手足保護システム"""

    def __init__(
        self,
        enable_pose_estimation: bool = True,
        enable_limb_completion: bool = True,
        protection_margin: int = 20,
    ):
        """
        Args:
            enable_pose_estimation: 骨格推定の有効化
            enable_limb_completion: 手足補完の有効化
            protection_margin: 保護マージン（ピクセル）
        """
        self.enable_pose_estimation = enable_pose_estimation
        self.enable_limb_completion = enable_limb_completion
        self.protection_margin = protection_margin

        # 人体の主要関節点（OpenPose形式）
        self.pose_keypoints = {
            "nose": 0,
            "neck": 1,
            "r_shoulder": 2,
            "r_elbow": 3,
            "r_wrist": 4,
            "l_shoulder": 5,
            "l_elbow": 6,
            "l_wrist": 7,
            "r_hip": 8,
            "r_knee": 9,
            "r_ankle": 10,
            "l_hip": 11,
            "l_knee": 12,
            "l_ankle": 13,
            "r_eye": 14,
            "l_eye": 15,
            "r_ear": 16,
            "l_ear": 17,
        }

        # 手足の接続関係
        self.limb_connections = [
            ("r_shoulder", "r_elbow"),
            ("r_elbow", "r_wrist"),  # 右腕
            ("l_shoulder", "l_elbow"),
            ("l_elbow", "l_wrist"),  # 左腕
            ("r_hip", "r_knee"),
            ("r_knee", "r_ankle"),  # 右脚
            ("l_hip", "l_knee"),
            ("l_knee", "l_ankle"),  # 左脚
        ]

        logger.info(
            f"LimbProtectionSystem初期化: pose={enable_pose_estimation}, "
            f"completion={enable_limb_completion}, margin={protection_margin}"
        )

    def protect_limbs_in_mask(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        マスクで手足保護処理を実行

        Args:
            image: 入力画像 (H, W, 3)
            mask: キャラクターマスク

        Returns:
            保護されたマスクと分析結果
        """
        logger.debug(f"手足保護処理開始: image={image.shape}, mask={mask.shape}")

        protection_result = {
            "pose_analysis": {},
            "limb_analysis": {},
            "protection_applied": False,
            "protection_quality": 0.0,
        }

        protected_mask = mask.copy()

        # 1. 人体骨格推定
        if self.enable_pose_estimation:
            pose_info = self._estimate_pose(image)
            protection_result["pose_analysis"] = pose_info

            if pose_info["keypoints_detected"] > 0:
                # 2. 手足欠損検出
                limb_analysis = self._analyze_limb_completeness(mask, pose_info)
                protection_result["limb_analysis"] = limb_analysis

                # 3. マスク拡張による手足保護
                if self.enable_limb_completion and limb_analysis["incomplete_limbs"]:
                    protected_mask = self._expand_mask_for_limbs(
                        image, mask, pose_info, limb_analysis
                    )
                    protection_result["protection_applied"] = True

        # 4. フォールバック: 骨格推定なしでの手足保護
        if not protection_result["protection_applied"]:
            fallback_protection = self._fallback_limb_protection(image, mask)
            if fallback_protection["applied"]:
                protected_mask = fallback_protection["mask"]
                protection_result["protection_applied"] = True
                protection_result["fallback_used"] = True

        # 5. 保護品質評価
        protection_result["protection_quality"] = self._evaluate_protection_quality(
            mask, protected_mask, protection_result
        )

        logger.debug("手足保護処理完了")
        return protected_mask, protection_result

    def _estimate_pose(self, image: np.ndarray) -> Dict[str, Any]:
        """簡易人体骨格推定（MediaPipeなしでの実装）"""
        # MediaPipeが利用できない環境での代替実装
        # 色と形状特徴に基づく簡易的な関節推定

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = image.shape[:2]

        # 1. 頭部推定（最上部の肌色領域）
        head_candidates = self._find_head_region(image)

        # 2. 胴体推定（頭部から下の中央領域）
        torso_region = self._find_torso_region(image, head_candidates)

        # 3. 手足末端推定（エッジが強い小領域）
        limb_extremities = self._find_limb_extremities(image)

        # 4. 簡易関節点推定
        estimated_keypoints = self._estimate_keypoints_simple(
            head_candidates, torso_region, limb_extremities, (width, height)
        )

        pose_info = {
            "keypoints_detected": len([kp for kp in estimated_keypoints if kp is not None]),
            "keypoints": estimated_keypoints,
            "head_region": head_candidates,
            "torso_region": torso_region,
            "limb_extremities": limb_extremities,
            "estimation_confidence": self._calculate_pose_confidence(estimated_keypoints),
        }

        logger.debug(f"骨格推定結果: {pose_info['keypoints_detected']}個の関節点")
        return pose_info

    def _find_head_region(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """頭部領域の検出"""
        # 肌色による頭部候補検出
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 肌色範囲
        skin_lower = np.array([0, 20, 70])
        skin_upper = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)

        # 上部1/3の領域で肌色領域を検索
        height = image.shape[0]
        upper_region = skin_mask[: height // 3, :]

        # 連結成分解析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(upper_region)

        head_candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 500:  # 十分な大きさの領域
                center_x, center_y = centroids[i]
                head_candidates.append((int(center_x), int(center_y)))

        return head_candidates

    def _find_torso_region(
        self, image: np.ndarray, head_candidates: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """胴体領域の推定"""
        height, width = image.shape[:2]

        if not head_candidates:
            # 頭部が見つからない場合は中央上部を仮定
            center_y = height // 4
            center_x = width // 2
        else:
            # 最も上部の頭部候補を使用
            center_x, center_y = min(head_candidates, key=lambda p: p[1])

        # 胴体領域（頭部の下、画像の中央部）
        torso_top = max(0, center_y + 50)
        torso_bottom = min(height, center_y + height // 2)
        torso_left = max(0, center_x - width // 6)
        torso_right = min(width, center_x + width // 6)

        return {
            "bbox": (torso_left, torso_top, torso_right - torso_left, torso_bottom - torso_top),
            "center": (center_x, (torso_top + torso_bottom) // 2),
        }

    def _find_limb_extremities(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """手足末端の検出"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)

        # 形態学的処理で手足のような細い構造を強調
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # 連結成分解析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_dilated)

        extremities = []
        height, width = image.shape[:2]

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 50 < area < 2000:  # 手足末端のサイズ範囲
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                center_x, center_y = centroids[i]

                # 画像端付近の領域を手足末端候補とする
                edge_distance = min(x, y, width - (x + w), height - (y + h))

                extremities.append(
                    {
                        "center": (int(center_x), int(center_y)),
                        "bbox": (x, y, w, h),
                        "area": area,
                        "edge_distance": edge_distance,
                        "aspect_ratio": w / h if h > 0 else 1.0,
                    }
                )

        # エッジからの距離でソート（端に近いほど手足末端の可能性が高い）
        extremities.sort(key=lambda x: x["edge_distance"])

        return extremities[:8]  # 上位8個を返す

    def _estimate_keypoints_simple(
        self,
        head_candidates: List[Tuple[int, int]],
        torso_region: Dict[str, Any],
        limb_extremities: List[Dict[str, Any]],
        image_size: Tuple[int, int],
    ) -> List[Optional[Tuple[int, int]]]:
        """簡易関節点推定"""
        width, height = image_size
        keypoints = [None] * 18  # OpenPoseの関節点数

        # 頭部・首の推定
        if head_candidates:
            head_x, head_y = head_candidates[0]
            keypoints[self.pose_keypoints["nose"]] = (head_x, head_y)
            keypoints[self.pose_keypoints["neck"]] = (head_x, head_y + 30)

        # 胴体中心の推定
        torso_center = torso_region["center"]
        torso_x, torso_y = torso_center

        # 肩の推定
        shoulder_y = torso_y - 20
        keypoints[self.pose_keypoints["r_shoulder"]] = (torso_x + 40, shoulder_y)
        keypoints[self.pose_keypoints["l_shoulder"]] = (torso_x - 40, shoulder_y)

        # 腰の推定
        hip_y = torso_y + 50
        keypoints[self.pose_keypoints["r_hip"]] = (torso_x + 20, hip_y)
        keypoints[self.pose_keypoints["l_hip"]] = (torso_x - 20, hip_y)

        # 手足末端から関節の推定
        for extremity in limb_extremities[:4]:  # 上位4個を使用
            ex_x, ex_y = extremity["center"]

            # 位置に基づいて手首・足首を推定
            if ex_y < torso_y:  # 上半身
                if ex_x > torso_x:  # 右側
                    if keypoints[self.pose_keypoints["r_wrist"]] is None:
                        keypoints[self.pose_keypoints["r_wrist"]] = (ex_x, ex_y)
                        # 肘の推定
                        if keypoints[self.pose_keypoints["r_shoulder"]]:
                            shoulder_x, shoulder_y = keypoints[self.pose_keypoints["r_shoulder"]]
                            elbow_x = (shoulder_x + ex_x) // 2
                            elbow_y = (shoulder_y + ex_y) // 2
                            keypoints[self.pose_keypoints["r_elbow"]] = (elbow_x, elbow_y)
                else:  # 左側
                    if keypoints[self.pose_keypoints["l_wrist"]] is None:
                        keypoints[self.pose_keypoints["l_wrist"]] = (ex_x, ex_y)
                        # 肘の推定
                        if keypoints[self.pose_keypoints["l_shoulder"]]:
                            shoulder_x, shoulder_y = keypoints[self.pose_keypoints["l_shoulder"]]
                            elbow_x = (shoulder_x + ex_x) // 2
                            elbow_y = (shoulder_y + ex_y) // 2
                            keypoints[self.pose_keypoints["l_elbow"]] = (elbow_x, elbow_y)
            else:  # 下半身
                if ex_x > torso_x:  # 右側
                    if keypoints[self.pose_keypoints["r_ankle"]] is None:
                        keypoints[self.pose_keypoints["r_ankle"]] = (ex_x, ex_y)
                        # 膝の推定
                        if keypoints[self.pose_keypoints["r_hip"]]:
                            hip_x, hip_y = keypoints[self.pose_keypoints["r_hip"]]
                            knee_x = (hip_x + ex_x) // 2
                            knee_y = (hip_y + ex_y) // 2
                            keypoints[self.pose_keypoints["r_knee"]] = (knee_x, knee_y)
                else:  # 左側
                    if keypoints[self.pose_keypoints["l_ankle"]] is None:
                        keypoints[self.pose_keypoints["l_ankle"]] = (ex_x, ex_y)
                        # 膝の推定
                        if keypoints[self.pose_keypoints["l_hip"]]:
                            hip_x, hip_y = keypoints[self.pose_keypoints["l_hip"]]
                            knee_x = (hip_x + ex_x) // 2
                            knee_y = (hip_y + ex_y) // 2
                            keypoints[self.pose_keypoints["l_knee"]] = (knee_x, knee_y)

        return keypoints

    def _calculate_pose_confidence(self, keypoints: List[Optional[Tuple[int, int]]]) -> float:
        """骨格推定の信頼度計算"""
        detected_count = len([kp for kp in keypoints if kp is not None])
        total_count = len(keypoints)

        # 主要関節点の重み付き評価
        important_joints = ["neck", "r_shoulder", "l_shoulder", "r_hip", "l_hip"]
        important_detected = 0

        for joint_name in important_joints:
            joint_idx = self.pose_keypoints[joint_name]
            if joint_idx < len(keypoints) and keypoints[joint_idx] is not None:
                important_detected += 1

        # 基本信頼度 + 重要関節ボーナス
        base_confidence = detected_count / total_count
        important_bonus = important_detected / len(important_joints) * 0.3

        return min(1.0, base_confidence + important_bonus)

    def _analyze_limb_completeness(
        self, mask: np.ndarray, pose_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """手足の完全性分析"""
        keypoints = pose_info["keypoints"]

        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask.copy()

        incomplete_limbs = []
        limb_status = {}

        # 各手足の接続を確認
        for connection in self.limb_connections:
            joint1_name, joint2_name = connection
            joint1_idx = self.pose_keypoints[joint1_name]
            joint2_idx = self.pose_keypoints[joint2_name]

            limb_name = f"{joint1_name}_to_{joint2_name}"

            if (
                joint1_idx < len(keypoints)
                and joint2_idx < len(keypoints)
                and keypoints[joint1_idx] is not None
                and keypoints[joint2_idx] is not None
            ):
                # 関節間の線がマスクで覆われているかチェック
                pt1 = keypoints[joint1_idx]
                pt2 = keypoints[joint2_idx]

                coverage = self._check_limb_coverage(mask_gray, pt1, pt2)

                limb_status[limb_name] = {
                    "coverage": coverage,
                    "complete": coverage > 0.7,
                    "joint1": pt1,
                    "joint2": pt2,
                }

                if coverage < 0.7:
                    incomplete_limbs.append(
                        {
                            "name": limb_name,
                            "coverage": coverage,
                            "joint1": pt1,
                            "joint2": pt2,
                            "missing_area": 1.0 - coverage,
                        }
                    )

        return {
            "incomplete_limbs": incomplete_limbs,
            "limb_status": limb_status,
            "completion_rate": len([ls for ls in limb_status.values() if ls["complete"]])
            / max(len(limb_status), 1),
        }

    def _check_limb_coverage(
        self, mask: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int]
    ) -> float:
        """手足部分のマスク被覆率チェック"""
        # 2点間の線上でマスクの被覆率を計算
        x1, y1 = pt1
        x2, y2 = pt2

        # 線の長さ
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 1:
            return 1.0

        # 線上の点を採取
        num_samples = max(10, int(length))
        covered_count = 0

        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            # 境界チェック
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                if mask[y, x] > 0:
                    covered_count += 1

        return covered_count / num_samples

    def _expand_mask_for_limbs(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        pose_info: Dict[str, Any],
        limb_analysis: Dict[str, Any],
    ) -> np.ndarray:
        """手足保護のためのマスク拡張"""
        expanded_mask = mask.copy()

        if len(expanded_mask.shape) == 3:
            expanded_mask = cv2.cvtColor(expanded_mask, cv2.COLOR_RGB2GRAY)

        # 不完全な手足を拡張
        for incomplete_limb in limb_analysis["incomplete_limbs"]:
            pt1 = incomplete_limb["joint1"]
            pt2 = incomplete_limb["joint2"]

            # 関節間の線を太くしてマスクに追加
            expanded_mask = self._draw_limb_protection(expanded_mask, pt1, pt2)

        # 手足末端の保護拡張
        keypoints = pose_info["keypoints"]
        extremity_joints = ["r_wrist", "l_wrist", "r_ankle", "l_ankle"]

        for joint_name in extremity_joints:
            joint_idx = self.pose_keypoints[joint_name]
            if joint_idx < len(keypoints) and keypoints[joint_idx] is not None:
                x, y = keypoints[joint_idx]
                # 末端周辺を円形に拡張
                cv2.circle(expanded_mask, (x, y), self.protection_margin, 255, -1)

        return expanded_mask

    def _draw_limb_protection(
        self, mask: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int]
    ) -> np.ndarray:
        """手足保護線の描画"""
        # 太い線でマスクを拡張
        cv2.line(mask, pt1, pt2, 255, thickness=self.protection_margin)

        # 両端も円形に拡張
        cv2.circle(mask, pt1, self.protection_margin // 2, 255, -1)
        cv2.circle(mask, pt2, self.protection_margin // 2, 255, -1)

        return mask

    def _fallback_limb_protection(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """フォールバック手足保護（骨格推定なし）- 改善版"""
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask.copy()

        # マスクの境界近辺で細い部分を検出・拡張
        # 細い部分は手足の可能性が高い

        # 1. マスクの骨格化（cv2.ximgproc.thinning の代替実装）
        skeleton = self._zhang_suen_thinning(mask_gray)

        # 2. 末端点と分岐点の検出
        endpoint_coords, branch_coords = self._find_skeleton_features(skeleton)

        # 3. 距離変換による細い部分の検出
        dist_transform = cv2.distanceTransform(mask_gray, cv2.DIST_L2, 5)
        thin_regions = (dist_transform > 0) & (dist_transform < 8)  # 細い部分

        # 4. 拡張処理
        expanded_mask = mask_gray.copy()

        # 末端点周辺を拡張（手足の末端保護）
        for y, x in endpoint_coords:
            cv2.circle(expanded_mask, (x, y), self.protection_margin, 255, -1)

        # 分岐点周辺も軽く拡張（関節部分の保護）
        for y, x in branch_coords:
            cv2.circle(expanded_mask, (x, y), self.protection_margin // 2, 255, -1)

        # 細い部分の膨張
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thin_expanded = cv2.dilate(thin_regions.astype(np.uint8) * 255, kernel, iterations=1)

        # 5. 白色系部位の特別保護（kana08評価で問題となった部分）
        expanded_mask = self._protect_white_regions(image, expanded_mask, mask_gray)

        # 最終マスク統合
        final_mask = cv2.bitwise_or(expanded_mask, thin_expanded)

        # さらに手足らしい形状の保護
        final_mask = self._protect_limb_like_shapes(final_mask, image)

        # 変化があったかチェック
        applied = not np.array_equal(mask_gray, final_mask)

        return {
            "applied": applied,
            "mask": final_mask,
            "endpoints_found": len(endpoint_coords),
            "branches_found": len(branch_coords),
            "method": "enhanced_fallback",
        }

    def _zhang_suen_thinning(self, mask: np.ndarray) -> np.ndarray:
        """Zhang-Suen細線化アルゴリズム（ximgproc.thinningの代替）"""
        # 入力を0/1のバイナリに変換
        binary = (mask > 0).astype(np.uint8)

        def neighbors(x, y, image):
            """8近傍の取得"""
            return [
                image[x - 1, y],
                image[x - 1, y + 1],
                image[x, y + 1],
                image[x + 1, y + 1],
                image[x + 1, y],
                image[x + 1, y - 1],
                image[x, y - 1],
                image[x - 1, y - 1],
            ]

        def transitions(neighbors):
            """0→1の遷移数を計算"""
            n = neighbors + neighbors[0:1]  # 循環
            return sum(n[i] == 0 and n[i + 1] == 1 for i in range(8))

        changed = True
        height, width = binary.shape

        # 境界は処理しない
        for _ in range(50):  # 最大50回の反復
            if not changed:
                break
            changed = False

            # Phase 1
            to_remove = []
            for x in range(1, height - 1):
                for y in range(1, width - 1):
                    if binary[x, y] == 1:
                        n = neighbors(x, y, binary)
                        if (
                            2 <= sum(n) <= 6
                            and transitions(n) == 1
                            and n[0] * n[2] * n[4] == 0
                            and n[2] * n[4] * n[6] == 0
                        ):
                            to_remove.append((x, y))

            for x, y in to_remove:
                binary[x, y] = 0
                changed = True

            # Phase 2
            to_remove = []
            for x in range(1, height - 1):
                for y in range(1, width - 1):
                    if binary[x, y] == 1:
                        n = neighbors(x, y, binary)
                        if (
                            2 <= sum(n) <= 6
                            and transitions(n) == 1
                            and n[0] * n[2] * n[6] == 0
                            and n[0] * n[4] * n[6] == 0
                        ):
                            to_remove.append((x, y))

            for x, y in to_remove:
                binary[x, y] = 0
                changed = True

        return binary * 255

    def _find_skeleton_features(
        self, skeleton: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """骨格の特徴点（末端点・分岐点）を検出"""
        # 8近傍での接続数をカウント
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbors = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)

        # 末端点（接続が1個）
        endpoints = np.where((skeleton > 0) & (neighbors == 1))
        endpoint_coords = list(zip(endpoints[0], endpoints[1]))

        # 分岐点（接続が3個以上）
        branches = np.where((skeleton > 0) & (neighbors >= 3))
        branch_coords = list(zip(branches[0], branches[1]))

        return endpoint_coords, branch_coords

    def _protect_white_regions(
        self, image: np.ndarray, mask: np.ndarray, original_mask: np.ndarray
    ) -> np.ndarray:
        """白色系部位の特別保護（カチューシャ、胸部分、足など）"""
        # HSV色空間での白色系検出
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # 多段階白色検出
        white_regions = (
            ((s < 30) & (v > 180))
            | ((s < 40) & (v > 160))  # 純白
            | ((s < 20) & (v > 140))  # 薄白  # グレー系
        )

        # オリジナルマスクとの交差領域を重視
        original_white = cv2.bitwise_and(white_regions.astype(np.uint8) * 255, original_mask)

        # 白色領域の境界を検出
        white_edges = cv2.Canny(original_white, 30, 100)

        # 境界から少し拡張して保護
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        white_protected = cv2.dilate(white_edges, kernel, iterations=2)

        # 元のマスクと統合
        return cv2.bitwise_or(mask, white_protected)

    def _protect_limb_like_shapes(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """手足らしい形状の保護"""
        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        protected_mask = mask.copy()

        for contour in contours:
            # 輪郭の特徴分析
            area = cv2.contourArea(contour)
            if area < 100:  # 小さすぎる領域は無視
                continue

            # バウンディングボックス
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / max(w, 1)

            # 手足らしい形状（細長い）の保護強化
            if aspect_ratio > 2.0 or aspect_ratio < 0.5:  # 縦長または横長
                # 輪郭を少し拡張
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # 拡張した輪郭で塗りつぶし
                expanded_contour = []
                center_x, center_y = x + w // 2, y + h // 2

                for point in approx:
                    px, py = point[0]
                    # 中心から外側に向かって少し拡張
                    direction_x = px - center_x
                    direction_y = py - center_y
                    length = math.sqrt(direction_x**2 + direction_y**2)

                    if length > 0:
                        # 3ピクセル外側に拡張
                        expansion = 3.0
                        new_x = int(px + (direction_x / length) * expansion)
                        new_y = int(py + (direction_y / length) * expansion)
                        expanded_contour.append([new_x, new_y])

                if len(expanded_contour) >= 3:
                    expanded_contour = np.array(expanded_contour, dtype=np.int32)
                    cv2.fillPoly(protected_mask, [expanded_contour], 255)

        return protected_mask

    def _evaluate_protection_quality(
        self,
        original_mask: np.ndarray,
        protected_mask: np.ndarray,
        protection_result: Dict[str, Any],
    ) -> float:
        """保護品質の評価"""
        # 基本品質スコア
        base_score = 0.5

        # 骨格推定の信頼度ボーナス
        if "pose_analysis" in protection_result:
            pose_confidence = protection_result["pose_analysis"].get("estimation_confidence", 0)
            base_score += pose_confidence * 0.3

        # 手足完全性の改善度
        if "limb_analysis" in protection_result:
            completion_rate = protection_result["limb_analysis"].get("completion_rate", 0)
            base_score += completion_rate * 0.2

        # マスクの変化量（適度な拡張が良い）
        if len(original_mask.shape) == 3:
            orig_area = np.sum(cv2.cvtColor(original_mask, cv2.COLOR_RGB2GRAY) > 0)
        else:
            orig_area = np.sum(original_mask > 0)

        if len(protected_mask.shape) == 3:
            prot_area = np.sum(cv2.cvtColor(protected_mask, cv2.COLOR_RGB2GRAY) > 0)
        else:
            prot_area = np.sum(protected_mask > 0)

        if orig_area > 0:
            expansion_ratio = prot_area / orig_area
            # 適度な拡張（1.05～1.3倍）が理想
            if 1.05 <= expansion_ratio <= 1.3:
                base_score += 0.2
            elif expansion_ratio > 1.0:
                base_score += 0.1

        return min(1.0, base_score)


def test_limb_protection_system():
    """手足保護システムのテスト"""
    protector = LimbProtectionSystem(
        enable_pose_estimation=True, enable_limb_completion=True, protection_margin=15
    )

    # テスト画像とマスク
    test_image_path = Path("/mnt/c/AItools/lora/train/yado/org/kana08/kana08_0009.jpg")
    test_mask_path = Path(
        "/mnt/c/AItools/lora/train/yado/clipped_boundingbox/kana08_robust_system_final/kana08_0009.jpg"
    )

    if test_image_path.exists():
        image = cv2.imread(str(test_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # マスクを生成（実際にはSAMの結果を使用）
        # テスト用に簡易マスクを作成
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        print(f"テスト画像読み込み: {image.shape}")
        print(f"テストマスク生成: {mask.shape}")

        # 手足保護実行
        protected_mask, analysis = protector.protect_limbs_in_mask(image, mask)

        # 分析結果表示
        print("\\n🦴 手足保護分析結果:")
        print(f"骨格分析: {analysis['pose_analysis']}")
        print(f"手足分析: {analysis['limb_analysis']}")
        print(f"保護適用: {analysis['protection_applied']}")
        print(f"保護品質: {analysis['protection_quality']:.3f}")

        # 結果保存
        output_path = Path("/tmp/limb_protection_test.jpg")
        cv2.imwrite(str(output_path), protected_mask)
        print(f"\\n💾 結果保存: {output_path}")
    else:
        print(f"テスト画像が見つかりません: {test_image_path}")


if __name__ == "__main__":
    test_limb_protection_system()
