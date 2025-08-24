#!/usr/bin/env python3
"""
失敗パターン分析システム
QCC-011: 失敗画像の共通パターンを自動検出・分類

主要機能:
1. DBSCAN クラスタリングによる失敗パターン分類
2. t-SNE による高次元特徴の可視化
3. Isolation Forest による異常検出
4. 失敗原因の自動分類とレポート生成
"""

import numpy as np
import cv2

import json
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
# scikit-learn components
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FailurePatternAnalyzer:
    """失敗パターン分析クラス"""
    
    def __init__(self, workspace_dir: str = None):
        """
        初期化
        
        Args:
            workspace_dir: ワークスペースディレクトリ
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.scaler = StandardScaler()
        self.failure_patterns = {}
        self.analysis_results = {}
        
    def extract_image_features(self, image_path: Path) -> np.ndarray:
        """
        画像から特徴量を抽出
        
        Args:
            image_path: 画像パス
            
        Returns:
            特徴量ベクトル
        """
        try:
            # 画像読み込み
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"画像読み込み失敗: {image_path}")
                return np.zeros(20)  # デフォルト特徴量
                
            # BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 基本統計量
            height, width = img.shape[:2]
            aspect_ratio = width / height if height > 0 else 0
            
            # 色特徴
            mean_color = np.mean(img_rgb, axis=(0, 1))
            std_color = np.std(img_rgb, axis=(0, 1))
            
            # 明度特徴
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness_mean = np.mean(gray)
            brightness_std = np.std(gray)
            
            # エッジ特徴
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            # コントラスト
            contrast = gray.max() - gray.min()
            
            # ヒストグラム特徴
            hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
            hist_features = hist.flatten() / (height * width)
            
            # 特徴量ベクトル構築
            features = np.concatenate([
                [height, width, aspect_ratio],
                mean_color,  # 3要素
                std_color,   # 3要素
                [brightness_mean, brightness_std],
                [edge_density, contrast],
                hist_features  # 8要素
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"特徴抽出エラー: {e}")
            return np.zeros(20)
    
    def analyze_failure_patterns(self, 
                                failed_images_dir: Path,
                                success_images_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        失敗パターンを分析
        
        Args:
            failed_images_dir: 失敗画像ディレクトリ
            success_images_dir: 成功画像ディレクトリ（比較用）
            
        Returns:
            分析結果
        """
        logger.info("失敗パターン分析開始")
        
        # 失敗画像の特徴量抽出
        failed_features = []
        failed_paths = []
        
        for img_path in Path(failed_images_dir).glob("*.jpg"):
            features = self.extract_image_features(img_path)
            failed_features.append(features)
            failed_paths.append(img_path)
            
        if not failed_features:
            logger.warning("失敗画像が見つかりません")
            return {}
            
        failed_features = np.array(failed_features)
        
        # 成功画像との比較（オプション）
        success_features = None
        if success_images_dir and Path(success_images_dir).exists():
            success_features = []
            for img_path in Path(success_images_dir).glob("*.jpg"):
                features = self.extract_image_features(img_path)
                success_features.append(features)
            success_features = np.array(success_features) if success_features else None
        
        # 特徴量正規化
        failed_features_scaled = self.scaler.fit_transform(failed_features)
        
        # 1. DBSCANクラスタリング
        clustering_results = self._perform_clustering(failed_features_scaled, failed_paths)
        
        # 2. t-SNE可視化
        visualization_results = self._perform_tsne(failed_features_scaled)
        
        # 3. Isolation Forest異常検出
        anomaly_results = self._detect_anomalies(failed_features_scaled, failed_paths)
        
        # 4. パターン分類
        pattern_classification = self._classify_patterns(
            failed_features, failed_paths, clustering_results
        )
        
        # 結果統合
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "total_failed_images": len(failed_paths),
            "clustering": clustering_results,
            "visualization": visualization_results,
            "anomalies": anomaly_results,
            "pattern_classification": pattern_classification,
            "feature_statistics": self._calculate_feature_statistics(failed_features)
        }
        
        # 成功画像との比較分析
        if success_features is not None:
            self.analysis_results["comparison"] = self._compare_with_success(
                failed_features, success_features
            )
        
        return self.analysis_results
    
    def _perform_clustering(self, features: np.ndarray, paths: List[Path]) -> Dict[str, Any]:
        """DBSCANクラスタリング実行"""
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        clusters = dbscan.fit_predict(features)
        
        # クラスタ統計
        unique_clusters = np.unique(clusters)
        cluster_info = {}
        
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_paths = [str(p) for i, p in enumerate(paths) if mask[i]]
            
            cluster_name = "noise" if cluster_id == -1 else f"pattern_{cluster_id}"
            cluster_info[cluster_name] = {
                "size": len(cluster_paths),
                "percentage": len(cluster_paths) / len(paths) * 100,
                "sample_images": cluster_paths[:5]  # 最初の5枚
            }
        
        return {
            "n_clusters": int(len(unique_clusters) - (1 if -1 in unique_clusters else 0)),
            "n_noise": int(np.sum(clusters == -1)),
            "clusters": cluster_info,
            "labels": [int(label) for label in clusters.tolist()]  # int64 → int変換
        }
    
    def _perform_tsne(self, features: np.ndarray) -> Dict[str, Any]:
        """t-SNE次元削減と可視化"""
        try:
            # データが少ない場合やWSL環境でのセグフォルトを回避
            if len(features) < 5:
                logger.warning("データサイズが小さすぎます。PCAで代替します。")
                return self._perform_pca_fallback(features)
            
            # 高次元の場合はPCAで前処理
            if features.shape[1] > 50:
                pca = PCA(n_components=50)
                features = pca.fit_transform(features)
            
            # t-SNE実行（安全なパラメータ）
            perplexity = min(5, max(1, len(features) // 3))  # より安全な perplexity
            tsne = TSNE(
                n_components=2, 
                random_state=42, 
                perplexity=perplexity,
                max_iter=250,  # 修正: n_iter → max_iter
                method='exact' if len(features) < 30 else 'barnes_hut'
            )
            embeddings = tsne.fit_transform(features)
            
            return {
                "embeddings": embeddings.tolist(),
                "explained_variance": None  # t-SNEには分散説明率がない
            }
        except Exception as e:
            logger.warning(f"t-SNE実行エラー: {e}. PCAで代替します。")
            return self._perform_pca_fallback(features)
    
    def _perform_pca_fallback(self, features: np.ndarray) -> Dict[str, Any]:
        """t-SNE失敗時のPCA代替手段"""
        try:
            pca = PCA(n_components=min(2, features.shape[1], len(features)-1))
            embeddings = pca.fit_transform(features)
            
            return {
                "embeddings": embeddings.tolist(),
                "explained_variance": pca.explained_variance_ratio_.tolist()
            }
        except Exception as e:
            logger.error(f"PCA代替も失敗: {e}")
            return {
                "embeddings": [[0, 0] for _ in range(len(features))],
                "explained_variance": None
            }
    
    def _detect_anomalies(self, features: np.ndarray, paths: List[Path]) -> Dict[str, Any]:
        """Isolation Forestによる異常検出"""
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(features)
        anomaly_scores = iso_forest.score_samples(features)
        
        # 異常画像の特定
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        anomaly_paths = [str(paths[i]) for i in anomaly_indices]
        
        return {
            "n_anomalies": int(len(anomaly_indices)),
            "anomaly_rate": float(len(anomaly_indices) / len(paths) * 100),
            "anomaly_images": anomaly_paths,
            "anomaly_scores": [float(score) for score in anomaly_scores.tolist()],
            "threshold": float(iso_forest.score_samples(features).mean())
        }
    
    def _classify_patterns(self, features: np.ndarray, paths: List[Path], 
                          clustering_results: Dict) -> Dict[str, Any]:
        """失敗パターンの分類と命名"""
        patterns = {}
        cluster_labels = clustering_results["labels"]
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:
                continue
                
            cluster_mask = np.array(cluster_labels) == cluster_id
            cluster_features = features[cluster_mask]
            
            # パターン特性分析
            pattern_name = self._determine_pattern_type(cluster_features)
            
            patterns[f"cluster_{cluster_id}"] = {
                "pattern_type": pattern_name,
                "size": int(np.sum(cluster_mask)),
                "characteristics": self._extract_pattern_characteristics(cluster_features)
            }
        
        return patterns
    
    def _determine_pattern_type(self, features: np.ndarray) -> str:
        """パターンタイプを決定"""
        mean_features = np.mean(features, axis=0)
        
        # 特徴量インデックス
        brightness_idx = 9
        edge_density_idx = 11
        aspect_ratio_idx = 2
        
        # パターン判定ロジック
        if mean_features[brightness_idx] < 50:
            return "dark_image"
        elif mean_features[brightness_idx] > 200:
            return "overexposed"
        elif mean_features[edge_density_idx] < 0.05:
            return "low_detail"
        elif mean_features[edge_density_idx] > 0.3:
            return "high_complexity"
        elif mean_features[aspect_ratio_idx] > 2.0:
            return "extreme_aspect_ratio"
        else:
            return "general_failure"
    
    def _extract_pattern_characteristics(self, features: np.ndarray) -> Dict[str, float]:
        """パターンの特性を抽出"""
        return {
            "mean_brightness": float(np.mean(features[:, 9])),
            "mean_edge_density": float(np.mean(features[:, 11])),
            "mean_aspect_ratio": float(np.mean(features[:, 2])),
            "brightness_variance": float(np.var(features[:, 9])),
            "size_variance": float(np.var(features[:, 0] * features[:, 1]))
        }
    
    def _calculate_feature_statistics(self, features: np.ndarray) -> Dict[str, Any]:
        """特徴量の統計情報を計算"""
        return {
            "mean": np.mean(features, axis=0).tolist(),
            "std": np.std(features, axis=0).tolist(),
            "min": np.min(features, axis=0).tolist(),
            "max": np.max(features, axis=0).tolist(),
            "median": np.median(features, axis=0).tolist()
        }
    
    def _compare_with_success(self, failed_features: np.ndarray, 
                            success_features: np.ndarray) -> Dict[str, Any]:
        """成功画像との比較分析"""
        # 統計的差異
        failed_mean = np.mean(failed_features, axis=0)
        success_mean = np.mean(success_features, axis=0)
        
        difference = failed_mean - success_mean
        
        # 最も差が大きい特徴量
        top_differences_idx = np.argsort(np.abs(difference))[-5:]
        
        feature_names = [
            "height", "width", "aspect_ratio",
            "mean_r", "mean_g", "mean_b",
            "std_r", "std_g", "std_b",
            "brightness_mean", "brightness_std",
            "edge_density", "contrast"
        ] + [f"hist_{i}" for i in range(8)]
        
        top_differences = {
            feature_names[idx] if idx < len(feature_names) else f"feature_{idx}": 
            float(difference[idx])
            for idx in top_differences_idx
        }
        
        return {
            "top_differences": top_differences,
            "failed_characteristics": {
                "mean_brightness": float(failed_mean[9]),
                "mean_edge_density": float(failed_mean[11]),
                "mean_aspect_ratio": float(failed_mean[2])
            },
            "success_characteristics": {
                "mean_brightness": float(success_mean[9]),
                "mean_edge_density": float(success_mean[11]),
                "mean_aspect_ratio": float(success_mean[2])
            }
        }
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """分析レポートを生成"""
        if not self.analysis_results:
            return "分析結果がありません"
        
        report = []
        report.append("=" * 80)
        report.append("失敗パターン分析レポート")
        report.append("=" * 80)
        report.append(f"生成日時: {self.analysis_results['timestamp']}")
        report.append(f"分析対象: {self.analysis_results['total_failed_images']}枚")
        report.append("")
        
        # クラスタリング結果
        clustering = self.analysis_results["clustering"]
        report.append("【クラスタリング結果】")
        report.append(f"検出パターン数: {clustering['n_clusters']}")
        report.append(f"ノイズ画像数: {clustering['n_noise']}")
        report.append("")
        
        for cluster_name, info in clustering["clusters"].items():
            report.append(f"  {cluster_name}:")
            report.append(f"    - 画像数: {info['size']} ({info['percentage']:.1f}%)")
            report.append("")
        
        # 異常検出結果
        anomalies = self.analysis_results["anomalies"]
        report.append("【異常検出結果】")
        report.append(f"異常画像数: {anomalies['n_anomalies']} ({anomalies['anomaly_rate']:.1f}%)")
        report.append("")
        
        # パターン分類
        patterns = self.analysis_results["pattern_classification"]
        report.append("【パターン分類】")
        for cluster_id, pattern_info in patterns.items():
            report.append(f"  {cluster_id}:")
            report.append(f"    タイプ: {pattern_info['pattern_type']}")
            report.append(f"    サイズ: {pattern_info['size']}枚")
            chars = pattern_info['characteristics']
            report.append(f"    平均明度: {chars['mean_brightness']:.1f}")
            report.append(f"    エッジ密度: {chars['mean_edge_density']:.3f}")
            report.append("")
        
        # 成功画像との比較（存在する場合）
        if "comparison" in self.analysis_results:
            comp = self.analysis_results["comparison"]
            report.append("【成功画像との比較】")
            report.append("主要な差異:")
            for feature, diff in comp["top_differences"].items():
                report.append(f"  - {feature}: {diff:+.3f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        # ファイル出力
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text, encoding="utf-8")
            
            # JSON形式でも保存
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        return report_text


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="失敗パターン分析")
    parser.add_argument("--failed-dir", required=True, help="失敗画像ディレクトリ")
    parser.add_argument("--success-dir", help="成功画像ディレクトリ（比較用）")
    parser.add_argument("--output", help="レポート出力パス")
    
    args = parser.parse_args()
    
    # 分析実行
    analyzer = FailurePatternAnalyzer()
    results = analyzer.analyze_failure_patterns(
        Path(args.failed_dir),
        Path(args.success_dir) if args.success_dir else None
    )
    
    # レポート生成
    report = analyzer.generate_report(Path(args.output) if args.output else None)
    print(report)
    
    return 0


if __name__ == "__main__":
    exit(main())