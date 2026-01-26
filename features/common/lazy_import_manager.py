"""
P1-023: VSCode安定性向上 - 遅延インポート最適化システム

GPT-4分析に基づく大型ライブラリの最適化:
- 必要時のみインポートによるメモリ節約
- 循環インポート問題の回避
- 起動時間の大幅短縮
- ML環境での安定性向上
"""

import importlib
import logging
import sys
import time
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class LazyImportManager:
    """
    遅延インポート管理クラス

    大型ML/AIライブラリの必要時インポートを管理し、
    メモリ使用量の最適化と循環インポート問題を回避する。
    """

    def __init__(self):
        self._import_cache: Dict[str, Any] = {}
        self._import_lock = Lock()
        self._import_stats = {
            "total_imports": 0,
            "cache_hits": 0,
            "import_times": {},
            "failed_imports": set(),
        }

        logger.info("LazyImportManager初期化完了")

    def register_lazy_import(
        self,
        module_name: str,
        alias: Optional[str] = None,
        required_attrs: Optional[List[str]] = None,
        fallback_func: Optional[Callable] = None,
    ) -> Callable:
        """
        遅延インポート登録

        Args:
            module_name: インポートするモジュール名
            alias: エイリアス名
            required_attrs: 必須属性リスト
            fallback_func: インポート失敗時のフォールバック関数

        Returns:
            遅延インポートされたモジュールを返す関数
        """
        cache_key = alias or module_name

        def lazy_loader():
            with self._import_lock:
                # キャッシュヒット確認
                if cache_key in self._import_cache:
                    self._import_stats["cache_hits"] += 1
                    logger.debug(f"遅延インポートキャッシュヒット: {module_name}")
                    return self._import_cache[cache_key]

                # インポート失敗履歴確認
                if module_name in self._import_stats["failed_imports"]:
                    logger.debug(f"インポート失敗履歴あり、フォールバック実行: {module_name}")
                    if fallback_func:
                        return fallback_func()
                    return None

                # 実際のインポート実行
                try:
                    start_time = time.time()
                    logger.debug(f"遅延インポート開始: {module_name}")

                    module = importlib.import_module(module_name)

                    # 必須属性確認
                    if required_attrs:
                        for attr in required_attrs:
                            if not hasattr(module, attr):
                                raise ImportError(f"必須属性不在: {module_name}.{attr}")

                    # キャッシュに保存
                    self._import_cache[cache_key] = module

                    # 統計更新
                    import_time = time.time() - start_time
                    self._import_stats["total_imports"] += 1
                    self._import_stats["import_times"][module_name] = import_time

                    logger.info(f"遅延インポート成功: {module_name} ({import_time:.3f}s)")
                    return module

                except ImportError as e:
                    logger.warning(f"遅延インポート失敗: {module_name} - {e}")
                    self._import_stats["failed_imports"].add(module_name)

                    if fallback_func:
                        logger.debug(f"フォールバック関数実行: {module_name}")
                        result = fallback_func()
                        # フォールバック結果もキャッシュ
                        self._import_cache[cache_key] = result
                        return result

                    return None

        return lazy_loader

    def get_torch(self):
        """PyTorch遅延インポート"""
        if not hasattr(self, "_torch_loader"):

            def torch_fallback():
                logger.warning("PyTorch利用不可 - CPU-onlyモードで継続")
                return type(
                    "MockTorch",
                    (),
                    {
                        "cuda": type(
                            "MockCuda",
                            (),
                            {
                                "is_available": lambda: False,
                                "device_count": lambda: 0,
                                "empty_cache": lambda: None,
                            },
                        )()
                    },
                )()

            self._torch_loader = self.register_lazy_import(
                "torch", required_attrs=["cuda"], fallback_func=torch_fallback
            )

        return self._torch_loader()

    def get_segment_anything(self):
        """Segment Anything遅延インポート"""
        if not hasattr(self, "_sam_loader"):

            def sam_fallback():
                logger.warning("SAM利用不可 - モックオブジェクト返却")
                return type(
                    "MockSAM", (), {"sam_model_registry": {}, "SamPredictor": lambda model: None}
                )()

            self._sam_loader = self.register_lazy_import(
                "segment_anything",
                required_attrs=["sam_model_registry"],
                fallback_func=sam_fallback,
            )

        return self._sam_loader()

    def get_ultralytics(self):
        """Ultralytics YOLO遅延インポート"""
        if not hasattr(self, "_yolo_loader"):

            def yolo_fallback():
                logger.warning("YOLO利用不可 - モックオブジェクト返却")
                return type(
                    "MockYOLO",
                    (),
                    {
                        "YOLO": lambda model_path: type(
                            "MockYOLOModel", (), {"predict": lambda *args, **kwargs: []}
                        )()
                    },
                )()

            self._yolo_loader = self.register_lazy_import(
                "ultralytics", required_attrs=["YOLO"], fallback_func=yolo_fallback
            )

        return self._yolo_loader()

    def get_opencv(self):
        """OpenCV遅延インポート"""
        if not hasattr(self, "_cv2_loader"):

            def cv2_fallback():
                logger.warning("OpenCV利用不可 - 基本画像処理のみ")
                return type(
                    "MockCV2", (), {"imread": lambda path: None, "imwrite": lambda path, img: False}
                )()

            self._cv2_loader = self.register_lazy_import(
                "cv2", required_attrs=["imread", "imwrite"], fallback_func=cv2_fallback
            )

        return self._cv2_loader()

    def get_google_auth(self):
        """Google認証ライブラリ遅延インポート"""
        if not hasattr(self, "_google_auth_loader"):

            def google_auth_fallback():
                logger.warning("Google認証ライブラリ利用不可")
                return None

            self._google_auth_loader = self.register_lazy_import(
                "google.auth", fallback_func=google_auth_fallback
            )

        return self._google_auth_loader()

    def preload_essential_modules(self, modules: Optional[List[str]] = None):
        """重要モジュールの事前ロード"""
        if modules is None:
            modules = ["torch", "cv2"]  # 最小限の事前ロード

        logger.info(f"重要モジュール事前ロード開始: {modules}")

        for module_name in modules:
            try:
                if module_name == "torch":
                    self.get_torch()
                elif module_name == "cv2":
                    self.get_opencv()
                elif module_name == "segment_anything":
                    self.get_segment_anything()
                elif module_name == "ultralytics":
                    self.get_ultralytics()
                else:
                    # 汎用インポート
                    loader = self.register_lazy_import(module_name)
                    loader()

            except Exception as e:
                logger.warning(f"事前ロード失敗: {module_name} - {e}")

        logger.info("重要モジュール事前ロード完了")

    def clear_cache(self, module_name: Optional[str] = None):
        """インポートキャッシュクリア"""
        with self._import_lock:
            if module_name:
                if module_name in self._import_cache:
                    del self._import_cache[module_name]
                    logger.debug(f"キャッシュクリア: {module_name}")
            else:
                self._import_cache.clear()
                logger.info("全インポートキャッシュクリア")

    def get_import_statistics(self) -> Dict[str, Any]:
        """インポート統計情報取得"""
        return {
            "cached_modules": list(self._import_cache.keys()),
            "cache_size": len(self._import_cache),
            "statistics": self._import_stats.copy(),
            "memory_usage_estimate": sum(
                sys.getsizeof(module) for module in self._import_cache.values()
            ),
        }


# グローバルマネージャー
_global_lazy_manager: Optional[LazyImportManager] = None


def get_lazy_import_manager() -> LazyImportManager:
    """グローバル遅延インポートマネージャー取得"""
    global _global_lazy_manager
    if _global_lazy_manager is None:
        _global_lazy_manager = LazyImportManager()
    return _global_lazy_manager


# 簡易インターフェース関数群
def lazy_torch():
    """PyTorch遅延インポート（簡易）"""
    return get_lazy_import_manager().get_torch()


def lazy_sam():
    """SAM遅延インポート（簡易）"""
    return get_lazy_import_manager().get_segment_anything()


def lazy_yolo():
    """YOLO遅延インポート（簡易）"""
    return get_lazy_import_manager().get_ultralytics()


def lazy_cv2():
    """OpenCV遅延インポート（簡易）"""
    return get_lazy_import_manager().get_opencv()


def lazy_google_auth():
    """Google認証遅延インポート（簡易）"""
    return get_lazy_import_manager().get_google_auth()


# デコレータ: 遅延インポート対応関数
def with_lazy_imports(*module_names):
    """
    関数デコレータ: 必要なモジュールを遅延インポート

    Args:
        module_names: 必要なモジュール名リスト
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_lazy_import_manager()

            # 必要モジュールを事前ロード
            if module_names:
                logger.debug(f"{func.__name__} 用モジュール遅延ロード: {module_names}")
                for module_name in module_names:
                    if module_name == "torch":
                        manager.get_torch()
                    elif module_name == "sam" or module_name == "segment_anything":
                        manager.get_segment_anything()
                    elif module_name == "yolo" or module_name == "ultralytics":
                        manager.get_ultralytics()
                    elif module_name == "cv2":
                        manager.get_opencv()
                    elif module_name == "google_auth":
                        manager.get_google_auth()

            return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def lazy_import_context(preload_modules: Optional[List[str]] = None):
    """
    遅延インポートコンテキストマネージャー

    Args:
        preload_modules: 事前ロードするモジュールリスト
    """
    manager = get_lazy_import_manager()

    try:
        if preload_modules:
            manager.preload_essential_modules(preload_modules)

        logger.debug("遅延インポートコンテキスト開始")
        yield manager

    except Exception as e:
        logger.error(f"遅延インポートコンテキストエラー: {e}")
        raise

    finally:
        logger.debug("遅延インポートコンテキスト終了")


# モデル取得関数（P1-B003でのsympy問題回避）
@with_lazy_imports("torch", "sam")
def get_sam_model_lazy(model_type: str = "vit_h", checkpoint_path: Optional[str] = None):
    """SAMモデル遅延取得（循環インポート回避）"""
    try:
        sam_module = lazy_sam()
        torch_module = lazy_torch()

        if model_type not in sam_module.sam_model_registry:
            logger.warning(f"不明なSAMモデルタイプ: {model_type}")
            return None

        model = sam_module.sam_model_registry[model_type]()

        if checkpoint_path and torch_module.cuda.is_available():
            try:
                model.load_state_dict(torch_module.load(checkpoint_path))
                logger.info(f"SAMモデルロード成功: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"SAMモデルロード失敗: {e}")

        return model

    except Exception as e:
        logger.error(f"SAMモデル取得失敗: {e}")
        return None


@with_lazy_imports("yolo")
def get_yolo_model_lazy(model_path: str = "yolov8x.pt"):
    """YOLOモデル遅延取得"""
    try:
        yolo_module = lazy_yolo()
        model = yolo_module.YOLO(model_path)
        logger.info(f"YOLOモデル取得成功: {model_path}")
        return model

    except Exception as e:
        logger.error(f"YOLOモデル取得失敗: {e}")
        return None


if __name__ == "__main__":
    # テスト実行
    import json

    def test_lazy_import_manager():
        """遅延インポートマネージャーのテスト"""
        print("=== P1-023 遅延インポート最適化システム テスト ===")
        print()

        manager = LazyImportManager()

        # テスト1: 基本的な遅延インポート
        print("テスト1: 基本遅延インポート")
        torch_module = manager.get_torch()
        print(f"PyTorch: {torch_module is not None}")
        print(f"CUDA利用可能: {torch_module.cuda.is_available() if torch_module else False}")
        print()

        # テスト2: キャッシュヒット確認
        print("テスト2: キャッシュヒット")
        torch_module2 = manager.get_torch()
        print(f"同一インスタンス: {torch_module is torch_module2}")
        print()

        # テスト3: フォールバック動作
        print("テスト3: 存在しないモジュール（フォールバック）")

        def mock_fallback():
            return "フォールバック結果"

        loader = manager.register_lazy_import("nonexistent_module", fallback_func=mock_fallback)
        result = loader()
        print(f"フォールバック結果: {result}")
        print()

        # テスト4: デコレータ使用
        print("テスト4: デコレータテスト")

        @with_lazy_imports("torch", "cv2")
        def test_function():
            torch_mod = lazy_torch()
            cv2_mod = lazy_cv2()
            return torch_mod is not None and cv2_mod is not None

        decorator_result = test_function()
        print(f"デコレータ結果: {decorator_result}")
        print()

        # テスト5: 統計情報
        print("テスト5: インポート統計")
        stats = manager.get_import_statistics()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        print()

        # テスト6: コンテキストマネージャー
        print("テスト6: コンテキストマネージャー")
        with lazy_import_context(["torch"]) as ctx:
            torch_in_context = ctx.get_torch()
            print(f"コンテキスト内PyTorch: {torch_in_context is not None}")

    # テスト実行
    logging.basicConfig(level=logging.INFO)
    test_lazy_import_manager()
