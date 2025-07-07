#!/usr/bin/env python3
"""
エンコーダーのみ学習を実行するためのサンプルスクリプト

使用方法:
1. 基本的な実行:
   python encoder_only_train.py

2. カスタム設定での実行:
   python encoder_only_train.py training.learning_rate=5e-4 training.batch_size=64

3. 設定ファイルを指定:
   python train.py --config-name=encoder_only
"""

import hydra
import omegaconf
from train import run

@hydra.main(config_path="configs", version_base=None, config_name="encoder_only")
def run_encoder_only_training(cfg: omegaconf.DictConfig):
    """エンコーダーのみ学習を実行"""
    
    # エンコーダーのみ学習が有効になっているかチェック
    assert cfg.training.get("encoder_only", False), "encoder_only フラグが設定されていません"
    
    print("=" * 60)
    print("🚀 エンコーダーのみ学習を開始します")
    print("=" * 60)
    print(f"学習率: {cfg.training.learning_rate}")
    print(f"バッチサイズ: {cfg.training.batch_size}")
    print(f"事前分布KL係数: {cfg.training.prior_kl_coeff}")
    print(f"ペアワイズKL係数: {cfg.training.pairwise_kl_coeff}")
    print("=" * 60)
    
    # 通常の学習実行（内部でencoder_onlyフラグが処理される）
    run(cfg)
    
    print("=" * 60)
    print("✅ エンコーダーのみ学習が完了しました")
    print("=" * 60)

if __name__ == "__main__":
    run_encoder_only_training()
