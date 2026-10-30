# PAI最終課題_46069：言語指示によるロボット操作と失敗原因の診断

既存の公開シミュレータ実験のログから、目標達成と対象外物体の変位を分けて集計する。
ログ解析部分は外部PythonパッケージやGPU不要。Python 3.10以降を想定する。
ロボット推論・録画には別途GPUと外部依存が必要。

## ファイルと検証状況

|ファイル|役割|確認済み範囲|
|---|---|---|
|`analyze_rollouts.py`|結果・接触ログの集計|既存8試行を処理済み|
|`record_episode.py`|シミュレータ実行・録画・ログ記録|構文・CLIのみ。GPU実行は未確認|
|`resvla_policy.py`|既存ResVLAの推論アダプタ|構文のみ。切り出し後のGPU実行は未確認|

提出用コードの作業版。GitHub公開済みでも、全体再現検証・課題提出完了を意味しない。
動画・PDF・提出URLの登録は別途行う。

```sh
python analyze_rollouts.py --source /path/to/diagnostic_results --output analysis.json
```

入力は `タスク名/試行名/result.json` と同じ場所にある `trace.json`。
解析は8試行で動作確認済み。`record_episode.py` と `resvla_policy.py` は独立実行向けに
切り出したコードで、構文・CLI検査のみ完了。

1mmは今回の実験で使用した変位閾値であり、一般的な安全規格ではない。
ゼロ行動との比較も、接触が一切ないことの証明にはならない。
選んだ実演動画の成功数をモデル全体の成功率として扱わない。

第三者のモデル・シミュレータ・配布環境は同梱していない。

## 外部依存と取得元

- ResVLA: https://github.com/4DVLab/ResVLA
- 公開LIBERO重み: https://huggingface.co/GaussionZhong/resvla_libero_all_2B
- 基盤VLM: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- 元実験の派生環境LIBERO-plus: https://github.com/sylvestf/LIBERO-plus

これらは元コード・READMEで参照された取得元。各提供元の現在の手順・利用条件を確認して
別途インストールすること。第三者コードや重みをこの提出用フォルダから再配布しない。
LIBERO-plusと配布アセットの再公開条件は未確認。標準LIBEROへの変更で映像・結果が
同一になるとは限らない。セットアップの完全な再現検証は未完了。

ResVLAのPython環境ではtorch、transformers、numpy等が必要。
録画側はLIBERO、MuJoCo、robosuite、torch、numpy、imageio、imageio-ffmpegを使用する。
既存環境が複数あるため、一律の未検証バージョンをrequirementsに固定していない。
モデル設定・正規化統計はチェックポイントと一緒に保持する。

## 実行例（設定例であり、コピーだけでは実行できない）

Linuxの適切なGPU・EGL環境で、取得したResVLAがimportできる状態にする。
BDDLと初期状態は同じ環境・同じタスクのファイルを指定する。
`--interest` はBDDLの対象物を正しく転記する。評価をよく見せるために対象物を追加しない。
`torch.load`対象は信頼できる配布元のファイルだけにする。

```sh
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python record_episode.py \
  --bddl /path/to/task.bddl --initial-states /path/to/task.pruned_init \
  --index 1 --seed 43 --instruction "instruction from the public task" \
  --interest OBJECT_NAMES_FROM_BDDL \
  --checkpoint /path/to/checkpoints/resVLA_libero.pt \
  --base-vlm /path/to/Qwen3-VL-2B-Instruct \
  --output recordings/task/init1_resvla
```

対照実験では同じタスク・初期状態・対象物で `--mode stationary_control` を指定し、
出力先を `recordings/task/init1_stationary_control` に変える。モデル引数は不要。
どちらも既存出力先を上書きしない。

```sh
python analyze_rollouts.py --source recordings --output analysis.json
```


第三者ソフトウェアの取り扱いは `THIRD_PARTY_NOTICES.md` を参照。
