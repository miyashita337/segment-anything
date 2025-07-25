# 修正2202507051556
## sam_yolo_character_segment.pyの方向性
今まで修正をしてくれてありがとうございます
さらにもっと抽出力を高めるため以下のような、自動的な精査ができるパイプラインがほしい

# sam_yolo_character_segment.pyのキャラクター抽出に関して

* 汎用性のあるキャラクターの抽出をしてください

## 参考
* 以下に私自身が手入力と画像エディタで切り出した画像があります。まずinput outputのフォルダを見てください
  * --inpput : C:\AItools\lora\train\diff_aichi\org
  * --output : C:\AItools\lora\train\diff_aichi\clipped_bounding_box
    * 私は C:\AItools\lora\train\diff_aichi\orgをみて、手動でキャラクターを抜き出しました、切り出した出力先は C:\AItools\lora\train\diff_aichi\clipped_bounding_box です
      * (※outputにinputの画像が存在しない場合はskipしてます)

# 依頼
* Claude Codeには上記私が手動でやった手動抽出を再現できるぐらいのキャラクター抽出力を実装してほしいです
* 実装、モデル、調整のパラメータ、その他設定など、基本細かいことはお任せします
* ただし、あなたが再現したキャラ抽出はの評価である「再現度」の評価は私が行います
  * 一律〜％上がったという判定模することもあれば、この画像だけ再現度低いという細かい評価もします
  * 何度も見直して、フィードバック来たらブラッシュアップして、また私が再評価する、その繰り返しをします
    * 最終的に[再現度９０％]を私が出せれるように実装してほしい

## まずは最初に
私が手動で抽出した
```
  * --inpput : C:\AItools\lora\train\diff_aichi\org
  * --output : C:\AItools\lora\train\diff_aichi\clipped_bounding_box
```
上記をあなた(ClaudeCode)で再現してほしい

## フィードバッグがあるたびに
git でバージョン管理をしてほしい
ブランチ名はお任せ
* ただし、tag(version)は以下のように
  * 0.0.1から開始
  * 小数点第３位：マイナーバージョンアップ
  * 小数点第２位：ミドルバージョンアップ
  * 小数点第１位：メジャーバージョンアップ


# マシンスペック
* スペックは現状のWindowsマシンはこちらの記事のマシンを利用してます
https://zenn.dev/harieshokunin/articles/3aca5170f9ee8a

CPU	AMD Ryzen 7 7700（無印、内蔵GPUあり）
SSD	WD_BLACK SN850X 2TB Gen4
ケース	Thermaltake Core P6 TG
OS	Windows 11 Home（パッケージ版）
メモリ	Crucial DDR5-5600 32GB（2x16GB）
GPU	玄人志向 RTX 4070 Ti SUPER
マザーボード	MSI MAG B650 TOMAHAWK WIFI
PSU	玄人志向 850W GOLD（KRPW-GA850W/90+）

# tips
簡単に使えるリソースはカジュアルに使って
* 時間
* 自己検証
* 自動繰り返し



# 実装、デバッグ、フィードバック

* 目的に向かってPDCAを取る
  * 修正したらデバッグして,再現率を高くするように修正をしてください
  * 再現率の高い低いは最終的に私が教師となって何％かをフィードバックします
  * 一度抽出をしたら、私にフィードバックさせるためにリクエストしに来てください
  * 私のフィードバッグはできるだけ具体的に数値を言えるようにしますが、時には抽象度が高いものもあります
    * 理解ができない場合は、実行する前にヒアリングを先にしてください、先に実行してはいけません


## 注意事項
* ハングアップしないぐらいのスペックで実装をしてください
* WSLの実行でもWindowsネイティブの実行でもどちらでも構いません
* マシンそのものの全部なくすような振る舞いはしないようにしてください
https://zenn.dev/taiyogakuse/articles/claude-safe-command
この記事の対応をしておいて

### 解決ができない場合
* ClaudeCodeだけではどうにもできない場合があります
  * 目的に向かって回り道をしてる可能性がある(具体的に３回ほど迂回策をとってるなら、本筋ではない対応をしてるなら)
  * その場合は私にヒアリングしてきてください
#### 最近の例
  * 最近だと、修正内容をgitに上げる依頼をしたけど、gitのssh鍵やprivate access tokenなどを渡してなかった
  * それをわからずClaudeCodeは迂回をしようとして試行錯誤してました
```
● Bash(git push fork main)
  ⎿  Error: ssh_askpass: exec(/usr/bin/ssh-askpass): No such file or directory
     Host key verification failed.
     fatal: Could not read from remote repository.

     Please make sure you have the correct access rights
     and the repository exists.

● Bash(ssh-keyscan github.com >> ~/.ssh/known_hosts)
  ⎿  # github.com:22 SSH-2.0-423d6210
     # github.com:22 SSH-2.0-423d6210
     # github.com:22 SSH-2.0-423d6210
     … +2 lines (ctrl+r to expand)

● Bash(git push fork main)
  ⎿  Error: git@github.com: Permission denied (publickey).
     fatal: Could not read from remote repository.

     Please make sure you have the correct access rights
     and the repository exists.

● Bash(ssh-add ~/.ssh/id_rsa)
  ⎿  Error: Could not open a connection to your authentication agent.

● Bash(chmod 600 ~/.ssh/id_rsa && eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_rsa)
  ⎿  Error: ssh_askpass: exec(/usr/bin/ssh-askpass): No such file or directory

     Agent pid 4075

● Bash(ssh-add ~/.ssh/id_rsa)
  ⎿  Error: Could not open a connection to your authentication agent.

● Bash(export SSH_AUTH_SOCK=/tmp/ssh-agent.sock && ssh-agent -a $SSH_AUTH_SOCK > /dev/null 2>&1 && ssh-add ~/.ssh/id_rsa)
  ⎿  Error: ssh_askpass: exec(/usr/bin/ssh-askpass): No such file or directory

```
  * 今後似たようなことがあれば、作業をストップして聞きに来てください


# ヒアリングに関して
* 依頼の実行前
* 依頼の実行中
* フィードバック後
不明点があれば基本すぐに作業を中断して、ヒアリングしてきてください
* 手戻りは発生したくないです

まずはこの依頼書を読んだら不明点を聞き出してください
