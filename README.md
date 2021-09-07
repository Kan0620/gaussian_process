# ガウス過程　ベイズ最適化
* ガウス過程 ベイズ最適化で勉強したことをまとめるリポジトリ

## src/sampling_from_gaussian_process.py: sample_1d()
* gaussian, exponetial, periodicの3つのカーネルでガウス過程から1変数の関数をサンプリングして表示する関数です
* 引数

| No | 名前 | 型 | 説明 | 
| --- | --- | --- | --- |
| 1 | min_x | float | サンプリングする領域の最小値 デフォルトで-5.0 |
| 2 |  max_x | float |サンプリングする領域の最大値 デフォルトで5.0 |
| 3 |  n | int |サンプリングする点の個数 デフォルトで101 |
* 戻り値 

なし

* 実行例
以下のようなグラフが表示されます　seedは固定していないので実行のたびに違う表示がされます

![sample_1d](https://user-images.githubusercontent.com/80816547/132398592-4ae1dacb-02a4-4d9f-a212-1cf32b703c20.png)

## src/sampling_from_gaussian_process.py: sample_2d()
* 引数: kernel_modeのカーネルでガウス過程から2変数の関数をサンプリングして表示する関数です
* 引数

| No | 名前 | 型 | 説明 | 
| --- | --- | --- | --- |
| 1 | kernel_mode | str | 使用するカーネルの名前 'gaussian', 'exponetial', 'periodic'のいずれか |
| 2 | min_x | float | サンプリングする領域の最小値 デフォルトで-5.0 |
| 3 |  max_x | float |サンプリングする領域の最大値 デフォルトで5.0 |
| 4 |  n | int |サンプリングする点の個数 デフォルトで51 |

* 戻り値 

なし

* 実行例
以下のようなグラフが表示されます　seedは固定していないので実行のたびに違う表示がされます

![gaussian](https://user-images.githubusercontent.com/80816547/132399188-7c1d684a-5900-4c5b-862e-a5fc3ec08e50.png)
![exponetial](https://user-images.githubusercontent.com/80816547/132399362-eceec563-03b5-4ae4-a203-e048ba571082.png)
![periodic](https://user-images.githubusercontent.com/80816547/132399443-63c4376e-a223-4161-ad92-b80af8ca794c.png)
