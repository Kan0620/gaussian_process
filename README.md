# ガウス過程　ベイズ最適化
* ガウス過程 ベイズ最適化で勉強したことをまとめるリポジトリ
* ガウス過程 ベイズ最適化を扱うライブラリを使わないよう実装しました

## src/sampling_from_gaussian_process.py: sample_1d()
gaussian, exponetial, periodicの3つのカーネルでガウス過程から1変数の関数をサンプリングして表示する関数です
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
引数で指定されたカーネルでガウス過程から2変数の関数をサンプリングして表示する関数です
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

## src/gaussian_process_regression.py: regression()
ある区間の中で均等にサンプリングされた点から区間全体での関数値を予測し表示する関数です

* 引数

| No | 名前 | 型 | 説明 | 
| --- | --- | --- | --- |
| 1 | n_observed | int | 観測するデータ点の数 |
| 2 | objectve_function |  | 予測する関数　例を下に記載しています |
| 3 |  min_x | float |サンプリングする領域の最大値 デフォルトで-10.0 |
| 3 |  max_x | float |サンプリングする領域の最大値 デフォルトで10.0 |
| 4 |  n | int |サンプリングする点の個数 デフォルトで201 |

object_functionの例
```
def objective_function(x):

return 3 * np.cos(x-1) - np.abs(x-1)
```
* 戻り値 

なし

* 実行例
以下のようなグラフが表示されます　objective_functionは上の関数です
![regression](https://user-images.githubusercontent.com/80816547/132416545-6a065aec-2b3e-41a3-a5e7-8f51b9063c7e.png)

## src/gaussian_process_regression.py: bayesian_optim()
与えられた関数の最大値を与える入力をベイズ最適化を用いて見つける関数です

実装中
