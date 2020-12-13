import argparse

import librosa
import numpy as np
import pyworld as pw
import soundfile
from scipy.io.wavfile import read, write


def world_analysis_harvest(data_in: np.ndarray, fs_in: int) -> tuple:
    """WORLD特徴量の推定（高SNR音声用）

    :param data_in: 音声信号
    :param fs_in: サンプリングレート
    :return: f0, sp, ap
    """
    f_data = data_in.astype(np.float64)  # 念のためキャスト
    my_f0, _time = pw.harvest(f_data, fs_in)  # 基本周波数の推定
    my_sp = pw.cheaptrick(f_data, my_f0, _time, fs_in)  # スペクトル包絡の推定
    my_ap = pw.d4c(f_data, my_f0, _time, fs_in)  # 非周期性指標の推定
    return my_f0, my_sp, my_ap


def world_analysis(data_in: np.ndarray, fs_in: int) -> tuple:
    """WORLD特徴量の推定（DIO+StoneMask）

    :param data_in: 音声信号
    :param fs_in: サンプリングレート
    :return: f0, sp, ap
    """
    f_data = data_in.astype(np.float64)  # 念のためキャスト
    return pw.wav2world(f_data, fs_in)  # 一括で3種類の特徴量を推定


if __name__ == '__main__':
    # コマンドライン引数のパース（引数を変数に適宜代入する）
    parser = argparse.ArgumentParser(description='Wav file choice.')  # パーサーオブジェクトを作成
    path_default = "yukkuri.wav"  # デフォルトの値（引数が与えられなかったとき）を設定
    parser.add_argument('--path', type=str, help='Select WAV path', default=path_default)  # パーサーに引数を設定
    argv = parser.parse_args()  # 実行時の引数を解析
    path = argv.path  # 解析した引数argv.pathをpathに代入

    # 読み込み
    fs_, _ = read(path)  # サンプリングレートを算出するために一旦scipyで読み込み
    data, fs = librosa.load(path, sr=fs_, mono=True)  # Librosaでwavを読み込み
    data = data.astype(np.float64)  # WORLDに対応させるようにキャスト

    # WORLD変換
    # f0, ap, sp = world_analysis(data, fs)
    f0, ap, sp = world_analysis_harvest(data, fs)  # 特徴量抽出
    modified_f0 = f0 * 1.0  # 基本周波数変換
    out = pw.synthesize(modified_f0, sp, ap, fs)  # 音声再合成

    # out = data
    # soundfile 書き出し
    out_sf = out.astype(np.float32)  # soundfile書き出し用にキャスト
    soundfile.write("out_sf_" + path, out_sf, fs)  # 書き出し

    # scipy書き出し（おまけ）
    out_sp = out * 8000.0  # 整数型にする前に値を大きく
    out_sp = out_sp.astype(np.int16)  # 整数型にキャスト
    write("out_sp_" + path, fs, out_sp)  # 書き出し
