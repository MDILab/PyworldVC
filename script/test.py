import argparse
import struct
import wave

import librosa
import numpy as np
import pyworld as pw
import soundfile
from scipy.io.wavfile import read


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


def wav_write(filename, data, channels, fs):
    """ファイル保存
    """
    binaryData = struct.pack("h" * len(data), *data)
    # out = wave.Wave_write(filename)
    with wave.Wave_write(filename) as out:
        # "lowpass.wav","highpass.wav","bandpass.wav","VC1_test.wav"
        param = (channels, 2, fs, len(binaryData), 'NONE', 'not compressed')
        out.setparams(param)
        out.writeframes(binaryData)
    # out.close()


def wav_read(filename):
    """ファイル読み込み

    :param filename: ファイルパス
    :return: data, channels, fs
    """
    # openしたらcloseが必要になるので、忘れないようにwith文を使いましょう。
    with wave.open(filename, "r") as wf:
        channels = wf.getnchannels()
        fs = wf.getframerate()

        buf = wf.readframes(wf.getnframes())
    data = np.frombuffer(buf, dtype="int16").astype(np.float)
    return data, channels, fs


if __name__ == '__main__':
    # デバッグで使ったやつ（Falseで元のコードが動きます）
    debug = True

    # コマンドライン引数のパース（引数を変数に適宜代入する）
    parser = argparse.ArgumentParser(description='Wav file choice.')  # パーサーオブジェクトを作成
    path_default = "yukkuri.wav"  # デフォルトの値（引数が与えられなかったとき）を設定
    parser.add_argument('--path', type=str, help='Select WAV path', default=path_default)  # パーサーに引数を設定
    argv = parser.parse_args()  # 実行時の引数を解析
    path = argv.path  # 解析した引数argv.pathをpathに代入

    # 読み込み
    if debug:
        # Librosaを使うと楽なのでおすすめ
        fs_, _ = read(path)  # サンプリングレートを算出するために一旦scipyで読み込み
        data, fs = librosa.load(path, sr=fs_, mono=True)  # Librosaでwavを読み込み
        data = data.astype(np.float64)  # WORLDに対応させるようにキャスト
        channels = 1  # モノラルに指定しているので1
    else:
        # 元のコードと同じ手順で読み込み
        data, channels, fs = wav_read(path)

    # WORLD変換
    if debug:
        f0, sp, ap = world_analysis_harvest(data, fs)  # 特徴量抽出（高SNR用）
    else:
        f0, sp, ap = world_analysis(data, fs)  # 特徴量抽出（元のコード）
    modified_f0 = f0 * 2.0  # 基本周波数変換
    out = pw.synthesize(modified_f0, sp, ap, fs)  # 音声再合成

    # 書き出し
    if debug:
        # soundfile書き出し（Librosa標準の方法）
        out_sf = out.astype(np.float32)  # soundfile書き出し用にキャスト
        soundfile.write("out_sf_" + path, out_sf, fs)  # 書き出し
    else:
        # 元のコードと同じ手順で書き出し
        wav_write("out_org_" + path, out.astype(np.int16), channels, fs)
