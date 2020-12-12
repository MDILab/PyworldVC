import argparse

import librosa
import numpy as np
import pyworld as pw
import soundfile
from scipy.io.wavfile import read, write


def world_analysis(data_in: np.ndarray, fs_in: int):
    """WORLD特徴量の推定

    :param data_in: 音声信号
    :param fs_in: サンプリングレート
    :return: f0, sp, ap
    """
    f_data = data_in.astype(np.float64)  # 念のためキャスト
    return pw.wav2world(f_data, fs_in)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wav file choice.')
    path_default = "yukkuri.wav"
    parser.add_argument('--path', type=str, help='Select WAV path', default=path_default)
    argv = parser.parse_args()
    path = argv.path

    # 読み込み
    fs_, data_ = read(path)
    data, fs = librosa.load(path, sr=fs_, mono=True)
    data = data.astype(np.float64)

    # WORLD変換
    f0, ap, sp = world_analysis(data, fs)
    modified_f0 = f0 * 2.0
    out = pw.synthesize(modified_f0, sp, ap, fs)

    # soundfile 書き出し
    out_sf = out.astype(np.float32)
    soundfile.write("out_sf_" + path, out_sf, fs)

    # scipy書き出し
    out_sp = out * 8000.0
    out_sp = out_sp.astype(np.int16)
    write("out_sp_" + path, fs, out_sp)
