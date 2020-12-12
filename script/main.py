import argparse
import struct
import tkinter
import wave

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import pyworld as pw


# ファイル保存
def wavWrite(filename, data, channels, fs):
    binaryData = struct.pack("h" * len(data), *data)
    out = wave.Wave_write(filename)
    # "lowpass.wav","highpass.wav","bandpass.wav","VC1_test.wav"
    param = (channels, 2, fs, len(binaryData), 'NONE', 'not compressed')
    out.setparams(param)
    out.writeframes(binaryData)
    out.close


# ファイル読み込み
def wavRead(filename):
    wf = wave.open(filename, "r")
    channels = wf.getnchannels()
    fs = wf.getframerate()

    buf = wf.readframes(wf.getnframes())
    data = np.frombuffer(buf, dtype="int16").astype(np.float)
    return data, channels, fs


# WORLDの特徴量推定部分をファイル読み込みから分離しましょう。
def world_analysis(data, fs):
    _f0, _time4 = pw.dio(data, fs)  # 基本周波数の抽出
    f0 = pw.stonemask(data, _f0, _time4, fs)  # 基本周波数の修正

    sp = pw.cheaptrick(data, f0, _time4, fs)  # スペクトル包絡の抽出
    ap = pw.d4c(data, f0, _time4, fs)  # 非周期性指標の抽出
    return f0, sp, ap


# sinc関数
def sinc(x):
    if x == 0.0:
        return 1.0
    else:
        return np.sin(x) / x


# フィルタ計数bの決定(ローパスフィルタ)
# fs:サンプリング周波数
# cutoff:カットオフ周波数
# delta:遷移帯域幅
def lpf(fs, cutoff, delta):
    cutoff = float(cutoff) / fs  # カットオフ周波数の正規化
    delta = float(delta) / fs  # 遷移帯域幅の正規化

    # タップ数(フィルタ係数j + 1の数) J + 1は奇数になるように
    j = round(3.1 / delta) - 1  # round:四捨五入
    if (j + 1) % 2 == 0:  # j + 1は奇数である必要があるてめ、2で割ったあまりが0のときは1を足す
        j += 1
    j = int(j)

    # タップ数の確認(遷移帯域幅と反比例)
    print("filter coefficients: " + str(j + 1))

    # フィルタ係数の計算
    b = []
    for m in range(int(-j / 2), int(j / 2 + 1)):  # 0を中心にする必要があるため、-j/ 2 ～ j /2
        b.append(2.0 * cutoff * sinc(2.0 * np.pi * cutoff * m))

    # ハニング窓関数をかける(窓関数法)
    hanningWindow = np.hanning(j + 1)
    b = b * hanningWindow

    return b


####### フィルタ係数bの決定（ハイパスフィルタ）
# fs, サンプリング周波数
# cutoff, カットオフ周波数
# delta, 遷移帯域幅
def hpf(fs, cutoff, delta):
    cutoff = float(cutoff) / fs  # カットオフ周波数の正規化
    delta = float(delta) / fs  # 遷移帯域幅の正規化

    # タップ数（フィルタ係数J+1の数）j+1は奇数になるように
    j = round(3.1 / delta) - 1
    if (j + 1) % 2 == 0:
        j += 1
    j = int(j)

    # タップ数の確認（遷移帯域幅と反比例）
    print("filter coefficients: " + str(j + 1))

    # フィルタ係数の計算
    b = []
    for m in range(int(-j / 2), int(j / 2 + 1)):
        b.append(sinc(np.pi * m) - 2 * cutoff * sinc(2 * np.pi * cutoff * m))

    # ハニング窓関数をかける（窓関数法）
    hanningWindow = np.hanning(j + 1)
    b = b * hanningWindow

    return b


####### フィルタ係数bの決定（バンドパスフィルタ）
# fs, サンプリング周波数
# cutoff1, カットオフ周波数(下限)
# cutoff2, カットオフ周波数（上限）
# delta, 遷移帯域幅
def bpf(fs, cutoff1, cutoff2, delta):
    cutoff1 = float(cutoff1) / fs  # カットオフ周波数の正規化
    cutoff2 = float(cutoff2) / fs  # カットオフ周波数の正規化
    delta = float(delta) / fs  # 遷移帯域幅の正規化

    # タップ数（フィルタ係数J+1の数）J+1は奇数になるように
    J = round(3.1 / delta) - 1
    if (J + 1) % 2 == 0:
        J += 1
    J = int(J)

    # タップ数の確認（遷移帯域幅と反比例）
    print("filter coefficients: " + str(J + 1))

    # フィルタ係数の計算
    b = []
    for m in range(int(-J / 2), int(J / 2 + 1)):
        b.append(2.0 * cutoff2 * sinc(2.0 * np.pi * cutoff2 * m) - 2.0 * cutoff1 * sinc(2.0 * np.pi * cutoff1 * m))

    # ハニング窓関数をかける（窓関数法）
    hanningWindow = np.hanning(J + 1)
    b = b * hanningWindow

    return b


# clickイベント
# 以下のように書くとボタン1～3をまとめられます。
# ここの処理はデコレータというものを用いているので、調べてみれば詳しい挙動が分かると思います。
def deco_btn_click(file_path: str):
    def btn_click():
        chunk = 1024

        wf = wave.open(file_path, "rb")
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                        rate=wf.getframerate(), output=True)
        data = wf.readframes(chunk)

        while len(data) != 0:
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()

        # 可視化
        data, _, fs = wavRead(file_path)
        f0, ap, sp = world_analysis(data, fs)

        plt.plot(out, label="Raw Data")  # 読み込んだデータを可視化
        plt.legend(fontsize=10)
        plt.show()

        plt.plot(f0, linewidth=3, color="green", label="F0 contour")
        plt.legend(fontsize=10)
        plt.show()

        plt.plot(sp, linewidth=3, color="green", label="F0 contour")
        plt.legend(fontsize=10)
        plt.show()

        plt.plot(ap, linewidth=3, color="green", label="F0 contour")
        plt.legend(fontsize=10)
        plt.show()

    return btn_click


def btn4_click():
    chunk = 1024

    wf = wave.open("VC1_test.wav", "rb")
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)
    data = wf.readframes(chunk)

    while len(data) != 0:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()


# ここからメイン
if __name__ == '__main__':
    # とりあえずデバッグ用に作っただけです。GUIをオフにして実行する用です。
    # GUI実行時はFalseにしてください。
    debug = True

    # 実行時の引数はこのように明示的に指定しましょう（分からなかったらまた聞いてください）
    parser = argparse.ArgumentParser(description='Wav file choice.')
    path_default = "yukkuri.wav"
    parser.add_argument('--path', type=str, help='Select WAV path', default=path_default)
    argv = parser.parse_args()
    path = argv.path

    ##########################
    # フィルタを掛ける処理の部分
    ##########################

    data, channels, fs = wavRead(path)  # コマンドラインから入力ファイル名を指定

    lpf_cutoff = 500  # カットオフ周波数 [Hz]
    lpf_delta = 100  # 遷移帯域幅 [Hz]

    hpf_cutoff = 500  # カットオフ周波数 [Hz]
    hpf_delta = 100  # 遷移帯域幅 [Hz]

    bpf_cutoff1 = 500  # カットオフ周波数（下限） [Hz]
    bpf_cutoff2 = 1000  # カットオフ周波数（上限） [Hz]
    bpf_delta = 100  # 遷移帯域幅 [Hz]

    b_lpf = lpf(fs, lpf_cutoff, lpf_delta)  # フィルタ係数bの計算
    b_hpf = hpf(fs, hpf_cutoff, hpf_delta)  # フィルタ係数bの計算
    b_bpf = bpf(fs, bpf_cutoff1, bpf_cutoff2, bpf_delta)  # フィルタ係数bの計算

    # dataとbを畳み込み
    out = np.convolve(data, b_lpf, 'same')
    out2 = np.convolve(data, b_hpf, 'same')
    out3 = np.convolve(data, b_bpf, 'same')

    wavWrite("lowpass.wav", out.astype(np.int16), channels, fs)
    wavWrite("highpass.wav", out2.astype(np.int16), channels, fs)
    wavWrite("bandpass.wav", out3.astype(np.int16), channels, fs)

    ##########################
    # WORLDでVC処理をする部分
    ##########################
    f0, ap, sp = world_analysis(data, fs)
    modified_f0 = 2.0 * f0  # ピッチシフト
    out4 = pw.synthesize(modified_f0, sp, ap, fs)

    wavWrite("VC1_test.wav", out4.astype(np.int16), channels, fs)

    if not debug:
        # 画面作成
        tki = tkinter.Tk()
        tki.geometry('300x200')  # 画面サイズの設定
        tki.title('ボタンのサンプル')  # 画面タイトルの設定

        # ボタンの作成

        # ボタンの設定(text=ボタンに表示するテキスト)
        btn1 = tkinter.Button(tki, text='ローパスフィルタ', command=deco_btn_click("lowpass.wav"))
        btn2 = tkinter.Button(tki, text='ハイパスフィルタ', command=deco_btn_click("highpass.wav"))
        btn3 = tkinter.Button(tki, text='バンドパスフィルタ', command=deco_btn_click("bandpass.wav"))
        btn4 = tkinter.Button(tki, text='声1', command=btn4_click)
        # ボタンを配置する位置の設定
        btn1.place(x=20, y=100)
        btn2.place(x=110, y=100)
        btn3.place(x=200, y=100)
        btn4.place(x=20, y=150)

        # 画面をそのまま表示
        tki.mainloop()
