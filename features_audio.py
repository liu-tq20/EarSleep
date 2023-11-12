import numpy as np
import wavio
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy import signal


frame_num = "41349_328320000"

frame_rate = 24000 

# In[2]:

def file_reading():
    wav = wavio.read("../file_3min/"+frame_num+".wav")

    left_wav, right_wav = wav.data.T # 根据上述的描述，我们需要转置，才能得到两个声道的波形文件



    resample_rate = 1600
    time_len = int(len(left_wav)/wav.rate*1.0)

    wav = left_wav - right_wav

    resample_wav = signal.resample(wav, resample_rate*time_len)

    return resample_wav, resample_rate


# In[5]:


from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# In[6]:


from os.path import dirname, join as pjoin
from scipy.io import wavfile
from scipy.signal import butter, lfilter

def get_window_wav():
    lowcut,highcut=150,400

    resample_wav, resample_rate = file_reading()

    window_wav = butter_bandpass_filter(resample_wav, lowcut, highcut, fs=resample_rate, order=6)  # 150~400Hz带通
    # wavfile.write('off_plus_noise_filtered.wav', resample_rate, window_wav.astype(np.int16))


    window_wav = window_wav[100:]    # 去掉前100个点 还有15-100/1600s
    Timenew = np.linspace(0,len(resample_wav)/resample_rate*1.0,num=len(resample_wav))
    Timenew = Timenew[100:]

    return window_wav, resample_rate



import numpy as np
from scipy.signal import hilbert, chirp
import matplotlib.pyplot as plt

def get_inst_amplitude_wav():
    window_wav, resample_rate = get_window_wav()
    fs = resample_rate #sampling frequency
    duration = len(window_wav)/fs #duration of the signal
    t = np.arange(int(fs*duration)) / fs #time base

    envelop_wav = hilbert(window_wav) #form the analytical signal

    inst_amplitude_wav = np.abs(envelop_wav) #envelope extraction
    inst_phase_wav = np.unwrap(np.angle(envelop_wav))#inst phase
    inst_freq_wav = np.diff(inst_phase_wav)/(2*np.pi)*fs #inst frequency

    #Regenerate the carrier from the instantaneous phase
    regenerated_carrier_wav = np.cos(inst_phase_wav)
    return inst_amplitude_wav, resample_rate, duration


# In[11]:

def get_resample16_wav():
    inst_amplitude_wav, resample_rate, duration = get_inst_amplitude_wav()

    resample_rate = 16
    time_len = duration

    wav_16 = signal.resample(inst_amplitude_wav, int(resample_rate*time_len))

    Timenew = np.linspace(0,len(wav_16)/resample_rate*1.0,num=len(wav_16))


    plt.figure(figsize=(200, 16), dpi=100)
    plt.plot(Timenew, wav_16)
    plt.xlabel("time")

    plt.savefig("./wav5.jpg", bbox_inches='tight', dpi=100)
    # plt.show()

    return wav_16, resample_rate


# In[12]:


# In[13]:


import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



# In[14]:
def get_window16_wav():
    wav_16, resample_rate = get_resample16_wav()
    sampling_rate = resample_rate

    Timenew = np.linspace(0,len(wav_16)/sampling_rate*1.0,num=len(wav_16))
    cutoff = 1 # HZ
    window_wav = butter_lowpass_filter(wav_16, cutoff, fs=resample_rate, order=6)  # 0.1~5Hz带通

    plt.figure(figsize=(200, 16), dpi=100)
    plt.plot(Timenew, window_wav)
    plt.xlabel("time",fontsize=80)
    plt.xticks(size=40)
    plt.yticks(size=40)

    for item in Timenew:
        plt.axvline(x=item,c='r',ls='--',lw=1)
    plt.savefig("./wav7.jpg", bbox_inches='tight', dpi=100)
    # plt.show()

    return window_wav, resample_rate


# In[15]:


# sampling_rate = resample_rate

# Timenew = np.linspace(0,len(wav_16)/sampling_rate*1.0,num=len(wav_16))
# # Timenew = Timenew[16:]
# # window_wav = window_wav[16:]  # 不需要了

# plt.figure(figsize=(200, 16), dpi=100)
# plt.plot(Timenew, window_wav)
# plt.xlabel("time",fontsize=80)
# plt.xticks(size=40)
# plt.yticks(size=40)

# # for item in Timenew:
# #     plt.axvline(x=item,c='r',ls='--',lw=1)
# plt.savefig("./wav7.jpg", bbox_inches='tight', dpi=100)
# # plt.show()


# In[16]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rsp_plot_new(rsp_signals, sampling_rate=None, figsize=(10, 10), static=True):
    """**Visualize respiration (RSP) data**
    rsp_signals : DataFrame
        DataFrame obtained from :func:`.rsp_process`.
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    figsize : tuple
        The size of the figure (width, height) in inches.
    static : bool
        If True, a static plot will be generated with matplotlib.
        If False, an interactive plot will be generated with plotly.
        Defaults to True.
    """
    # Mark peaks, troughs and phases.
    peaks = np.where(rsp_signals["RSP_Peaks"] == 1)[0]
    troughs = np.where(rsp_signals["RSP_Troughs"] == 1)[0]
    inhale = np.where(rsp_signals["RSP_Phase"] == 1)[0]
    exhale = np.where(rsp_signals["RSP_Phase"] == 0)[0]

    nrow = 2

    # Determine mean rate.
    rate_mean = np.mean(rsp_signals["RSP_Rate"])

    if "RSP_Amplitude" in list(rsp_signals.columns):
        nrow += 1
        # Determine mean amplitude.
        amplitude_mean = np.mean(rsp_signals["RSP_Amplitude"])
    if "RSP_RVT" in list(rsp_signals.columns):
        nrow += 1
        # Determine mean RVT.
        rvt_mean = np.mean(rsp_signals["RSP_RVT"])
    if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
        nrow += 1

    # Get signals marking inspiration and expiration.
    exhale_signal, inhale_signal = _rsp_plot_phase(rsp_signals, troughs, peaks)

    # Determine unit of x-axis.
    if sampling_rate is not None:
        x_label = "Time (seconds)"
        x_axis = np.linspace(0, len(rsp_signals) / sampling_rate, len(rsp_signals))
    else:
        x_label = "Samples"
        x_axis = np.arange(0, len(rsp_signals))

    if static:
        fig, ax = plt.subplots(nrows=nrow, ncols=1, sharex=True, figsize=figsize)

        last_ax = fig.get_axes()[-1]
        last_ax.set_xlabel(x_label)

        # Plot cleaned and raw respiration as well as peaks and troughs.
        ax[0].set_title("Raw and Cleaned Signal")
        fig.suptitle("Respiration (RSP)", fontweight="bold")

        ax[0].plot(
            x_axis, rsp_signals["RSP_Raw"], color="#B0BEC5", label="Raw", zorder=1
        )
        ax[0].plot(
            x_axis,
            rsp_signals["RSP_Clean"],
            color="#2196F3",
            label="Cleaned",
            zorder=2,
            linewidth=1.5,
        )
        
        for x_peaks in x_axis[peaks]:
            ax[0].axvline(x=x_peaks,c='r',ls='--',lw=1)
        
        ax[0].scatter(
            x_axis[peaks],
            rsp_signals["RSP_Clean"][peaks],
            color="red",
            label="Exhalation Onsets",
            zorder=3,
        )
        ax[0].scatter(
            x_axis[troughs],
            rsp_signals["RSP_Clean"][troughs],
            color="orange",
            label="Inhalation Onsets",
            zorder=4,
        )

        # Shade region to mark inspiration and expiration.
        ax[0].fill_between(
            x_axis[exhale],
            exhale_signal[exhale],
            rsp_signals["RSP_Clean"][exhale],
            where=rsp_signals["RSP_Clean"][exhale] > exhale_signal[exhale],
            color="#CFD8DC",
            linestyle="None",
            label="exhalation",
        )
        ax[0].fill_between(
            x_axis[inhale],
            inhale_signal[inhale],
            rsp_signals["RSP_Clean"][inhale],
            where=rsp_signals["RSP_Clean"][inhale] > inhale_signal[inhale],
            color="#ECEFF1",
            linestyle="None",
            label="inhalation",
        )

        ax[0].legend(loc="upper right")

        # Plot rate and optionally amplitude.
        ax[1].set_title("Breathing Rate")
        ax[1].plot(
            x_axis,
            rsp_signals["RSP_Rate"],
            color="#4CAF50",
            label="Rate",
            linewidth=1.5,
        )
        ax[1].axhline(y=rate_mean, label="Mean", linestyle="--", color="#4CAF50")
        ax[1].legend(loc="upper right")

        if "RSP_Amplitude" in list(rsp_signals.columns):
            ax[2].set_title("Breathing Amplitude")

            ax[2].plot(
                x_axis,
                rsp_signals["RSP_Amplitude"],
                color="#009688",
                label="Amplitude",
                linewidth=1.5,
            )
            ax[2].axhline(
                y=amplitude_mean, label="Mean", linestyle="--", color="#009688"
            )
            ax[2].legend(loc="upper right")

        if "RSP_RVT" in list(rsp_signals.columns):
            ax[3].set_title("Respiratory Volume per Time")

            ax[3].plot(
                x_axis,
                rsp_signals["RSP_RVT"],
                color="#00BCD4",
                label="RVT",
                linewidth=1.5,
            )
            ax[3].axhline(y=rvt_mean, label="Mean", linestyle="--", color="#009688")
            ax[3].legend(loc="upper right")

        if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
            ax[4].set_title("Cycle Symmetry")

            ax[4].plot(
                x_axis,
                rsp_signals["RSP_Symmetry_PeakTrough"],
                color="green",
                label="Peak-Trough Symmetry",
                linewidth=1.5,
            )
            ax[4].plot(
                x_axis,
                rsp_signals["RSP_Symmetry_RiseDecay"],
                color="purple",
                label="Rise-Decay Symmetry",
                linewidth=1.5,
            )
            ax[4].legend(loc="upper right")
        return fig
    else:
        # Generate interactive plot with plotly.
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

        except ImportError as e:
            raise ImportError(
                "NeuroKit error: rsp_plot(): the 'plotly'",
                " module is required when 'static' is False.",
                " Please install it first (`pip install plotly`).",
            ) from e

        subplot_titles = ["Raw and Cleaned Signal", "Breathing Rate"]
        if "RSP_Amplitude" in list(rsp_signals.columns):
            subplot_titles.append("Breathing Amplitude")
        if "RSP_RVT" in list(rsp_signals.columns):
            subplot_titles.append("Respiratory Volume per Time")
        if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
            subplot_titles.append("Cycle Symmetry")
        subplot_titles = tuple(subplot_titles)
        fig = make_subplots(
            rows=nrow,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
        )

        # Plot cleaned and raw RSP
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=rsp_signals["RSP_Raw"], name="Raw", marker_color="#B0BEC5"
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=rsp_signals["RSP_Clean"],
                name="Cleaned",
                marker_color="#2196F3",
            ),
            row=1,
            col=1,
        )

        # Plot peaks and troughs.
        fig.add_trace(
            go.Scatter(
                x=x_axis[peaks],
                y=rsp_signals["RSP_Clean"][peaks],
                name="Exhalation Onsets",
                marker_color="red",
                mode="markers",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis[troughs],
                y=rsp_signals["RSP_Clean"][troughs],
                name="Inhalation Onsets",
                marker_color="orange",
                mode="markers",
            ),
            row=1,
            col=1,
        )

        # TODO: Shade region to mark inspiration and expiration.

        # Plot rate and optionally amplitude.
        fig.add_trace(
            go.Scatter(
                x=x_axis, y=rsp_signals["RSP_Rate"], name="Rate", marker_color="#4CAF50"
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=[rate_mean] * len(x_axis),
                name="Mean Rate",
                marker_color="#4CAF50",
                line=dict(dash="dash"),
            ),
            row=2,
            col=1,
        )

        if "RSP_Amplitude" in list(rsp_signals.columns):
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=rsp_signals["RSP_Amplitude"],
                    name="Amplitude",
                    marker_color="#009688",
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=[amplitude_mean] * len(x_axis),
                    name="Mean Amplitude",
                    marker_color="#009688",
                    line=dict(dash="dash"),
                ),
                row=3,
                col=1,
            )

        if "RSP_RVT" in list(rsp_signals.columns):
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=rsp_signals["RSP_RVT"],
                    name="RVT",
                    marker_color="#00BCD4",
                ),
                row=4,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=[rvt_mean] * len(x_axis),
                    name="Mean RVT",
                    marker_color="#00BCD4",
                    line=dict(dash="dash"),
                ),
                row=4,
                col=1,
            )

        if "RSP_Symmetry_PeakTrough" in list(rsp_signals.columns):
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=rsp_signals["RSP_Symmetry_PeakTrough"],
                    name="Peak-Trough Symmetry",
                    marker_color="green",
                ),
                row=5,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=rsp_signals["RSP_Symmetry_RiseDecay"],
                    name="Rise-Decay Symmetry",
                    marker_color="purple",
                ),
                row=5,
                col=1,
            )

        fig.update_layout(title_text="Respiration (RSP)", height=1250, width=750)
        for i in range(1, nrow + 1):
            fig.update_xaxes(title_text=x_label, row=i, col=1)
        
        return fig


# =============================================================================
# Internals
# =============================================================================
def _rsp_plot_phase(rsp_signals, troughs, peaks):
    exhale_signal = pd.Series(np.full(len(rsp_signals), np.nan))
    exhale_signal[troughs] = rsp_signals["RSP_Clean"][troughs].values
    exhale_signal[peaks] = rsp_signals["RSP_Clean"][peaks].values
    exhale_signal = exhale_signal.fillna(method="backfill")

    inhale_signal = pd.Series(np.full(len(rsp_signals), np.nan))
    inhale_signal[troughs] = rsp_signals["RSP_Clean"][troughs].values
    inhale_signal[peaks] = rsp_signals["RSP_Clean"][peaks].values
    inhale_signal = inhale_signal.fillna(method="ffill")

    return exhale_signal, inhale_signal


import neurokit2 as neuro

def neuro_features(window):
    window_wav, sampling_rate = get_window16_wav()
    ignals, info = neuro.rsp_process(window_wav, sampling_rate=16, method="biosppy", report='text') # better
    # ignals, info = nk.rsp_process(window_wav, sampling_rate=16, report="text")
    rsp_intervalrelated_features = neuro.rsp_intervalrelated(ignals, sampling_rate=16)
    print("rsp_intervalrelated_features: ", rsp_intervalrelated_features)
    fig1 = rsp_plot_new(ignals, sampling_rate=16, figsize=(100,40))
    fig1.savefig("./rsp_2_"+frame_num+".jpg", bbox_inches='tight', dpi=100)

    # print(ignals)
    # print(info)

    return rsp_intervalrelated_features




# In[17]:

def extract_features_audio(window):
    """
        Make sure that X is an N x d matrix, where N is the number 
    of data points and d is the number of features.
    
    """
    features = neuro_features(window)
    print(features)
    # x = np.array([neuro_features(window)])
    # return x

