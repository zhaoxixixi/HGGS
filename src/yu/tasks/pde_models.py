from random import random
from typing import List, Any

import numpy as np
from scipy import signal
from scipy.integrate import odeint

from yu.const.normal import PDEType
from yu.core.logger import logger
from yu.exception.normal import PDEException
from yu.tools.array import has_nan, is_exceeded
from yu.tools.fft import to_fft, search_by_acf

from yu.tasks.ode_model import Normal_Equation, Normal_Parameters

class Truth:
    """
    Validity of Periods Values:
    If the calculation process fails, resulting in NAN or INF values and rendering further computation impossible, periods will not be saved.
    Periods and curve_pde_types values are considered valid only when the calculation is successful.
    Note that even if the calculation is successful, it is still possible for the periods to be INF.
    """
    MIN_PEAKS = 2
    Y_RATIO_DELTA = 0.15
    Y_RATIO_PERIOD_N = 2
    Y_RATIO_PERIOD_N_list = [2, 3, 5, 7, 13, 31, 67]
    Y_TAIL_RATE = 0.1
    Y_TAIL_DELTA = 0.01
    # config fields
    model_name: str
    curve_names: List[str]
    curve_colors: List[str] = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    equation_idx: List[int]  # for output Params
    threshold: float

    # calc fields
    # y init
    y0: Any
    # Sequence of t (to share the t coordinates)
    t: Any
    # Number of samples
    N: int
    # Sampling frequency
    Fs: float
    truth: Any

    # prepare fields
    # Ground truth values
    periods: List[Any]
    # Flag to indicate if a valid solution was obtained
    success: bool = True
    # Type of PDE solution
    pde_type: PDEType
    # List of PDE types for each component of y
    curve_pde_types: List[PDEType]

    def config(
            self,
            model_name: str,
            curve_names: List[str],
            equation_idx: List[int],
            threshold: float,
            curve_colors: List[str] = None,
            equs: Normal_Equation=None,
            _Parameters: Any=None
    ):
        self.model_name = model_name
        self.curve_names = curve_names
        if curve_colors:
            self.curve_colors = curve_colors
        self.equation_idx = equation_idx
        self.threshold = threshold

        self.equs = equs
        self._Parameters = _Parameters

        if self._Parameters.ode_model_name == 'brusselator':
            self.Y_RATIO_DELTA = 0.10
        elif self._Parameters.ode_model_name == 'PS2_01':
            self.Y_RATIO_DELTA = 0.10
            self.Y_RATIO_PERIOD_N_list.append(101)
        return self

    def prepare(self):
        """ Preparation before computation: Reset the success, pde_type, periods, and curve_pde_types fields. """
        self.success = True
        self.pde_type = PDEType.OSCILLATION
        self.periods = []
        self.curve_pde_types = []

    def validate(self):
        """ Verify the validity of the solution after computation. """
        for item in self.truth:
            if has_nan(item):
                self.success = False
                raise PDEException(PDEType.NAN)
        return self.success

    def calc(self, y0, T, T_unit, params: Normal_Parameters):
        """ Compute the equations and update the relevant fields. """
        # print(y0)
        self.y0 = y0
        self.N = int(T / T_unit)
        self.Fs = 1 / T_unit
        self.t = np.asarray([i * T_unit for i in range(self.N)])
        try:
            self.prepare()
            self.truth = odeint(self.equs.instantiate, self.y0, self.t, (params, ))
            self.validate()
            # Reinitialize values and compute.
            y0 = self.truth[-1]
            self.prepare()
            self.truth = odeint(self.equs.instantiate, y0, self.t, (params,))
            self.validate()

            if self._Parameters.ode_model_name == 'PP':
                # Reinitialize values and compute.
                for _ in range(0):
                    y0 = self.truth[-1]
                    self.prepare()
                    self.truth = odeint(self.equs.instantiate, y0, self.t, (params,))
                    self.validate()
            elif self._Parameters.ode_model_name == 'PS2_01':
                for _ in range(0):
                    y0 = self.truth[-1]
                    self.prepare()
                    self.truth = odeint(self.equs.instantiate, y0, self.t, (params,))
                    self.validate()
            elif self._Parameters.ode_model_name == 'brusselator':
                for _ in range(0):
                    y0 = self.truth[-1]
                    self.prepare()
                    self.truth = odeint(self.equs.instantiate, y0, self.t, (params,))
                    self.validate()
            
        except PDEException as e:
            self.success = False
            self.pde_type = e.pde_type
            logger.error(f'pde calc fails: {e}')
        return self

    def _update_periods(
            self,
            period,
            score,
            fft_series,
            sample_freq,
            y_ratio,
            y_tail_var,
    ):
        """ update periods """
        self.periods.append((period, score, fft_series, sample_freq, y_ratio, y_tail_var))

    def find_period(self):
        """ calculate frequency """
        if not self.success:
            return self

        for i in range(self.truth.shape[1]):
            if i not in self.equation_idx:
                continue

            data = np.swapaxes(self.truth, 0, 1)[i]
            y_tail_var = np.std(data[-int(len(data) * self.Y_TAIL_RATE):])
            # Perform Fourier transform to identify the dominant periodic component.
            fft_series, sample_freq = to_fft(data, self.Fs, self.N, normalized=True)

            # print(sample_freq)
            top_k_idx = np.argpartition(fft_series[1:], -1)[-1]

            # print(sample_freq[top_k_idx])
            # No fundamental frequency detected, non-periodic behavior.
            if not sample_freq[top_k_idx]:
                self._update_periods(
                    np.inf, 0,
                    fft_series,
                    sample_freq,
                    0,
                    y_tail_var=y_tail_var,
                )
                continue
            if self._Parameters.ode_model_name == 'PP':
                peaks = signal.find_peaks_cwt(data, max(1, int(self.Fs / sample_freq[top_k_idx] / 8)))
            else:
                peaks, _ = signal.find_peaks(data, distance=max(1, int(self.Fs / sample_freq[top_k_idx] / 2)))
            # print(max(1, int(self.Fs / sample_freq[top_k_idx] / 2)))
            # print(int(self.Fs / sample_freq[top_k_idx] / 2))
            # print(peaks.size, self.MIN_PEAKS)
            # No peaks detected, non-periodic behavior.
            if peaks.size <= self.MIN_PEAKS:
                self._update_periods(
                    np.inf, 0,
                    fft_series,
                    sample_freq,
                    0,
                    y_tail_var=y_tail_var,
                )
                continue
            period = peaks[-1] - peaks[-2]

            # print(period)

            # print(period / self.Fs)
            # Calculate the autocorrelation coefficient to identify the maximum value, 
            # which represents the desired period. 
            # A higher autocorrelation coefficient indicates a stronger periodic relationship between successive images.
            period, score = search_by_acf(data, period=period, ode_model=self._Parameters.ode_model_name)

            if self._Parameters.ode_model_name == 'brusselator' or self._Parameters.ode_model_name == 'PS2_01':
                y_ratio = self._cal_y_multi_ratio(data, period)
            else:
                y_ratio = self._cal_y_ratio(data, period)

            self._update_periods(
                period / self.Fs,
                score,
                fft_series,
                sample_freq,
                y_ratio,
                y_tail_var=y_tail_var,
            )

        self._judge()
        return self

    def _judge(self):
        """
        After a successful calculation that yields a non-periodic result, 
        determine the type of solution based on the following rules:
            If all lines exhibit oscillation, classify the solution as oscillatory.
            If any line shows divergence, classify the solution as divergent.
            Otherwise, classify the solution as convergent.
        """
        not_osci = False
        not_conv = False
        for item in self.periods:
            is_inf = np.isinf(item[0])
            # print(abs(item[4] - 1))
            if not is_inf and item[1] >= self.threshold and abs(item[4] - 1) <= self.Y_RATIO_DELTA:
                curve_pde_type = PDEType.OSCILLATION
            else:
                not_osci = True
                # divergent
                if item[4] < 1 - self.Y_TAIL_DELTA and item[5] > self.Y_TAIL_DELTA:
                    not_conv = True
                    curve_pde_type = PDEType.INF
                # divergent
                elif is_inf and item[5] > self.Y_TAIL_DELTA:
                    not_conv = True
                    curve_pde_type = PDEType.INF
                else:
                    curve_pde_type = PDEType.CONVERGENCE
            self.curve_pde_types.append(curve_pde_type)

        if not not_osci:
            self.pde_type = PDEType.OSCILLATION
        elif not not_conv:
            self.pde_type = PDEType.CONVERGENCE
        else:
            self.pde_type = PDEType.INF

    @classmethod
    def _cal_y_ratio(cls, y, period: int):
        """ y_ratio """
        tmp = y - np.mean(y)
        head = np.linalg.norm(tmp[:cls.Y_RATIO_PERIOD_N * period])
        tail = np.linalg.norm(tmp[-cls.Y_RATIO_PERIOD_N * period:])
        # print(period, head, tail, len(y))
        return head / tail if tail else np.inf

    @classmethod
    def _cal_y_multi_ratio(cls, y, period: int):
        """ y_ratio """
        Min_y_ratio = None
        Max_y_ratio = None
        for Y_ratio_PERIOD in cls.Y_RATIO_PERIOD_N_list:
            tmp = y - np.mean(y)
            head = np.linalg.norm(tmp[:Y_ratio_PERIOD * period])
            tail = np.linalg.norm(tmp[-Y_ratio_PERIOD * period:])


            if tail:
                if Min_y_ratio is None:
                    Min_y_ratio = head / tail
                    Max_y_ratio = head / tail
                else:
                    Min_y_ratio = min(head / tail, Min_y_ratio)
                    Max_y_ratio = max(head / tail, Max_y_ratio)
            else:
                return np.inf

            # print(head / tail)
        if Max_y_ratio - Min_y_ratio > cls.Y_RATIO_DELTA:
            return np.inf
        return Min_y_ratio
        # print(period, head, tail, len(y))
        # return head / tail if tail else np.inf



# noinspection PyTypeChecker
def transform(line: str, xs_selected: List[int], ys_selected: List[int], ys_weight: List[float] = None
              , xs_weight: List[List[float]]=None, model_name: str='HSECC', **kwargs):
    # process data
    line = line.strip()
    # print(line)
    if not line:
        return None, None

    ok = True
    items = line.split(',')

    if model_name == 'HSECC':
        # param begin with 13
        CONV_indices = (1, 5, 9)
        Y_Ratio_indices = (3, 7, 11)
        Cycle_time_indices = (4, 8, 12)
    elif model_name == 'PS2_01':
        # param begin with 25
        CONV_indices = (1, 5)
        Y_Ratio_indices = (3, 7)
        Cycle_time_indices = (4, 8)
    elif model_name == 'brusselator':
        # param begin with 25
        CONV_indices = (1, 5)
        Y_Ratio_indices = (3, 7)
        Cycle_time_indices = (4, 8)
    elif model_name == 'MPF_2_Var':
        # param begin with 25
        CONV_indices = (1, 5)
        Y_Ratio_indices = (3, 7)
        Cycle_time_indices = (4, 8)

    if items[0][1] == '_':
        # name
        items[0] = int(items[0].split('_')[1])
        # CONV
        for idx in CONV_indices:
            if items[idx] == 'OSCI':
                items[idx] = 0
            elif items[idx] == 'CONV':
                items[idx] = 1
                ok = False
            elif items[idx] == 'INF':
                items[idx] = 2
                ok = False
        # Y_Ratio
        for idx in Y_Ratio_indices:
            if items[idx].find('>') >= 0:
                items[idx] = 20
        # Cycle time
        for idx in Cycle_time_indices:
            if not ok or items[idx].find('inf') >= 0:
                items[idx] = 0
            else:
                items[idx] = 1 / float(items[idx])
                # items[idx] = 1
    # print(items)
    # select input, output
    xs = []
    tmp = 0
    # print(xs_selected)
    # print(items, line)
    for idx in xs_selected:
        if xs_weight:
            xs.append((float(items[idx]) - xs_weight[tmp][0]) / (xs_weight[tmp][1] - xs_weight[tmp][0]))
        else:
            xs.append(float(items[idx]))
        tmp += 1
    
    # print(items)

    ys = []
    tmp = 0
    for idx in ys_selected:
        # print(ys_weight)
        if ys_weight is not None:
            ys.append(float(items[idx]) / ys_weight[tmp])
        else:
            ys.append(float(items[idx]))
        tmp += 1

    return xs, ys
