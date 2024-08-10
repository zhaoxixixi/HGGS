import sys
sys.path.extend(['../../../../src'])    # noqa
from typing import List
from pydantic import validator
from yu.const.grid import SearchType
from yu.core.config import BaseConfig, auto_config
from yu.tasks.pde_models import Truth
import os
import time
import numpy as np

from yu.const.normal import PDEType, RandomType, DisplayType
from yu.core.logger import logger
from yu.tools.grid import Grid
from yu.tools.misc import sample_lhs, makedir
from yu.tools.plot import save_plot, show_plot

from yu.tasks.ode_model import Normal_Equation, Normal_Parameters


class GeneratorConfig(BaseConfig):
    total_good: int
    total_bad: int
    n_good: int = 0
    n_bad: int = 0

    keep_single: bool
    display: DisplayType = DisplayType.NONE

    random_type: RandomType
    threshold: List[float]
    params_selected: List[int]

    # RandomType.LHS
    dimension: int
    log_lb: float
    log_ub: float
    use_log: bool
    per_split_n: List[int]
    max_split_times: int = 0
    init_split_n: int
    cube_n: List[float]
    search_type: SearchType

    # RandomType.RANDOM
    r_rate: List[List[float]] = None

    good_dir: str = 'good'
    bad_dir: str = 'bad'
    good_f: str = 'good.csv'
    bad_f: str = 'bad.csv'

    db_path: str = 'grid'
    tablename: str = 'generate_data'

    @validator('good_dir', pre=True)
    def set_good_dir(cls, v, values):
        root_dir = values.get('save_dir')
        good_dir = os.path.join(root_dir, v)
        makedir(good_dir)
        return good_dir

    @validator('bad_dir', pre=True)
    def set_bad_dir(cls, v, values):
        root_dir = values.get('save_dir')
        bad_dir = os.path.join(root_dir, v)
        makedir(bad_dir)
        return bad_dir

    @validator('good_f', pre=True)
    def set_good_f(cls, v, values):
        root_dir = values.get('save_dir')
        return os.path.join(root_dir, v)

    @validator('bad_f', pre=True)
    def set_bad_f(cls, v, values):
        root_dir = values.get('save_dir')
        return os.path.join(root_dir, v)

    @validator('db_path', pre=True)
    def set_db_path(cls, v, values):
        root_dir = values.get('save_dir')
        return os.path.join(root_dir, v)

    @validator('max_split_times', pre=True)
    def set_max_split_times(cls, v, values):
        return len(values.get('per_split_n'))


def callback(fig, axis, xs, ys):
    max_y = ys[0][0]
    for y in ys:
        tmp = int(np.max(y[1:]) * 1.2)
        if max_y < tmp:
            max_y = tmp
    if not max_y:
        max_y = 1
    axis.set_ylim(-0.01 * max_y, max_y)
    axis.set_xlim(-xs[0][1], xs[0][-1] / 4)

def save_img(
        truth: Truth,
        model_name: str,
        png_name: str,
        keep_single: bool = False,
        display: DisplayType = DisplayType.NONE,
):
    """ save img """
    curve_size = len(truth.equation_idx)

    # ODE
    ys = np.zeros((2, curve_size), dtype=np.ndarray)
    xs = np.zeros((2, curve_size), dtype=np.ndarray)
    labels = np.full((2, curve_size), '', dtype=object)
    titles = np.asarray([f'Truth of {truth.model_name} Model({model_name})'])

    x_labels = np.asarray(['t', 'frequency'])
    y_labels = np.asarray(['function', 'FFT / t'])

    truth_plot = np.swapaxes(truth.truth, 0, 1)
    curve_idx = 0
    for j, line in enumerate(truth_plot):
        if j not in truth.equation_idx:
            continue

        if truth.pde_type == PDEType.OSCILLATION:
            y_ratio = '> 10' if truth.periods[curve_idx][4] > 10 else f'{truth.periods[curve_idx][4]:.2f}'
            curve_name = f'{truth.curve_names[curve_idx]}(T: {truth.periods[curve_idx][0]:.4f},' \
                         f' ACF: {truth.periods[curve_idx][1]:.4f}, y_ratio: {y_ratio})'
        else:
            curve_name = f'{truth.curve_names[curve_idx]}' \
                         f'(ACF: {truth.periods[curve_idx][1]:.4f}, y_delta: {truth.periods[curve_idx][5]:.4f})'
        labels[0][curve_idx] = curve_name
        ys[0][curve_idx] = line
        xs[0][curve_idx] = truth.t

        labels[1][curve_idx] = f'{truth.curve_names[curve_idx]}'
        ys[1][curve_idx] = truth.periods[curve_idx][2]
        xs[1][curve_idx] = truth.periods[curve_idx][3]

        curve_idx += 1

    if display == DisplayType.SHOW:
        show_plot(
            ys=ys,
            xs=xs,
            labels=labels,
            titles=titles,
            x_labels=x_labels,
            y_labels=y_labels,
            callbacks=[None, callback],
            colors=truth.curve_colors,
        )
    elif display == DisplayType.SAVE:
        save_plot(
            ys=ys,
            xs=xs,
            save_name=png_name,
            labels=labels,
            titles=titles,
            x_labels=x_labels,
            y_labels=y_labels,
            callbacks=[None, callback],
            colors=truth.curve_colors,
        )
    if keep_single:
        for curve_idx in range(curve_size):
            singe_name = os.path.join(
                os.path.dirname(png_name),
                f'{truth.curve_names[curve_idx]}.png',
            )
            if display == DisplayType.SHOW:
                show_plot(
                    ys=ys[..., curve_idx].reshape(2, 1),
                    xs=xs[..., curve_idx].reshape(2, 1),
                    labels=labels[..., curve_idx].reshape(2, 1),
                    titles=titles,
                    x_labels=x_labels,
                    y_labels=y_labels,
                    callbacks=[None, callback],
                    colors=truth.curve_colors[curve_idx:curve_idx + 1],
                )
            elif display == DisplayType.SAVE:
                save_plot(
                    ys=ys[..., curve_idx].reshape(2, 1),
                    xs=xs[..., curve_idx].reshape(2, 1),
                    save_name=singe_name,
                    # labels=labels[..., curve_idx].reshape(2, 1),
                    labels = None,
                    titles=titles,
                    x_labels=x_labels,
                    y_labels=y_labels,
                    callbacks=[None, callback],
                    colors=truth.curve_colors[curve_idx:curve_idx + 1],
                )


def score_calculator(
        coords, edge, depth,
        config: GeneratorConfig,
        truth: Truth,
        good_f, bad_f,
        t1: float,
        _Parameters: Normal_Parameters,
        _y0: List[float], _T: float, _T_unit: float,
        **kwargs
):
    current_n_good = 0
    current_n_bad = 0
    scale = (config.log_ub - config.log_lb) / edge

    cube_n = int(max(config.cube_n[depth - 1], 1))
    samples = sample_lhs(lb=0, ub=scale, d=len(config.params_selected), n=cube_n)
    # trans position
    samples += np.full((len(samples), len(config.params_selected)), coords * scale + config.log_lb)
    if config.use_log:
        samples = 10 ** samples
    for sample in samples:
        if config.random_type == RandomType.NO:
            params = _Parameters()
        elif config.random_type == RandomType.RANDOM:
            params = _Parameters()
            params.random(config.r_rate)
        else:
            coefficients = []
            k = 0
            for i in range(config.dimension):
                if i in config.params_selected:
                    coefficients.append(sample[k])
                    k += 1
                else:
                    coefficients.append(1)
            params = _Parameters.of_coefficients(coefficients)

        truth.calc(
            y0=np.asarray(_y0),
            T=_T,
            T_unit=_T_unit,
            params=params
        ).find_period()

        # if not success will destroy it
        if not truth.success and truth.pde_type == PDEType.NAN:
            continue

        if truth.pde_type == PDEType.OSCILLATION:
            model_name = f'g_{config.n_good}'
            png_name = os.path.join(config.good_dir, model_name, 'model.png')
            config.n_good += 1
            current_n_good += 1
        else:
            model_name = f'b_{config.n_bad}'
            png_name = os.path.join(config.bad_dir, model_name, 'model_name.png')
            config.n_bad += 1
            current_n_bad += 1

        # 保存数据信息
        csv_line = params.build_csv(
            truth=truth,
            params=params,
            name=model_name,
        )

        if truth.pde_type == PDEType.OSCILLATION:
            good_f.write(f'{csv_line}\n')
        else:
            bad_f.write(f'{csv_line}\n')

        # only success
        if truth.success and config.display != DisplayType.NONE:
            save_img(
                truth=truth,
                model_name=model_name,
                png_name=png_name,
                keep_single=config.keep_single,
                display=config.display,
            )

        total = config.n_good + config.n_bad
        if total % 500 == 0:
            t2 = time.time()
            logger.info(
                f'total: {total}/{config.total_good + config.total_bad},'
                f' n_good: {config.n_good}/{config.total_good}, n_bad: {config.n_bad}/{config.total_good},'
                f' running time: {t2 - t1}s'
            )

    current_total = current_n_good + current_n_bad
    return current_n_good / current_total if current_total else 0


# HSECC
from yu.tasks.model_HSECC import Parameters as HSECC_Params
from yu.tasks.model_HSECC import Equation as HSECC_Equs
@auto_config(GeneratorConfig, __file__)
def main(config: GeneratorConfig, truth: Truth, **kwargs):
    """"""

    root_dir = config.save_dir

    good_f = open(config.good_f, 'a+')
    bad_f = open(config.bad_f, 'a+')

    t1 = time.time()
    grid = Grid(
        max_split_times=config.max_split_times,
        per_split_n=config.per_split_n,
        init_split_n=config.init_split_n,
        dimensions=len(config.params_selected),
        threshold=config.threshold,
        save_dir=root_dir,
        search_type=config.search_type,
        db_path=config.db_path,
        table=config.tablename,
    )
    if os.path.exists(grid.config_file):
        grid.reload(root_dir, grid)
    grid.execute(
        score_calculator=score_calculator,
        config=config,
        truth=truth,
        good_f=good_f,
        bad_f=bad_f,
        t1=time.time(),
        _Parameters=truth._Parameters,
        _y0=_y0, _T=_T, _T_unit=_T_unit,
    )
    t2 = time.time()
    logger.info(
        f'total: {config.n_good + config.n_bad}/{config.total_good + config.total_bad},'
        f' n_good: {config.n_good}/{config.total_good}, n_bad: {config.n_bad}/{config.total_good},'
        f' running time: {t2 - t1}s'
    )
    good_f.close()
    bad_f.close()


if __name__ == '__main__':
    model_name = 'HSECC'

    if model_name == 'HSECC':
        Parameters = HSECC_Params
        Equation = HSECC_Equs

        model_name = model_name + ' Cell Cycle'
        equation_indices = [1, 3, 4]
        curve_names = ['X', 'Y', 'Z']
        curve_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        _y0 = [30, 320, 100, 100, 200]
        _T = 1000
        _T_unit = 1
    truth = Truth()
    truth.config(
        model_name=model_name,
        curve_names=curve_names,
        curve_colors=curve_colors,
        equation_idx=equation_indices,
        threshold=0.40,
        equs=Equation,
        _Parameters=Parameters,
    )
    main(truth=truth, _y0=_y0, _T=_T, _T_unit=_T_unit)
    exit(0)
