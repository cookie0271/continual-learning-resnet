#!/usr/bin/env python3
import os
import numpy as np
from matplotlib.pyplot import get_cmap
# -custom-written code
import main
import utils
from params.param_stamp import get_param_stamp_from_args
from params.param_values import check_for_errors, set_default_values
from params import options
from visual import visual_plt as my_plt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

## Parameter-values to compare
lamda_list = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000., 100000000., 1000000000., 10000000000.,
              100000000000., 1000000000000., 10000000000000.]
lamda_list_permMNIST = [1., 10., 100., 1000., 10000., 100000., 1000000., 10000000.]
c_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 5., 10., 50., 100., 500., 1000., 5000., 10000., 50000., 100000.]
c_list_permMNIST = [0.01, 0.1, 1., 10., 100., 1000., 10000., 100000.]
xdg_list = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
xdg_list_permMNIST = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dg_prop_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
tau_list = [0.001, 0.01, 0.1, 1., 10., 100., 1000., 10000., 100000.]
budget_list_splitMNIST = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
budget_list_splitCIFAR100 = [1, 2, 5, 10, 20]
lambda_0_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 可根据需要调整
# lambda_0_list = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
scaling_power_list = [0.1, 0.2, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 新增 scaling_power 搜索范围


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'comparison': True, 'compare_hyper': True}
    # Define input options
    parser = options.define_args(filename="compare_hyperParams", description='Hyperparamer gridsearches.')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_cl_options(parser, **kwargs)
    # Should the gridsearch not be run for some methods?
    parser.add_argument('--no_xdg', action='store_true', help="no XdG")
    parser.add_argument('--no_reg', action='store_true', help="no EWC or SI")
    parser.add_argument('--no_fromp', action='store_true', help="no FROMP")
    parser.add_argument('--no_bir', action='store_true', help="no BI-R")
    parser.add_argument('--no_ada', action='store_true', help="no adaptive")
    # use adaptive regularization
    parser.add_argument("--lambda_0", type=float, default=0.3, help="Initial lambda value")
    parser.add_argument('--use_adaptive', action='store_true', default=False,
                        help='Enable adaptive regularization on top of basic regularization')
    # Add scaling_power as an optional argument
    parser.add_argument('--scaling_power', type=float, default=0.33,
                        help='Scaling power for adaptive regularization (default: 0.33)')
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    args.log_per_context = True
    set_default_values(args, also_hyper_params=False)  # -set defaults, some are based on chosen scenario / experiment
    check_for_errors(args, **kwargs)  # -check whether incompatible options are selected
    return args


## Function for running experiments and collecting results
def get_result(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile('{}/acc-{}.txt'.format(args.r_dir, param_stamp)):
        print(" already run: {}".format(param_stamp))
    else:
        args.train = True
        print("\n ...running: {} ...".format(param_stamp))
        main.run(args)
    # -get average accuracy
    fileName = '{}/acc-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -return it
    return ave


if __name__ == '__main__':

    ## Load input-arguments
    args = handle_inputs()

    # Create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    ## Select parameter-lists based on chosen experiment
    xdg_list = xdg_list_permMNIST if args.experiment == "permMNIST" else xdg_list
    lamda_list = lamda_list_permMNIST if args.experiment == "permMNIST" else lamda_list
    c_list = c_list_permMNIST if args.experiment == "permMNIST" else c_list
    budget_list = budget_list_splitMNIST if args.experiment == "splitMNIST" else budget_list_splitCIFAR100

    # -------------------------------------------------------------------------------------------------#

    # --------------------------#
    # ----- RUN ALL MODELS -----#
    # --------------------------#

    ## Baselline
    args.replay = "none"
    BASE = get_result(args)

    ## XdG
    if args.scenario == "task" and not utils.checkattr(args, 'no_xdg'):
        XDG = {}
        always_xdg = utils.checkattr(args, 'xdg')
        if always_xdg:
            gating_prop_selected = args.gating_prop
        args.xdg = True
        for xdg in xdg_list:
            args.gating_prop = xdg
            XDG[xdg] = get_result(args)
        args.xdg = always_xdg
        if always_xdg:
            args.gating_prop = gating_prop_selected

    ## EWC
    if not utils.checkattr(args, 'no_reg'):
        EWC = {}
        args.weight_penalty = True
        args.offline = True
        args.importance_weighting = 'fisher'
        for ewc_lambda in lamda_list:
            args.reg_strength = ewc_lambda
            EWC[ewc_lambda] = get_result(args)
        args.weight_penalty = False
        args.offline = False

    '''
     # ======== 🚀 EWC + Adaptive (坐标上升法) ========
    if not utils.checkattr(args, 'no_ada'):
        print("\n\n" + "="*80)
        print(">>> 开始为 EWC + Adaptive 执行坐标上升搜索...")
        print("="*80)

        args.weight_penalty = True
        args.offline = True
        args.importance_weighting = 'fisher'
        args.use_adaptive = True

        # --- 步骤 1: 固定 scaling_power，寻找最优的 lambda ---
        print("\n[步骤 1/2] 正在搜索最优 lambda...")
        ewc_adaptive_step1_results = {}
        fixed_scaling_power_ewc = 0.7
        print(f"--- (固定 scaling_power = {fixed_scaling_power_ewc}) ---")
        args.scaling_power = fixed_scaling_power_ewc
        for ewc_lambda in lambda_0_list:
            args.reg_strength = ewc_lambda
            args.lambda_0 = ewc_lambda # 对于EWC, lambda_0 和 reg_strength 是一致的
            ewc_adaptive_step1_results[ewc_lambda] = get_result(args)
        best_ewc_lambda = max(ewc_adaptive_step1_results, key=ewc_adaptive_step1_results.get)
        print(f"---> 找到最优 lambda = {best_ewc_lambda} (准确率: {ewc_adaptive_step1_results[best_ewc_lambda]:.4f})")

        # --- 步骤 2: 固定最优的 lambda, 寻找最优的 scaling_power ---
        print("\n[步骤 2/2] 正在搜索最优 scaling_power...")
        ewc_adaptive_step2_results = {}
        print(f"--- (固定 lambda = {best_ewc_lambda}) ---")
        args.reg_strength = best_ewc_lambda
        args.lambda_0 = best_ewc_lambda
        for scaling_power in scaling_power_list:
            args.scaling_power = scaling_power
            ewc_adaptive_step2_results[scaling_power] = get_result(args)
        best_scaling_power_ewc = max(ewc_adaptive_step2_results, key=ewc_adaptive_step2_results.get)
        final_best_acc_ewc = ewc_adaptive_step2_results[best_scaling_power_ewc]
        print(f"---> 找到最优 scaling_power = {best_scaling_power_ewc} (准确率: {final_best_acc_ewc:.4f})")

        print("\n" + "="*80)
        print(">>> EWC + Adaptive 坐标上升搜索完成.")
        print(f"最终最优参数: lambda = {best_ewc_lambda}, scaling_power = {best_scaling_power_ewc}")
        print(f"最终最高准确率: {final_best_acc_ewc:.4f}")
        print("="*80 + "\n")

        args.weight_penalty = False
        args.use_adaptive = False
        args.offline = False
    '''

    ## SI
    if not utils.checkattr(args, 'no_reg'):
        SI = {}
        args.weight_penalty = True
        args.importance_weighting = 'si'
        for si_c in c_list:
            args.reg_strength = si_c
            SI[si_c] = get_result(args)
        args.weight_penalty = False

    # ======== 🚀 MAS + Adaptive (坐标上升法) ========
    if not utils.checkattr(args, 'no_ada'):
        print("\n\n" + "=" * 80)
        print(">>> 开始为 MAS + Adaptive 执行坐标上升搜索...")
        print("=" * 80)

        args.weight_penalty = True
        args.importance_weighting = 'mas'
        args.use_adaptive = True

        # --- 步骤 1: 固定 scaling_power，寻找最优的 lambda_0 ---
        print("\n[步骤 1/2] 正在搜索最优 lambda_0...")
        mas_adaptive_step1_results = {}
        fixed_scaling_power_mas = 0.7
        print(f"--- (固定 scaling_power = {fixed_scaling_power_mas}) ---")
        args.scaling_power = fixed_scaling_power_mas
        for lambda_0 in lambda_0_list:
            args.lambda_0 = lambda_0
            args.reg_strength = lambda_0
            mas_adaptive_step1_results[lambda_0] = get_result(args)
        best_lambda_0_mas = max(mas_adaptive_step1_results, key=mas_adaptive_step1_results.get)
        print(
            f"---> 找到最优 lambda_0 = {best_lambda_0_mas} (准确率: {mas_adaptive_step1_results[best_lambda_0_mas]:.4f})")

        # --- 步骤 2: 固定最优的 lambda_0, 寻找最优的 scaling_power ---
        print("\n[步骤 2/2] 正在搜索最优 scaling_power...")
        mas_adaptive_step2_results = {}
        print(f"--- (固定 lambda_0 = {best_lambda_0_mas}) ---")
        args.lambda_0 = best_lambda_0_mas
        args.reg_strength = best_lambda_0_mas
        for scaling_power in scaling_power_list:
            args.scaling_power = scaling_power
            mas_adaptive_step2_results[scaling_power] = get_result(args)
        best_scaling_power_mas = max(mas_adaptive_step2_results, key=mas_adaptive_step2_results.get)
        final_best_acc_mas = mas_adaptive_step2_results[best_scaling_power_mas]
        print(f"---> 找到最优 scaling_power = {best_scaling_power_mas} (准确率: {final_best_acc_mas:.4f})")

        print("\n" + "=" * 80)
        print(">>> MAS + Adaptive 坐标上升搜索完成.")
        print(f"最终最优参数: lambda_0 = {best_lambda_0_mas}, scaling_power = {best_scaling_power_mas}")
        print(f"最终最高准确率: {final_best_acc_mas:.4f}")
        print("=" * 80 + "\n")

        args.weight_penalty = False
        args.use_adaptive = False


    ## FROMP
    if not utils.checkattr(args, 'no_fromp'):
        FROMP = {}
        args.fromp = True
        args.sample_selection = 'fromp'
        for budget in budget_list:
            args.budget = budget
            FROMP[budget] = {}
            for tau in tau_list:
                args.tau = tau
                FROMP[budget][tau] = get_result(args)
        args.fromp = False

    ## BI-R
    if not utils.checkattr(args, 'no_bir'):
        BIR = {}
        args.replay = "generative"
        args.feedback = True
        args.hidden = True
        args.dg_gates = True
        args.prior = "GMM"
        args.per_class = True
        args.distill = True
        for dg_prop in dg_prop_list:
            args.dg_prop = dg_prop
            BIR[dg_prop] = get_result(args)

    # -------------------------------------------------------------------------------------------------#

    # -----------------------------------------#
    # ----- COLLECT DATA & PRINT ON SCREEN-----#
    # -----------------------------------------#

    ext_c_list = [0] + c_list
    ext_lambda_list = [0] + lamda_list
    ext_tau_list = [0] + tau_list
    print("\n")

    ###---XdG---###
    if args.scenario == "task" and not utils.checkattr(args, 'no_xdg'):
        ave_acc_xdg = [XDG[c] for c in xdg_list]
        print("\n\nCONTEXT-DEPENDENT GATING (XDG))")
        print(" param list (gating_prop): {}".format(xdg_list))
        print("  {}".format(ave_acc_xdg))
        print("---> gating_prop = {}     --    {}".format(xdg_list[np.argmax(ave_acc_xdg)], np.max(ave_acc_xdg)))

    ###---EWC---###
    if not utils.checkattr(args, 'no_reg'):
        ave_acc_ewc = [BASE] + [EWC[ewc_lambda] for ewc_lambda in lamda_list]
        print("\n\nELASTIC WEIGHT CONSOLIDATION (EWC)")
        print(" param-list (lambda): {}".format(ext_lambda_list))
        print("  {}".format(ave_acc_ewc))
        print("--->  lambda = {}     --    {}".format(ext_lambda_list[np.argmax(ave_acc_ewc)], np.max(ave_acc_ewc)))

    '''
    # ======== 🚀 EWC + Adaptive (打印结果) ========
    if not utils.checkattr(args, 'no_ada'):
        print("\n\nEWC + ADAPTIVE (步骤 1: 寻找 lambda)")
        print(f" (固定 scaling_power = {fixed_scaling_power_ewc})")
        print(" param-list (lambda): {}".format(list(ewc_adaptive_step1_results.keys())))
        print("  {}".format(list(ewc_adaptive_step1_results.values())))
        print("---> best lambda = {}  --  {}".format(best_ewc_lambda, ewc_adaptive_step1_results[best_ewc_lambda]))

        print("\n\nEWC + ADAPTIVE (步骤 2: 寻找 scaling_power)")
        print(f" (固定 lambda = {best_ewc_lambda})")
        print(" param-list (scaling_power): {}".format(list(ewc_adaptive_step2_results.keys())))
        print("  {}".format(list(ewc_adaptive_step2_results.values())))
        print("---> best scaling_power = {}  --  {}".format(best_scaling_power_ewc, final_best_acc_ewc))
    '''

    ###---SI---###
    if not utils.checkattr(args, 'no_reg'):
        ave_acc_si = [BASE] + [SI[c] for c in c_list]
        print("\n\nSYNAPTIC INTELLIGENCE (SI)")
        print(" param list (si_c): {}".format(ext_c_list))
        print("  {}".format(ave_acc_si))
        print("---> si_c = {}     --    {}".format(ext_c_list[np.argmax(ave_acc_si)], np.max(ave_acc_si)))

    # ======== 🚀 MAS + Adaptive (打印结果) ========
    if not utils.checkattr(args, 'no_ada'):
        print("\n\nSYNAPTIC INTELLIGENCE + ADAPTIVE (步骤 1: 寻找 lambda_0)")
        print(f" (固定 scaling_power = {fixed_scaling_power_mas})")
        print(" param-list (lambda_0): {}".format(list(mas_adaptive_step1_results.keys())))
        print("  {}".format(list(mas_adaptive_step1_results.values())))
        print(
            "---> best lambda_0 = {}  --  {}".format(best_lambda_0_mas, mas_adaptive_step1_results[best_lambda_0_mas]))

        print("\n\nSYNAPTIC INTELLIGENCE + ADAPTIVE (步骤 2: 寻找 scaling_power)")
        print(f" (固定 lambda_0 = {best_lambda_0_mas})")
        print(" param-list (scaling_power): {}".format(list(mas_adaptive_step2_results.keys())))
        print("  {}".format(list(mas_adaptive_step2_results.values())))
        print("---> best scaling_power = {}  --  {}".format(best_scaling_power_mas, final_best_acc_mas))
    ###---FROMP---###
    if not utils.checkattr(args, 'no_fromp'):
        ave_acc_fromp_per_budget = []
        for budget in budget_list:
            ave_acc_fromp = [FROMP[budget][tau] for tau in tau_list]
            ave_acc_fromp_ext = [BASE] + [FROMP[budget][tau] for tau in tau_list]
            print("\n\nFROMP (budget={})".format(budget))
            print(" param-list (tau): {}".format(ext_tau_list))
            print("  {}".format(ave_acc_fromp_ext))
            print("--->  tau = {}     --    {}".format(ext_tau_list[np.argmax(ave_acc_fromp_ext)],
                                                       np.max(ave_acc_fromp_ext)))
            ave_acc_fromp_per_budget.append(ave_acc_fromp)

    ###---BI-R---###
    if not utils.checkattr(args, 'no_bir'):
        ave_acc_bir = [BIR[dg_prop] for dg_prop in dg_prop_list]
        print("\n\nBRAIN-INSPIRED REPLAY (BI-R)")
        print(" param list (dg_prop): {}".format(dg_prop_list))
        print("  {}".format(ave_acc_bir))
        print("---> dg_prop = {}     --    {}".format(dg_prop_list[np.argmax(ave_acc_bir)], np.max(ave_acc_bir)))
        print('\n')

    # -------------------------------------------------------------------------------------------------#

    # --------------------#
    # ----- PLOTTING -----#
    # --------------------#

    # name for plot
    plot_name = "hyperParams-{}{}-{}".format(args.experiment, args.contexts, args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    ylabel = "Test accuracy (after all contexts)"

    # calculate limits y-axes (to have equal axes for all graphs)
    full_list = []
    if not utils.checkattr(args, 'no_reg'):
        full_list += ave_acc_ewc + ave_acc_si
    if not utils.checkattr(args, 'no_fromp'):
        for item in ave_acc_fromp_per_budget:
            full_list += item
    if not utils.checkattr(args, 'no_bir'):
        full_list += ave_acc_bir
    if args.scenario == "task" and not utils.checkattr(args, 'no_xdg'):
        full_list += ave_acc_xdg
    if not utils.checkattr(args, 'no_ada'):
        # full_list += list(ewc_adaptive_step1_results.values())
        # full_list += list(ewc_adaptive_step2_results.values())
        full_list += list(mas_adaptive_step1_results.values())
        full_list += list(mas_adaptive_step2_results.values())
    miny = np.min(full_list)
    maxy = np.max(full_list)
    marginy = 0.1 * (maxy - miny)
    ylim = (np.max([miny - 2 * marginy, 0]),
            np.min([maxy + marginy, 1])) if not args.scenario == "class" else (0, np.min([maxy + marginy, 1]))

    # open pdf
    pp = my_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    figure_list = []

    ###---XdG---###
    if args.scenario == "task" and not utils.checkattr(args, 'no_xdg'):
        figure = my_plt.plot_lines([ave_acc_xdg], x_axes=xdg_list, ylabel=ylabel,
                                   line_names=["XdG"], colors=["deepskyblue"], ylim=ylim,
                                   title=title, x_log=False, xlabel="XdG: % of nodes gated",
                                   with_dots=True, h_line=BASE, h_label="None")
        figure_list.append(figure)

    ###---EWC---###
    if not utils.checkattr(args, 'no_reg'):
        figure = my_plt.plot_lines([ave_acc_ewc[1:]], x_axes=lamda_list, ylabel=ylabel, line_names=["EWC"],
                                   colors=["darkgreen"], title=title, x_log=True, xlabel="EWC: lambda (log-scale)",
                                   with_dots=True, ylim=ylim, h_line=BASE, h_label="None")
        figure_list.append(figure)

    '''
     # ======== 🚀 EWC + Adaptive (绘图) ========
    if not utils.checkattr(args, 'no_ada'):
        # 图1: 寻找 lambda
        figure = my_plt.plot_lines([list(ewc_adaptive_step1_results.values())], x_axes=list(ewc_adaptive_step1_results.keys()),
                                   ylabel=ylabel, line_names=[f"EWC+Adaptive (sp={fixed_scaling_power_ewc})"],
                                   colors=["darkcyan"], title=title, x_log=True, xlabel="EWC+Adaptive: lambda (log-scale)",
                                   with_dots=True, ylim=ylim, h_line=BASE, h_label="None")
        figure_list.append(figure)

        # 图2: 寻找 scaling_power
        figure = my_plt.plot_lines([list(ewc_adaptive_step2_results.values())], x_axes=list(ewc_adaptive_step2_results.keys()),
                                   ylabel=ylabel, line_names=[f"EWC+Adaptive (λ={best_ewc_lambda})"],
                                   colors=["teal"], title=title, x_log=False, xlabel="EWC+Adaptive: scaling_power",
                                   with_dots=True, ylim=ylim, h_line=BASE, h_label="None")
        figure_list.append(figure)
    '''

    ###---SI---###
    if not utils.checkattr(args, 'no_reg'):
        figure = my_plt.plot_lines([ave_acc_si[1:]], x_axes=c_list, ylabel=ylabel, line_names=["SI"],
                                   colors=["yellowgreen"], title=title, x_log=True, xlabel="SI: c (log-scale)",
                                   with_dots=True, ylim=ylim, h_line=BASE, h_label="None")
        figure_list.append(figure)

    # ======== 🚀 MAS + Adaptive (绘图) ========
    if not utils.checkattr(args, 'no_ada'):
        # 图1: 寻找 lambda_0
        figure = my_plt.plot_lines([list(mas_adaptive_step1_results.values())],
                                   x_axes=list(mas_adaptive_step1_results.keys()),
                                   ylabel=ylabel, line_names=[f"MAS+Adaptive (sp={fixed_scaling_power_mas})"],
                                   colors=["yellowgreen"], title=title, x_log=False, xlabel="MAS+Adaptive: lambda_0",
                                   with_dots=True, ylim=ylim, h_line=BASE, h_label="None")
        figure_list.append(figure)

        # 图2: 寻找 scaling_power
        figure = my_plt.plot_lines([list(mas_adaptive_step2_results.values())],
                                   x_axes=list(mas_adaptive_step2_results.keys()),
                                   ylabel=ylabel, line_names=[f"MAS+Adaptive (λ₀={best_lambda_0_mas})"],
                                   colors=["gold"], title=title, x_log=False, xlabel="MAS+Adaptive: scaling_power",
                                   with_dots=True, ylim=ylim, h_line=BASE, h_label="None")
        figure_list.append(figure)

    ###---FROMP---###
    if not utils.checkattr(args, 'no_fromp'):
        colors = get_cmap('YlOrBr')(np.linspace(1.0, 0.5, len(budget_list))).tolist()
        figure = my_plt.plot_lines(ave_acc_fromp_per_budget, x_axes=tau_list, ylabel=ylabel,
                                   line_names=["FROMP (budget={})".format(budget) for budget in budget_list],
                                   colors=colors, title=title, x_log=True, xlabel="FROMP: tau (log-scale)",
                                   with_dots=True, ylim=ylim, h_line=BASE, h_label="None")
        figure_list.append(figure)

    ###---BI-R---###
    if not utils.checkattr(args, 'no_bir'):
        figure = my_plt.plot_lines([ave_acc_bir], x_axes=dg_prop_list, ylabel=ylabel, line_names=["BI-R"],
                                   colors=["lightcoral"], title=title, x_log=False, with_dots=True,
                                   xlabel="BI-R: % of nodes gated in decoder", ylim=ylim, h_line=BASE, h_label="None")
        figure_list.append(figure)

    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))