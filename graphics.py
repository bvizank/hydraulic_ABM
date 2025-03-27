from base import Graphics
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)

days = 180
plots = Graphics(
    publication=True,
    error="se",
    days=days,
    inp_file="Input Files/cities/clinton/clinton.inp",
    # scenario_ls=["base", "basebw", "pm", "pm_nobw", "sa"],
    scenario_ls=["base", "basebw", "pm", "pm_nobw"],
    skeletonized=True,
    single=False,
    remove_bg=False,
)

""" Demand plots """
# plots.demand_plots()

""" Flow plots """
# plots.flow_plots()

""" Age plots """
# plots.age_plots(map=True, threshold=True)

""" Industrial distance plots """
# plots.ind_dist_plots()

""" State variable comparison plots """
# plots.sv_comp_plots()

""" BBN decision plots """
# plots.bbn_plots()

""" Equity plots showing the burden of paying for water """
# plots.make_equity_plots()

""" Cost plots """
# plots.make_cost_plots(map=True)

""" %HI (cowpi) plots """
# plots.cowpi_barchart()
plots.cowpi_boxplot(demographics=False, di=False, perc=False, sa=False, map=True)

""" Block group map of city """
# plots.make_city_map()

""" Income plots """
# plots.income_plots()

""" Tap water avoidance plots """
# plots.make_twa_plots()

""" Make SA plots """
# plots.sa_plots(age=False, cost=False, cowpi=True, map=False)

""" SEIR plot """
# plots.make_seir_plot(days)

""" Make single plots """
# plots.make_single_plots("2025-03-07_15-50_0_results", 180, True)

# ''' Export comparison stats '''
# print("WFH model stats: " + str(plots.calc_model_stats(wn, only_wfh['avg_seir_data'], only_wfh['avg_age']/3600)))
# print("Dine model stats: " + str(calc_model_stats(wn, dine['avg_seir_data'], dine['avg_age']/3600)))
# print("Grocery model stats: " + str(calc_model_stats(wn, grocery['avg_seir_data'], grocery['avg_age']/3600)))
# print("PPE model stats: " + str(calc_model_stats(wn, ppe['avg_seir_data'], ppe['avg_age']/3600)))
# print("All PM model stats: " + str(calc_model_stats(wn, wfh['avg_seir_data'], wfh['avg_age']/3600)))
# print("No PM model stats: " + str(calc_model_stats(wn, no_wfh['avg_seir_data'], no_wfh['avg_age']/3600)))
