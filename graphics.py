from base import Graphics


plots = Graphics(publication=False, error='se')

''' Demand plots '''
# plots.demand_plots()

''' Flow plots '''
# plots.flow_plots()

''' Age plots '''
# plots.age_plots()

''' Industrial distance plots '''
# plots.ind_dist_plots()

''' State variable comparison plots '''
# plots.sv_comp_plots()

''' BBN decision plots '''
# plots.bbn_plots()

''' SEIR plot '''
# plots.make_seir_plot()

''' Make single plots '''
plots.make_single_plots('2024-03-12_17-58_0_results', 100)
plots.make_single_plots('2024-03-12_17-31_0_results', 100)

# ''' Export comparison stats '''
# print("WFH model stats: " + str(plots.calc_model_stats(wn, only_wfh['avg_seir_data'], only_wfh['avg_age']/3600)))
# print("Dine model stats: " + str(calc_model_stats(wn, dine['avg_seir_data'], dine['avg_age']/3600)))
# print("Grocery model stats: " + str(calc_model_stats(wn, grocery['avg_seir_data'], grocery['avg_age']/3600)))
# print("PPE model stats: " + str(calc_model_stats(wn, ppe['avg_seir_data'], ppe['avg_age']/3600)))
# print("All PM model stats: " + str(calc_model_stats(wn, wfh['avg_seir_data'], wfh['avg_age']/3600)))
# print("No PM model stats: " + str(calc_model_stats(wn, no_wfh['avg_seir_data'], no_wfh['avg_age']/3600)))
