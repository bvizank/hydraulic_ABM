import pandas as pd
import numpy as np
from wntr.epanet.toolkit import ENepanet
from wntr.sim.results import SimulationResults
from wntr.epanet.util import EN, HydParam, QualParam, FlowUnits, MassUnits, to_si
from wntr.network.io import write_inpfile


class HydraulicSim():
    def __init__(self, model):
        self._wn = model.wn

        # setup class variables
        self._temp_index = list()
        self._temp_link_report_lines = dict()
        self._temp_node_report_lines = dict()

        # define the epanet class
        self.enData = ENepanet(version=2.2)
        self._wn.options.time.hydraulic_timestep = 3600
        self._wn.options.time.pattern_timestep = 3600
        self._wn.options.time.duration = model.days * 86400
        self._wn.options.quality.parameter = 'AGE'

        inpfile = 'temp' + str(model.id) + ".inp"
        rptfile = 'temp' + str(model.id) + ".rpt"
        outfile = 'temp' + str(model.id) + ".bin"

        # write an input file for this process
        write_inpfile(
            self._wn,
            inpfile,
            units=self._wn.options.hydraulic.inpfile_units,
            version=2.2
        )
        # open the EPANET project
        self.enData.ENopen(inpfile, rptfile, outfile)

        # need to set node and link attributes for age calculation
        self._node_attributes = [
            (EN.QUALITY, "_quality", "quality", QualParam.WaterAge._to_si),
            (EN.DEMAND, "_demand", "demand", HydParam.Demand._to_si),
            (EN.HEAD, "_head", "head", HydParam.HydraulicHead._to_si),
            (EN.PRESSURE, "_pressure", "pressure", HydParam.Pressure._to_si),
        ]
        self._link_attributes = [
            (EN.LINKQUAL, "_quality", "quality", QualParam.WaterAge._to_si),
            (EN.FLOW, "_flow", "flowrate", HydParam.Flow._to_si),
            (EN.VELOCITY, "_velocity", "velocity", HydParam.Velocity._to_si),
            (EN.HEADLOSS, "_headloss", "headloss", HydParam.HeadLoss._to_si),
            (EN.STATUS, "_user_status", "status", None),
            (EN.SETTING, "_setting", "setting", None),
        ]

        # initialize flow and mass units
        self._flow_units = FlowUnits(self.enData.ENgetflowunits())
        self._mass_units = MassUnits.mg

        # set the time to 0
        self.h_t = 0

        # setup the results object with the number of hours in the simulation
        self._setup_results_object(model.days * 86400)
        print(self._results.node['demand'])

        # run initial steps of hydraulic and quality sim
        self.enData.ENopenH()
        self.enData.ENinitH(1)
        self.enData.ENopenQ()
        self.enData.ENinitQ(1)
        self.enData.ENrunH()
        self.enData.ENrunQ()

    def _setup_results_object(self, results_size):
        '''
        From WNTR stepwise PR located:
        https://github.com/USEPA/WNTR/pull/373/files#diff-7f9f99823c3b3e1828f59b11ef24d2a235d7be36c9a1319a25b8e063f33e4019
        '''
        # results_size = 1
        self._results = SimulationResults()
        self._results.node = dict()
        self._results.link = dict()
        self._node_name_idx = list()
        self._link_name_idx = list()
        self._node_name_str = self._wn.node_name_list
        self._link_name_str = self._wn.link_name_list
        # index = [self._report_start]
        if results_size > 0:
            index = np.arange(
                0, results_size, 3600,
            )
        for node_name in self._node_name_str:
            self._node_name_idx.append(self.enData.ENgetnodeindex(node_name))
        for link_name in self._link_name_str:
            self._link_name_idx.append(self.enData.ENgetlinkindex(link_name))
        for _, _, name, _ in self._node_attributes:
            self._results.node[name] = pd.DataFrame([], columns=self._node_name_str, index=index)
            self._temp_node_report_lines[name] = list()
        for _, _, name, _ in self._link_attributes:
            self._results.link[name] = pd.DataFrame([], columns=self._link_name_str, index=index)
            self._temp_link_report_lines[name] = list()

    def _save_intermediate_values(self):
        for name, vals in self._node_sensors.items():
            en_idx, at_idx = name  # (where, what) you are measuring
            node, attr, f = vals  # WNTR node object, attribute name, and conversion function
            value = self.enData.ENgetnodevalue(en_idx, at_idx)
            if f is not None:
                value = f(self._flow_units, value)
            setattr(node, attr, value)  # set the simulation value on the node object
        for name, vals in self._link_sensors.items():
            en_idx, at_idx = name
            link, attr, f = vals
            value = self.enData.ENgetlinkvalue(en_idx, at_idx)
            if f is not None:
                value = f(self._flow_units, value)
            setattr(link, attr, value)

    def _save_report_step(self):
        # t = self.enData.ENgettimeparam(EN.HTIME)
        # this is checking to make sure we are at a report step, or if past the step, but it didn't get reported, then report out.
        # report_line = -1 if t < self._report_start else (t - self._report_start) // self._report_timestep
        # if report_line > self._last_line_added:
        # time = self._report_start + report_line * self._report_timestep
        # self._last_line_added = report_line
        # logger.debug("Reporting at time {}".format(time))
        self._temp_index.append(self.h_t)
        demand = list()
        head = list()
        pressure = list()
        quality = list()
        for idx in self._node_name_idx:
            demand.append(self.enData.ENgetnodevalue(idx, EN.DEMAND))
            head.append(self.enData.ENgetnodevalue(idx, EN.HEAD))
            pressure.append(self.enData.ENgetnodevalue(idx, EN.PRESSURE))
            quality.append(self.enData.ENgetnodevalue(idx, EN.QUALITY))
        self._temp_node_report_lines["demand"].append(demand)
        self._temp_node_report_lines["head"].append(head)
        self._temp_node_report_lines["pressure"].append(pressure)
        self._temp_node_report_lines["quality"].append(quality)
        linkqual = list()
        flow = list()
        velocity = list()
        headloss = list()
        status = list()
        setting = list()
        for idx in self._link_name_idx:
            linkqual.append(self.enData.ENgetlinkvalue(idx, EN.LINKQUAL))
            flow.append(self.enData.ENgetlinkvalue(idx, EN.FLOW))
            velocity.append(self.enData.ENgetlinkvalue(idx, EN.VELOCITY))
            headloss.append(self.enData.ENgetlinkvalue(idx, EN.HEADLOSS))
            status.append(self.enData.ENgetlinkvalue(idx, EN.STATUS))
            setting.append(self.enData.ENgetlinkvalue(idx, EN.SETTING))
        self._temp_link_report_lines["quality"].append(linkqual)
        self._temp_link_report_lines["flowrate"].append(flow)
        self._temp_link_report_lines["velocity"].append(velocity)
        self._temp_link_report_lines["headloss"].append(headloss)
        self._temp_link_report_lines["status"].append(status)
        self._temp_link_report_lines["setting"].append(setting)

    def _copy_results_object(self):
        if self._temp_index == 0:
            return

        for _, _, name, f in self._node_attributes:
            df2 = np.array(self._temp_node_report_lines[name])
            # logger.info('Size of df2: {}'.format(df2.shape))
            if f is not None:
                df2 = f(self._flow_units, df2)
            # self._results.node[name].loc[self._temp_index, :] = df2
            self._results.node[name] = pd.DataFrame(
                df2, index=self._temp_index, columns=self._wn.node_name_list
            )  # .loc[self._temp_index, :] = df2
            self._temp_node_report_lines[name] = list()
        for _, _, name, f in self._link_attributes:
            df2 = np.array(self._temp_link_report_lines[name])
            if f is not None:
                df2 = f(self._flow_units, df2)
            # self._results.link[name].loc[self._temp_index, :] = df2
            self._results.link[name] = pd.DataFrame(
                df2, index=self._temp_index, columns=self._wn.link_name_list
            )  # .loc[self._temp_index, :] = df2
            self._temp_link_report_lines[name] = list()
        self._temp_index = list()

    def run_sim(self, curr_step, next_step):
        # self._wn._prev_sim_time = self.h_t
        # run one time step of hydraulic and quality simulation
        self.enData.ENrunH()
        self.enData.ENrunQ()
        # set the simulation time of the water network to match the current
        # hydraulic time
        # self._wn.sim_time = self.enData.ENgettimeparam(EN.HTIME)

        # set the hydraulic time
        self.h_t = self.enData.ENgettimeparam(EN.HTIME)
        # move EPANET forward one time step
        tstep = self.enData.ENnextH()
        qstep = self.enData.ENnextQ()

        print(f"Hydraulic step: {tstep}")
        print(f"Quality step: {qstep}")

        self.nh_t = self.h_t + tstep
        print(self.nh_t)

        ''' Next we save results '''
        self._save_report_step()
        self._copy_results_object()

    def close(self):
        self.enData.ENcloseH()
        self.enData.ENcloseQ()
        self.enData.ENreport()
        self.enData.ENclose()
        self.enData = None

        # self._copy_results_object()
