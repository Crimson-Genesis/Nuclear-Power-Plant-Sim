#!/usr/bin/env python

from os import wait
import re
import yaml
from pprint import pprint
import time
import random
import numpy as np

random.seed(69)


class reactor_test:
    def __init__(self):
        with open("../config/reactor.yaml", "r") as file:
            self.config = yaml.safe_load(file)
            self.params = self.config["parameters"]

        self.tick_index = 0
        self.tick_data = {
            "sim_time": 13.0,
            "tick_index": self.tick_index,
            "time_step": 1.0,
            "time_scale": 1.0,
            "running": True,
        }

        self.data_array = dict()

        # Control inputs reading
        self.kb = self.params["kb"]["value"]
        self.B_ref = self.params["boron_ppm"]["value"]
        self.B = self.params["boron_ppm"]["value"]
        self.fuel_alpha = self.params["fuel_alpha"]["value"]
        self.T_fuel = self.params["Tin_coolant"]["value"]
        self.T_fuel_ref = self.params["Tin_coolant"]["value"]
        self.mod_alpha = self.params["mod_alpha"]["value"]
        self.T_mod = (self.T_fuel + self.T_fuel_ref) / 2
        self.void_alpha = self.params["void_alpha"]["value"]
        self.void_fraction = 0.0
        self.BU_coeff = self.params["BU_coeff"]["value"]
        self.BU = self.params["burnup"]["value"]
        self.rho_scram = self.params["rho_scram"]["value"]
        self.scram_flag = self.params["scram_flag"]["value"]
        self.phi = self.params["phi"]["value"]
        self.beta_i = np.array(self.params["beta_i"]["value"])
        self.lambda_i = np.array(self.params["lambda_i"]["value"])
        self.Lambda = self.params["Lambda"]["value"]
        # self.C = np.array(
        #     [
        #         (bi / (self.Lambda * li)) * self.phi
        #         for bi, li in zip(self.beta_i, self.lambda_i)
        #     ]
        # )
        self.C = self.params["C_i"]["value"]
        self.P_rated_MW = self.params["P_rated_MW"]["value"]
        self.phi_nominal = self.params["phi_nominal"]["value"]
        self.R_f = self.params["R_f"]["value"]
        self.A_heated_total = self.params["A_heated_total"]["value"]
        self.L_fuel = self.params["L_fuel"]["value"]
        self.N_rods = self.params["N_rods"]["value"]
        self.K_f = self.params["k_fuel"]["value"]
        self.h_gap = self.params["h_gap"]["value"]
        self.M_fuel = self.params["M_fuel"]["value"]
        self.PCT = self.params["PCT"]["value"]

        # It will come from the master_clock tick (the 'dt_local','tick', 'dt_global', )
        self.dt = np.float64(self.params["dt"]["value"])

        # Temp variables which will be replased by dynamic values in the actual simulation.
        # Coolant input reading
        self.flow_rate_kg_s = self.params["flow_rate_kg_s"]["value"]
        self.Tin_coolant = self.params["Tin_coolant"]["value"]
        self.P_coolant_MPa = self.params["P_coolant_MPa"]["value"]
        self.qn_crit = 1.8e6

        # Hoock this shit up later
        self.cp = 4180.0  # J/kg-K
        self.rho_coolant = 700.0  # kg/m³
        self.mu_coolant = 1e-4  # Pa·s
        self.k_coolant = 0.6  # W/m·K
        self.cp_coolant = 4180.0  # J/kg·K
        self.A = 3.0

    def update_params_random_values(self):
        global params
        for i, j in self.params.items():
            if j["random"]:
                if type(j["value"]) == float:
                    self.params[i]["value"] = np.random.uniform(
                        self.params[i]["min"], self.params[i]["max"]
                    )
                elif type(j["value"]) == int:
                    self.params[i]["value"] = np.random.randint(
                        self.params[i]["min"], self.params[i]["max"]
                    )

    def main(self):
        self.update_params_random_values()
        dt_local = 1e-6
        dt_global = 1e-3
        steps = int(dt_global / dt_local)
        for _ in range(100):
            # Reactivity contributions
            self.total_rods = sum(
                [b["rods"] for b in self.params["rod_insertion_depth"]["banks"]]
            )
            self.rho_rods = 0.0
            for b in self.params["rod_insertion_depth"]["banks"]:
                worth_bank = self.params["control_rod_worth"]["value"] * (
                    b["rods"] / self.total_rods
                )
                frac_inserted = b["depth"] / b["length"]
                self.rho_rods += worth_bank * frac_inserted  # ***

            self.rho_boron = self.kb * (self.B - self.B_ref)  # ***
            self.rho_fuel = self.fuel_alpha * (self.T_fuel - self.T_fuel_ref)  # ***
            self.rho_mod = self.mod_alpha * (self.T_mod - self.T_fuel_ref)  # ***
            self.rho_void = self.void_alpha * self.void_fraction  # ***
            self.rho_temp = self.rho_fuel + self.rho_mod + self.rho_void  # ***
            self.rho_burnup = self.BU_coeff * self.BU  # ***
            self.rho_total = (
                self.rho_rods + self.rho_boron + self.rho_temp + self.rho_burnup
            )  # ***
            if self.scram_flag:
                self.rho_total = self.rho_scram  # ***

            # Neutronics integration
            self.total_beta_i = np.sum(self.beta_i)  # ***

            # self.dphi_dt = ((self.rho_total - self.total_beta_i) / self.Lambda) * self.phi \
            #                + np.sum(self.lambda_i * self.C) # ***
            # self.phi_new = self.phi + self.dt * self.dphi_dt # ***
            #
            # # Update precursors
            # self.C_new = []
            # for ci, bi, li in zip(self.C, self.beta_i, self.lambda_i):
            #     ci_new = (
            #         ci * np.exp(-li * self.dt)
            #         + (bi / self.Lambda) * self.phi * (1 - np.exp(-li * self.dt)) / li
            #     )
            #     self.C_new.append(ci_new)
            # self.C_new = np.array(self.C_new)
            #
            # # Update phi for next step
            # self.phi = self.phi_new

            dphi_dt = (
                (self.rho_total - self.total_beta_i) / self.Lambda
            ) * self.phi + np.sum(self.lambda_i * self.C)
            dC_dt = (self.beta_i / self.Lambda) * self.phi - self.lambda_i * self.C
            self.phi += dt_local * dphi_dt
            self.C += dt_local * dC_dt
            # print(f"{time} | Phi:- {self.phi} | C:- {self.C}")

            # Thermal power mapping
            self.P_th = self.P_rated_MW * (self.phi / self.phi_nominal)  # ***

            # ΔT and T\_out
            self.P_th_W = self.P_th * 1e6  # ***
            self.delta_T = self.P_th_W / (self.flow_rate_kg_s * self.cp)  # ***
            self.T_out = self.Tin_coolant + self.delta_T  # ***

            # Hydraulic calculations
            self.coolant_velocity = self.flow_rate_kg_s / (self.rho_coolant * self.A)  # ***
            self.G = self.flow_rate_kg_s / self.A  # ***

            self.P_wet = self.total_rods * (2 * np.pi * self.R_f)
            self.A_flow = self.A - (self.total_rods * np.pi * self.R_f**2)
            self.D_h = 4 * self.A_flow / self.P_wet  # ***

            # self.A_rods = self.total_rods * np.pi * self.R_f**2
            # self.A_flow = self.A - self.A_rods
            # self.P_wet = self.total_rods * (2 * np.pi * self.R_f)
            # self.D_h = 4 * self.A_flow / self.P_wet

            # Heat transfer coefficients
            self.Re = (
                self.rho_coolant * self.coolant_velocity * self.D_h
            ) / self.mu_coolant
            self.Pr = (self.cp_coolant * self.mu_coolant) / self.k_coolant
            self.Nu = 0.023 * (self.Re**0.8) * (self.Pr**0.4)
            self.h = (self.Nu * self.k_coolant) / self.D_h

            # Fuel / rod temps
            self.qn = self.P_th_W / self.A_heated_total
            self.per_rod_power = self.P_th_W / self.N_rods
            self.LHGR = (self.per_rod_power / 1000) / self.L_fuel
            self.qm = self.per_rod_power / (np.pi * (self.R_f ** 2) * self.L_fuel)

            # Fuel state updates
            self.T_coolant_film = self.T_out + 5
            self.T_clad = self.T_coolant_film + (self.qn / self.h)
            self.T_center = self.T_clad + ((self.qm * (self.R_f ** 2))/ (4 * self.K_f))
            self.Burnup_inc = ((self.P_th * self.dt) * (1/86400)) / self.M_fuel
            self.BU += self.Burnup_inc
            self.FGR = self.FGR_empirical(self.BU)

            # Safety parameters
            self.delta_R = self.BU * (1e-5)
            self.DNBR = self.qn_crit / self.qn

            if (self.DNBR < 1.3 and self.PCT > 1200 and self.LHGR > 20):
                print("SCRAM - SCRAM - SCRAM - SCRAM")
                # if this then include the scram signal here

            # Telemetry / alarms


            self.data_array[time.time()] =  self.__dict__
            # print(f"{time.time()} - Done...")

    def FGR_empirical(self, BU):
        return 0.25 * (1 - np.exp(-0.1 * (BU - 20))) if BU > 20 else 0.0


if __name__ == "__main__":
    r = reactor_test()
    r.main()
    pprint(r.data_array)
