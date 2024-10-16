# ==================================================================================================
# --- Imports
# ==================================================================================================
import json
import os
from functools import lru_cache
from importlib.resources import files

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
import yaml
from scipy import constants


# ==================================================================================================
# --- Class definition
# ==================================================================================================
class ColliderCheck:
    def __init__(self, collider, path_filling_scheme=None, type_particles="proton"):
        """Initialize the ColliderCheck class directly from a collider, potentially embedding a
        configuration file."""

        # Store the collider
        self.collider = collider

        # Store the filling scheme path
        self.path_filling_scheme = path_filling_scheme

        # Check the type of particles and store
        if type_particles in ["proton", "lead"]:
            self.type_particles = type_particles
        else:
            raise ValueError("type_particles must be either 'proton' or 'lead'.")

        # Record cross-section correspondinlgy
        if self.type_particles == "proton":
            self.cross_section = 81e-27
        elif self.type_particles == "lead":
            self.cross_section = 281e-24

        # Define the configuration through a property since it might not be there
        self._configuration = None
        self.configuration_str = None

        # Beam energy
        self.energy = self.collider.lhcb1.particle_ref._p0c[0] / 1e9

        # Get twiss and survey dataframes for both beams
        self.tw_b1, self.sv_b1 = [self.collider.lhcb1.twiss(), self.collider.lhcb1.survey()]
        self.tw_b2, self.sv_b2 = [self.collider.lhcb2.twiss(), self.collider.lhcb2.survey()]
        self.df_tw_b1, self.df_sv_b1 = [self.tw_b1.to_pandas(), self.sv_b1.to_pandas()]
        self.df_tw_b2, self.df_sv_b2 = [self.tw_b2.to_pandas(), self.sv_b2.to_pandas()]

        # Variables used to compute the separation (computed on the fly)
        self.dic_survey_per_ip = {"lhcb1": {}, "lhcb2": {}}

    @property
    def configuration(self):
        """Loads the configuration, as well as the luminosity and filling scheme arrays, if they're
        not already loaded."""

        if self._configuration is not None:
            return self._configuration
        # Get the corresponding configuration if it's there
        if hasattr(self.collider, "metadata") and self.collider.metadata != {}:
            self.configuration = self.collider.metadata
        return self._configuration

    @configuration.setter
    def configuration(self, configuration_dict):
        """This function is used to update the configuration, and the attributes that depend on it."""
        self._configuration = configuration_dict
        self._update_attributes_configuration()

    def _update_attributes_configuration(self):
        # Ensure that the configuration format is correct
        if "config_collider" not in self.configuration:
            print("Warning, the provided configuration doesn't embed mad configuration.")
            self.configuration = {
                "config_collider": self.configuration,
                "config_mad": {},
            }
        self.configuration_str = yaml.dump(self.configuration)

        # Compute luminosity and filling schemes attributes
        self._load_configuration_luminosity()
        self._load_filling_scheme_arrays()

        # Clean cache for separation computation
        self.compute_separation_variables.cache_clear()

    @property
    def nemitt_x(self):
        if self.configuration is not None:
            return self.configuration["config_collider"]["config_beambeam"]["nemitt_x"]
        print("Warning: no configuration provided. Using default value of 2.2e-6 for nemitt_x.")
        return 2.2e-6

    @property
    def nemitt_y(self):
        if self.configuration is not None:
            return self.configuration["config_collider"]["config_beambeam"]["nemitt_y"]
        print("Warning: no configuration provided. Using default value of 2.2e-6 for nemitt_y.")
        return 2.2e-6

    @property
    def n_lr_per_side(self):
        if self.configuration is not None:
            return self.configuration["config_collider"]["config_beambeam"][
                "num_long_range_encounters_per_side"
            ]["ip1"]
        print("Warning: no configuration provided. Using default value of 1 for n_lr_per_side.")
        return 16

    def _check_configuration(self):
        if self.configuration is None:
            raise ValueError(
                "No configuration has been provided when instantiating the ColliderCheck object."
            )

    def _load_configuration_luminosity(self):
        self.num_particles_per_bunch = float(
            self.configuration["config_collider"]["config_beambeam"]["num_particles_per_bunch"]
        )
        self.sigma_z = self.configuration["config_collider"]["config_beambeam"]["sigma_z"]

    def _load_filling_scheme_arrays(self):
        if self.path_filling_scheme is None:
            # Get the filling scheme path (should already be an absolute path)
            self.path_filling_scheme = self.configuration["config_collider"]["config_beambeam"][
                "mask_with_filling_pattern"
            ]["pattern_fname"]

            # Check if filling scheme file exists, and replace it by local if not
            if not os.path.isfile(self.path_filling_scheme):
                try:
                    package_path = str(files("collider_dashboard"))
                except NameError as e:
                    raise ValueError(
                        "collider_dashboard not installed... Filling scheme file could not be"
                        " loaded from the path in the configuration or locally."
                    ) from e
                if os.path.isfile(
                    f"{package_path}/data/" + self.path_filling_scheme.split("/")[-1]
                ):
                    print(
                        "Filling scheme file could not be loaded from the path in the"
                        " configuration. Loading it locally."
                    )
                    self.path_filling_scheme = (
                        f"{package_path}/data/" + self.path_filling_scheme.split("/")[-1]
                    )
                else:
                    raise ValueError(
                        "Filling scheme file could not be loaded from the path in the configuration"
                        " or locally."
                    )

        # Load the scheme (two boolean arrays representing the buckets in the two beams)
        with open(self.path_filling_scheme) as fid:
            filling_scheme = json.load(fid)

        self.array_b1 = np.array(filling_scheme["beam1"])
        self.array_b2 = np.array(filling_scheme["beam2"])

        # Get the bunches selected for tracking
        self.i_bunch_b1 = self.configuration["config_collider"]["config_beambeam"][
            "mask_with_filling_pattern"
        ]["i_bunch_b1"]
        self.i_bunch_b2 = self.configuration["config_collider"]["config_beambeam"][
            "mask_with_filling_pattern"
        ]["i_bunch_b2"]

    def return_number_of_collisions(self, IP=1):
        """Computes and returns the number of collisions at the requested IP."""

        # Ensure configuration is defined
        self._check_configuration()

        # Assert that the arrays have the required length, and do the convolution
        assert len(self.array_b1) == len(self.array_b2) == 3564
        if IP in [1, 5]:
            return self.array_b1 @ self.array_b2
        elif IP == 2:
            return np.roll(self.array_b1, 891) @ self.array_b2
        elif IP == 8:
            return np.roll(self.array_b1, 2670) @ self.array_b2
        else:
            raise ValueError("IP must be either 1, 2, 5 or 8.")

    def return_luminosity(self, IP=1):
        """Computes and returns the luminosity at the requested IP. External twiss (e.g. from before
        beam-beam) can be provided."""

        # Ensure configuration is defined
        self._check_configuration()

        # Check crab cavities
        crab = False
        if (
            "on_crab1"
            in self.configuration["config_collider"]["config_knobs_and_tuning"]["knob_settings"]
        ):
            crab_val = float(
                self.configuration["config_collider"]["config_knobs_and_tuning"]["knob_settings"][
                    "on_crab1"
                ]
            )
            if abs(crab_val) > 0:
                crab = True

        if IP not in [1, 2, 5, 8]:
            raise ValueError("IP must be either 1, 2, 5 or 8.")
        n_col = self.return_number_of_collisions(IP=IP)
        return xt.lumi.luminosity_from_twiss(
            n_colliding_bunches=n_col,
            num_particles_per_bunch=self.num_particles_per_bunch,
            ip_name=f"ip{str(IP)}",
            nemitt_x=self.nemitt_x,
            nemitt_y=self.nemitt_y,
            sigma_z=self.sigma_z,
            twiss_b1=self.tw_b1,
            twiss_b2=self.tw_b2,
            crab=crab,
        )

    def return_twiss_at_ip(self, beam=1, ip=1):
        """Returns the twiss parameters, position and angle at the requested IP."""
        if beam == 1:
            return (
                self.tw_b1.rows[f"ip{ip}"]
                .cols["s", "x", "px", "y", "py", "betx", "bety", "dx", "dy"]
                .to_pandas()
            )
        elif beam == 2:
            return (
                self.tw_b2.rows[f"ip{ip}"]
                .cols["s", "x", "px", "y", "py", "betx", "bety", "dx", "dy"]
                .to_pandas()
            )
        else:
            raise ValueError("Beam must be either 1 or 2.")

    def return_tune_and_chromaticity(self, beam=1):
        """Returns the tune and chromaticity for the requested beam."""
        if beam == 1:
            return self.tw_b1["qx"], self.tw_b1["dqx"], self.tw_b1["qy"], self.tw_b1["dqy"]
        elif beam == 2:
            return self.tw_b2["qx"], self.tw_b2["dqx"], self.tw_b2["qy"], self.tw_b2["dqy"]
        else:
            raise ValueError("Beam must be either 1 or 2.")

    def return_linear_coupling(self):
        """Returns the linear coupling for the two beams."""
        return self.tw_b1["c_minus"], self.tw_b2["c_minus"]

    def return_momentum_compaction_factor(self):
        """Returns the momentum compaction factor for the two beams."""
        return self.tw_b1["momentum_compaction_factor"], self.tw_b2["momentum_compaction_factor"]

    def return_polarity_ip_2_8(self):
        """Return the polarity (internal angle of the experiments) for IP2 and IP8."""
        # Ensure configuration is defined
        self._check_configuration()

        polarity_alice = self.configuration["config_collider"]["config_knobs_and_tuning"][
            "knob_settings"
        ]["on_alice_normalized"]
        polarity_lhcb = self.configuration["config_collider"]["config_knobs_and_tuning"][
            "knob_settings"
        ]["on_lhcb_normalized"]

        return polarity_alice, polarity_lhcb

    def _compute_ip_specific_separation(self, ip="ip1", beam_weak="b1"):
        # Compute survey at IP if needed
        if ip not in self.dic_survey_per_ip["lhcb1"] or ip not in self.dic_survey_per_ip["lhcb2"]:
            self.dic_survey_per_ip["lhcb1"][ip] = self.collider["lhcb1"].survey(element0=ip)
            self.dic_survey_per_ip["lhcb2"][ip] = (
                self.collider["lhcb2"].survey(element0=ip).reverse()
            )

        # Define strong and weak beams
        if beam_weak == "b1":
            beam_strong = "b2"
            twiss_weak = self.tw_b1
            twiss_strong = self.tw_b2.reverse()
            survey_weak = self.dic_survey_per_ip["lhcb1"]
            survey_strong = self.dic_survey_per_ip["lhcb2"]
        else:
            beam_strong = "b1"
            twiss_weak = self.tw_b2.reverse()
            twiss_strong = self.tw_b1
            survey_weak = self.dic_survey_per_ip["lhcb2"]
            survey_strong = self.dic_survey_per_ip["lhcb1"]

        my_filter_string = f"bb_(ho|lr)\.(r|l|c){ip[2]}.*"
        survey_filtered = {
            beam_strong: survey_strong[ip].rows[my_filter_string].cols[["X", "Y", "Z"]],
            beam_weak: survey_weak[ip].rows[my_filter_string].cols[["X", "Y", "Z"]],
        }
        twiss_filtered = {
            beam_strong: twiss_strong.rows[my_filter_string],
            beam_weak: twiss_weak.rows[my_filter_string],
        }
        s = survey_filtered[beam_strong]["Z"]
        # Compute if the beambeam element is on or off (list of 1 and 0)
        l_scale_strength = [
            (
                self.collider[f"lhc{beam_strong}"].vars[f"{name_el}_scale_strength"]._value
                * self.collider.vars["beambeam_scale"]._value
            )
            for name_el in twiss_filtered[beam_strong].name
        ]
        d_x_weak_strong_in_meter = (
            twiss_filtered[beam_weak]["x"]
            - twiss_filtered[beam_strong]["x"]
            + survey_filtered[beam_weak]["X"]
            - survey_filtered[beam_strong]["X"]
        )
        d_y_weak_strong_in_meter = (
            twiss_filtered[beam_weak]["y"]
            - twiss_filtered[beam_strong]["y"]
            + survey_filtered[beam_weak]["Y"]
            - survey_filtered[beam_strong]["Y"]
        )

        return (
            s,
            my_filter_string,
            beam_strong,
            twiss_filtered,
            survey_filtered,
            d_x_weak_strong_in_meter,
            d_y_weak_strong_in_meter,
            l_scale_strength,
        )

    def _compute_emittances_separation(self):
        if self.type_particles == "proton":
            # gamma relativistic of a proton
            gamma_rel = self.energy / (
                constants.physical_constants["proton mass energy equivalent in MeV"][0] / 1000
            )
        elif self.type_particles == "lead":
            # gamma relativistic of a lead ion (value needs to be double-checked)
            gamma_rel = self.energy / (193084.751 / 1000)
        else:
            raise ValueError("type_particles must be either 'proton' or 'lead'.")

        # beta relativistic of a proton at 7 TeV
        beta_rel = np.sqrt(1 - 1 / gamma_rel**2)

        emittance_strong_x = self.nemitt_x / gamma_rel / beta_rel
        emittance_strong_y = self.nemitt_y / gamma_rel / beta_rel

        emittance_weak_x = self.nemitt_x / gamma_rel / beta_rel
        emittance_weak_y = self.nemitt_y / gamma_rel / beta_rel

        return (
            gamma_rel,
            beta_rel,
            emittance_weak_x,
            emittance_weak_y,
            emittance_strong_x,
            emittance_strong_y,
        )

    def _compute_ip_specific_normalized_separation(
        self,
        twiss_filtered,
        beam_weak,
        beam_strong,
        emittance_strong_x,
        emittance_strong_y,
        emittance_weak_x,
        emittance_weak_y,
        d_x_weak_strong_in_meter,
        d_y_weak_strong_in_meter,
    ):
        # Size of the strong beams
        sigma_x_strong = np.sqrt(twiss_filtered[beam_strong]["betx"] * emittance_strong_x)
        sigma_y_strong = np.sqrt(twiss_filtered[beam_strong]["bety"] * emittance_strong_y)

        # Size of the weak beams
        sigma_x_weak = np.sqrt(twiss_filtered[beam_weak]["betx"] * emittance_weak_x)
        sigma_y_weak = np.sqrt(twiss_filtered[beam_weak]["bety"] * emittance_weak_y)

        # Normalized separation
        dx_sig = d_x_weak_strong_in_meter / sigma_x_strong
        dy_sig = d_y_weak_strong_in_meter / sigma_y_strong

        # Flatness of the beam
        A_w_s = sigma_x_weak / sigma_y_strong
        B_w_s = sigma_y_weak / sigma_x_strong

        fw = 1
        r = sigma_y_strong / sigma_x_strong

        return (
            d_x_weak_strong_in_meter,
            d_y_weak_strong_in_meter,
            dx_sig,
            dy_sig,
            A_w_s,
            B_w_s,
            fw,
            r,
        )

    # Cache function to gain time
    @lru_cache(maxsize=20)
    def compute_separation_variables(self, ip="ip1", beam_weak="b1"):
        """This function computes all the variables needed to compute the separation at the
        requested IP, in a weak-strong setting. The variables are stored and returned in a
        dictionnary.
        """

        # Get variables specific to the requested IP
        (
            s,
            my_filter_string,
            beam_strong,
            twiss_filtered,
            survey_filtered,
            d_x_weak_strong_in_meter,
            d_y_weak_strong_in_meter,
            l_scale_strength,
        ) = self._compute_ip_specific_separation(ip=ip, beam_weak=beam_weak)

        # Get emittances
        (
            gamma_rel,
            beta_rel,
            emittance_weak_x,
            emittance_weak_y,
            emittance_strong_x,
            emittance_strong_y,
        ) = self._compute_emittances_separation()

        # Get normalized separation
        (
            d_x_weak_strong_in_meter,
            d_y_weak_strong_in_meter,
            dx_sig,
            dy_sig,
            A_w_s,
            B_w_s,
            fw,
            r,
        ) = self._compute_ip_specific_normalized_separation(
            twiss_filtered,
            beam_weak,
            beam_strong,
            emittance_strong_x,
            emittance_strong_y,
            emittance_weak_x,
            emittance_weak_y,
            d_x_weak_strong_in_meter,
            d_y_weak_strong_in_meter,
        )

        return {
            "twiss_filtered": twiss_filtered,
            "survey_filtered": survey_filtered,
            "s": s,
            "dx_meter": d_x_weak_strong_in_meter,
            "dy_meter": d_y_weak_strong_in_meter,
            "dx_sig": dx_sig,
            "dy_sig": dy_sig,
            "A_w_s": A_w_s,
            "B_w_s": B_w_s,
            "fw": fw,
            "r": r,
            "emittance_strong_x": emittance_strong_x,
            "emittance_strong_y": emittance_strong_y,
            "emittance_weak_x": emittance_weak_x,
            "emittance_weak_y": emittance_weak_y,
            "gamma_rel": gamma_rel,
            "beta_rel": beta_rel,
            "energy": self.energy,
            "my_filter_string": my_filter_string,
            "beam_weak": beam_weak,
            "beam_strong": beam_strong,
            "ip": ip,
            "l_scale_strength": l_scale_strength,
        }

    def return_dic_position_all_ips(self):
        """This function computes all the variables needed to compute the position of the beam in
        all IRs. The variables are stored and returne in a dictionnary. The extreme positions are:
        IP1 : mqy.4l1.b1 to mqy.4r1.b1
        IP2 : mqy.b5l2.b1 to mqy.b4r2.b1
        IP5 : mqy.4l5.b1 to mqy.4r5.b1
        IP8 : mqy.b4l8.b1 to mqy.b4r8.b1
        """
        dic_larger_separation_ip = {"lhcb1": {"sv": {}, "tw": {}}, "lhcb2": {"sv": {}, "tw": {}}}
        for beam in ("lhcb1", "lhcb2"):
            for ip, el_start, el_end in zip(
                ["ip1", "ip2", "ip5", "ip8"],
                ["mqy.4l1", "mqy.b4l2", "mqy.4l5", "mqy.b4l8"],
                ["mqy.4r1", "mqy.b4r2", "mqy.4r5", "mqy.b4r8"],
            ):
                # Change element name for current beam
                el_start = f"{el_start}.{beam[3:]}"
                el_end = f"{el_end}.{beam[3:]}"

                # Recompute survey near IP
                if beam == "lhcb1":
                    df_sv = (
                        self.collider[beam].survey(element0=ip).rows[el_start:el_end].to_pandas()
                    )
                    df_tw = self.tw_b1.rows[el_start:el_end].to_pandas()
                else:
                    df_sv = (
                        self.collider[beam]
                        .survey(element0=ip)
                        .reverse()
                        .rows[el_start:el_end]
                        .to_pandas()
                    )
                    df_tw = self.tw_b2.reverse().rows[el_start:el_end].to_pandas()

                # Remove entry and exit elements
                df_tw = df_tw[~df_tw["name"].str.contains("entry")]
                df_tw = df_tw[~df_tw["name"].str.contains("exit")]
                df_sv = df_sv[~df_sv["name"].str.contains("entry")]
                df_sv = df_sv[~df_sv["name"].str.contains("exit")]

                # Store dataframe of elements between s_start and s_end
                dic_larger_separation_ip[beam]["sv"][ip] = df_sv
                dic_larger_separation_ip[beam]["tw"][ip] = df_tw

                # Delete all .b1 and .b2 from element names
                for tw_sv in ("sv", "tw"):
                    dic_larger_separation_ip[beam][tw_sv][ip].loc[:, "name"] = [
                        el.replace(f".{beam[3:]}", "").replace(f"{beam[3:]}_", "")
                        for el in dic_larger_separation_ip[beam][tw_sv][ip].name
                    ]

        for ip in ["ip1", "ip2", "ip5", "ip8"]:
            # Get intersection of names in twiss and survey
            s_intersection = (
                set(dic_larger_separation_ip["lhcb2"]["sv"][ip].name)
                .intersection(set(dic_larger_separation_ip["lhcb1"]["sv"][ip].name))
                .intersection(set(dic_larger_separation_ip["lhcb2"]["tw"][ip].name))
                .intersection(set(dic_larger_separation_ip["lhcb1"]["tw"][ip].name))
            )

            for tw_sv in ("sv", "tw"):
                # Clean dataframes in both beams so that they are comparable
                for beam in ["lhcb1", "lhcb2"]:
                    # Remove all rows whose name is not in both beams
                    dic_larger_separation_ip[beam][tw_sv][ip] = dic_larger_separation_ip[beam][
                        tw_sv
                    ][ip][dic_larger_separation_ip[beam][tw_sv][ip].name.isin(s_intersection)]

                    # Remove all elements whose name contains '..'
                    for i in range(1, 6):
                        dic_larger_separation_ip[beam][tw_sv][ip] = dic_larger_separation_ip[beam][
                            tw_sv
                        ][ip][
                            ~dic_larger_separation_ip[beam][tw_sv][ip].name.str.endswith(f"..{i}")
                        ]

                # Center s around IP for beam 1
                dic_larger_separation_ip["lhcb1"][tw_sv][ip].loc[:, "s"] = (
                    dic_larger_separation_ip["lhcb1"][tw_sv][ip].loc[:, "s"]
                    - dic_larger_separation_ip["lhcb1"][tw_sv][ip][
                        dic_larger_separation_ip["lhcb1"][tw_sv][ip].name == ip
                    ].s.to_numpy()
                )

                # Set the s of beam 1 as reference for all dataframes
                dic_larger_separation_ip["lhcb2"][tw_sv][ip].loc[:, "s"] = dic_larger_separation_ip[
                    "lhcb1"
                ][tw_sv][ip].s.to_numpy()

        return dic_larger_separation_ip

    def plot_orbits(self, ip="ip1", beam_weak="b1"):
        """Plots the beams orbits at the requested IP."""

        # Get separation variables
        ip_dict = self.compute_separation_variables(ip=ip, beam_weak=beam_weak)

        # Do the plot
        plt.figure()
        plt.title(f'IP{ip_dict["ip"][2]}')
        beam_weak = ip_dict["beam_weak"]
        beam_strong = ip_dict["beam_strong"]
        twiss_filtered = ip_dict["twiss_filtered"]
        plt.plot(ip_dict["s"], twiss_filtered[beam_weak]["x"], "ob", label=f"x {beam_weak}")
        plt.plot(ip_dict["s"], twiss_filtered[beam_strong]["x"], "sb", label=f"x {beam_strong}")
        plt.plot(ip_dict["s"], twiss_filtered[beam_weak]["y"], "or", label=f"y {beam_weak}")
        plt.plot(ip_dict["s"], twiss_filtered[beam_strong]["y"], "sr", label=f"y {beam_strong}")
        plt.xlabel("s [m]")
        plt.ylabel("x,y [m]")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_separation(self, ip="ip1", beam_weak="b1"):
        """Plots the normalized separation at the requested IP."""
        # Get separation variables
        ip_dict = self.compute_separation_variables(ip=ip, beam_weak=beam_weak)

        # Do the plot
        plt.figure()
        plt.title(f'IP{ip_dict["ip"][2]}')
        plt.plot(ip_dict["s"], np.abs(ip_dict["dx_sig"]), "ob", label="x")
        plt.plot(ip_dict["s"], np.abs(ip_dict["dy_sig"]), "sr", label="y")
        plt.xlabel("s [m]")
        plt.ylabel("separation in x,y [$\sigma$]")
        plt.legend()
        plt.grid(True)
        plt.show()

    def output_check_as_str(self, path_output=None):
        """Summarizes the collider observables in a string, and optionally to a file."""
        # Check tune and chromaticity
        qx_b1, dqx_b1, qy_b1, dqy_b1 = self.return_tune_and_chromaticity(beam=1)
        qx_b2, dqx_b2, qy_b2, dqy_b2 = self.return_tune_and_chromaticity(beam=2)
        str_file = "" + "Tune and chromaticity\n"
        str_file += (
            f"Qx_b1 = {qx_b1:.4f}, Qy_b1 = {qy_b1:.4f}, dQx_b1 = {dqx_b1:.4f}, dQy_b1 ="
            f" {dqy_b1:.4f}\n"
        )
        str_file += (
            f"Qx_b2 = {qx_b2:.4f}, Qy_b2 = {qy_b2:.4f}, dQx_b2 = {dqx_b2:.4f}, dQy_b2 ="
            f" {dqy_b2:.4f}\n"
        )
        str_file += "\n\n"

        # Check linear coupling
        c_minus_b1, c_minus_b2 = self.return_linear_coupling()
        str_file += "Linear coupling\n"
        str_file += f"C- b1 = {c_minus_b1:.4f}, C- b2 = {c_minus_b2:.4f}\n"

        # Check momentum compaction factor
        alpha_p_b1, alpha_p_b2 = self.return_momentum_compaction_factor()
        str_file += "Momentum compaction factor\n"
        str_file += f"alpha_p b1 = {alpha_p_b1:.4f}, alpha_p b2 = {alpha_p_b2:.4f}\n"

        str_file += "\n\n"

        # Check twiss observables at all IPs
        str_file += "Twiss observables\n"
        for ip in [1, 2, 5, 8]:
            tw_b1 = self.return_twiss_at_ip(beam=1, ip=ip).to_string(index=False)
            tw_b2 = self.return_twiss_at_ip(beam=2, ip=ip).to_string(index=False)
            str_file += f"IP{ip} (beam 1)\n"
            str_file += tw_b1 + "\n"
            str_file += f"IP{ip} (beam 2)\n"
            str_file += tw_b2 + "\n"
            str_file += "\n"

        str_file += "\n\n"

        if self.configuration is not None:
            # Check luminosity
            lumi1 = self.return_luminosity(IP=1)
            lumi2 = self.return_luminosity(IP=2)
            lumi5 = self.return_luminosity(IP=5)
            lumi8 = self.return_luminosity(IP=8)
            str_file += "Luminosity\n"
            str_file += (
                f"IP1 = {lumi1:.4e}, IP2 = {lumi2:.4e}, IP5 = {lumi5:.4e}, IP8 = {lumi8:.4e}\n"
            )

            str_file += "\n\n"

        if path_output is not None:
            # Write to file
            with open(path_output, "w") as fid:
                fid.write(str_file)

        return str_file


# ==================================================================================================
# --- Main script
# ==================================================================================================
if __name__ == "__main__":
    path_collider = "../test_data/collider.json"
    collider = xt.Multiline.from_json(path_collider)
    collider.build_trackers()

    # Do collider check
    collider_check = ColliderCheck(collider=collider)
    print(collider_check.output_check_as_str(path_output="../output/check.txt"))
    print(collider_check.output_check_as_str(path_output="../output/check.txt"))
