#### Imports
import numpy as np
import yaml
import json
import xtrack as xt
import os


class ColliderCheck:
    def __init__(self, collider):
        """Initialize the ColliderCheck class directly from a collider, potentially embedding a
        configuration file."""

        # Store the collider
        self.collider = collider

        # Define the configuration through a property since it might not be there
        self._configuration = None
        self._attributes_defined = False

        # Get twiss and survey dataframes for both beams
        self.tw_b1, self.sv_b1 = [self.collider.lhcb1.twiss(), self.collider.lhcb1.survey()]
        self.tw_b2, self.sv_b2 = [self.collider.lhcb2.twiss(), self.collider.lhcb2.survey()]
        self.df_tw_b1, self.df_sv_b1 = [self.tw_b1.to_pandas(), self.sv_b1.to_pandas()]
        self.df_tw_b2, self.df_sv_b2 = [self.tw_b2.to_pandas(), self.sv_b2.to_pandas()]

    @property
    def configuration(self):
        if self._configuration is not None:
            return self._configuration
        else:
            # Get the corresponding configuration if it's there
            if hasattr(self.collider, "metadata"):
                self._configuration = self.collider.metadata
                self.update_attributes_configuration()

        return self._configuration

    @configuration.setter
    def configuration(self, configuration_dict):
        self._configuration = configuration_dict
        self.update_attributes_configuration()

    def update_attributes_configuration(self):
        # Compute luminosity and filling schemes attributes
        self.load_configuration_luminosity()
        self.load_filling_scheme_arrays()
        self._attributes_defined = True

    def load_configuration_luminosity(self):
        """Returns the configuration file variables used to compute the luminosity."""
        if "num_particles_per_bunch_after_optimization" in self.configuration["config_beambeam"]:
            self.num_particles_per_bunch = float(
                self.configuration["config_beambeam"]["num_particles_per_bunch_after_optimization"]
            )
        else:
            print(
                "Warning: no num_particles_per_bunch_after_optimization provided in the config"
                " file. Using the one from the configuration before optimization."
            )
            self.num_particles_per_bunch = float(
                self.configuration["config_beambeam"]["num_particles_per_bunch"]
            )

        self.nemitt_x = self.configuration["config_beambeam"]["nemitt_x"]
        self.nemitt_y = self.configuration["config_beambeam"]["nemitt_y"]
        self.sigma_z = self.configuration["config_beambeam"]["sigma_z"]

    def load_filling_scheme_arrays(self):
        """Load the filling scheme arrays (two boolean arrays representing the buckets in the two
        beams) from a json file (whose path is in the configuration file)."""
        # Then get the filling scheme path (should already be an absolute path)
        self.path_filling_scheme = self.configuration["config_beambeam"][
            "mask_with_filling_pattern"
        ]["pattern_fname"]

        # Load the arrays
        with open(self.path_filling_scheme) as fid:
            filling_scheme = json.load(fid)

        self.array_b1 = np.array(filling_scheme["beam1"])
        self.array_b2 = np.array(filling_scheme["beam2"])

        # Get the bunches selected for tracking
        self.i_bunch_b1 = self.configuration["config_beambeam"]["mask_with_filling_pattern"][
            "i_bunch_b1"
        ]
        self.i_bunch_b2 = self.configuration["config_beambeam"]["mask_with_filling_pattern"][
            "i_bunch_b2"
        ]

    def return_number_of_collisions(self, IP=1):
        """Computes and returns the number of collisions at the requested IP."""
        if self._attributes_defined is None:
            raise ValueError(
                "No configuration has been provided when instantiating the ColliderCheck object."
            )
        # Assert that the arrays have the required length, and do the convolution
        assert len(self.array_b1) == len(self.array_b2) == 3564
        if IP == 1 or IP == 5:
            return self.array_b1 @ self.array_b2
        elif IP == 2:
            return np.roll(self.array_b1, -891) @ self.array_b2
        elif IP == 8:
            return np.roll(self.array_b1, -2670) @ self.array_b2
        else:
            raise ValueError("IP must be either 1, 2, 5 or 8.")

    def return_luminosity(self, IP=1, crab=False):
        """Computes and returns the luminosity at the requested IP. External twiss (e.g. from before
        beam-beam) can be provided."""
        if self._attributes_defined is None:
            raise ValueError(
                "No configuration has been provided when instantiating the ColliderCheck object."
            )

        if IP not in [1, 2, 5, 8]:
            raise ValueError("IP must be either 1, 2, 5 or 8.")
        n_col = self.return_number_of_collisions(IP=IP)
        luminosity = xt.lumi.luminosity_from_twiss(
            n_colliding_bunches=n_col,
            num_particles_per_bunch=self.num_particles_per_bunch,
            ip_name="ip" + str(IP),
            nemitt_x=self.nemitt_x,
            nemitt_y=self.nemitt_y,
            sigma_z=self.sigma_z,
            twiss_b1=self.tw_b1,
            twiss_b2=self.tw_b2,
            crab=crab,
        )
        return luminosity

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
        if self.configuration is not None:
            polarity_alice = self.configuration["config_knobs_and_tuning"]["knob_settings"][
                "on_alice_normalized"
            ]
            polarity_lhcb = self.configuration["config_knobs_and_tuning"]["knob_settings"][
                "on_lhcb_normalized"
            ]
        else:
            print(
                "Warning: no configuration provided when instantiating the ColliderCheck object to"
                " compute Alice and LHCb polarities."
            )
            polarity_alice = None
            polarity_lhcb = None
        return polarity_alice, polarity_lhcb

    # ! This needs to be updated
    # def return_normalized_separation(self, IP):
    #     """Returns the normalized separation at the requested IP."""
    #     if IP == 1:
    #         xing = float(self.tw_b1.rows[f"ip{IP}"]["px"])
    #         beta = float(self.tw_b1.rows[f"ip{IP}"]["bety"])
    #         sep = xing * np.sqrt(beta / self.nemitt_x)
    #     elif IP == 2:
    #         # ! Should I take py?
    #         xing = float(self.tw_b1.rows[f"ip{IP}"]["px"])
    #         beta = float(self.tw_b1.rows[f"ip{IP}"]["bety"])
    #         sep = xing * np.sqrt(beta / self.nemitt_x)
    #     elif IP == 5:
    #         xing = float(self.tw_b1.rows[f"ip{IP}"]["py"])
    #         beta = float(self.tw_b1.rows[f"ip{IP}"]["betx"])
    #         sep = xing * np.sqrt(beta / self.nemitt_y)
    #     elif IP == 8:
    #         xing = float(self.tw_b1.rows[f"ip{IP}"]["py"])
    #         beta = float(self.tw_b1.rows[f"ip{IP}"]["betx"])
    #         sep = xing * np.sqrt(beta / self.nemitt_y)
    #     return sep

    def output_check_as_txt(self, path_output="./check.txt"):
        str_file = ""

        # Check tune and chromaticity
        qx_b1, dqx_b1, qy_b1, dqy_b1 = self.return_tune_and_chromaticity(beam=1)
        qx_b2, dqx_b2, qy_b2, dqy_b2 = self.return_tune_and_chromaticity(beam=2)
        str_file += "Tune and chromaticity\n"
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

        # # Check normalized separation
        # sep1 = self.return_normalized_separation(IP=1)
        # sep2 = self.return_normalized_separation(IP=2)
        # sep5 = self.return_normalized_separation(IP=5)
        # sep8 = self.return_normalized_separation(IP=8)
        # str_file += "Normalized separation\n"
        # str_file += f"sep1 = {sep1:.4f}, sep2 = {sep2:.4f}, sep5 = {sep5:.4f}, sep8 = {sep8:.4f}\n"

        # str_file += "\n\n"

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

        # Write to file
        with open(path_output, "w") as fid:
            fid.write(str_file)


# if __name__ == "__main__":

#     # Do Twiss check
#     twiss_check = ColliderCheck(path_config, collider=build_collider.collider)
#     twiss_check.output_check_as_txt()
