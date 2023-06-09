#### Imports
import os
import numpy as np
import yaml
import json
import sys
import importlib
import pickle
import xtrack as xt
import xmask as xm
import xmask.lhc as xlhc


#### Build collider class
class BuildCollider:
    def __init__(self, path_configuration, save_collider_before_bb=True):
        """Initialize the BuildCollider class."""

        # Configuration path
        self.path_configuration = path_configuration

        # Load configuration
        self.configuration = self.load_configuration()

        # Correct path in configuration
        self.correct_configuration()

        # Load and tune collider
        self.collider, self.collider_before_bb = self.load_and_tune_collider(
            save_collider_before_bb
        )

    def load_configuration(self):
        """Loads the configuration from a yaml file."""
        with open(self.path_configuration, "r") as fid:
            configuration = yaml.safe_load(fid)
        return configuration

    def correct_configuration(self):
        """Corrects the paths in the configuration file (from relative to absolute)."""
        self.configuration["config_simulation"]["collider_file"] = (
            self.path_configuration.split("config.yaml")[0]
            + self.configuration["config_simulation"]["collider_file"]
        )

        # ? Correction of the configuration files is not required anymore as they're produced locally
        # for lhcb in ["lhcb1", "lhcb2"]:
        #     self.configuration["config_collider"]["config_knobs_and_tuning"][
        #         "closed_orbit_correction"
        #     ][lhcb] = (
        #         self.path_configuration.split("config.yaml")[0]
        #         + self.configuration["config_collider"]["config_knobs_and_tuning"][
        #             "closed_orbit_correction"
        #         ][lhcb]
        #     )

    def load_and_tune_collider(self, save_collider_before_bb):
        """Build the collider using the same script as in the initial configuration file."""

        # Path of the 2_configure_and_track file
        path_configure_and_track = self.path_configuration.split("config.yaml")[0]
        name_module = "2_configure_and_track.py"
        # Check that the twiss and track file exists
        if not os.path.exists(path_configure_and_track + name_module):
            raise ValueError(
                "The 2_configure_and_track file does not exist in the same directory as the config"
                " file. No collider can be built ensuring reproducibility."
            )
        else:
            # Add to sys
            sys.path.insert(1, path_configure_and_track)

            # Import the module
            configure_and_track = importlib.import_module("2_configure_and_track")

        # Build collider
        if save_collider_before_bb:
            collider, _, collider_before_bb = configure_and_track.configure_collider(
                self.configuration["config_simulation"],
                self.configuration["config_collider"],
                save_collider=False,
                return_collider_before_bb=save_collider_before_bb,
            )
        else:
            collider, _ = configure_and_track.configure_collider(
                self.configuration["config_simulation"],
                self.configuration["config_collider"],
                save_collider=False,
                return_collider_before_bb=save_collider_before_bb,
            )
            collider_before_bb = None

        # Remove the folder "correction" which was created during the process
        os.system("rm -rf correction")
        # Remove other temporary files
        os.system("rm -rf .__*")

        return collider, collider_before_bb

    def dump_collider(self, prefix=None, suffix="collider.json", dump_before_bb=False):
        """Dumps the collider to a json file."""
        path_collider = (
            self.path_configuration.split("/scans/")[1]
            .split("config.yaml")[0]
            .replace("/", "_")[:-5]
        )
        if prefix is not None:
            path_collider = prefix + path_collider + suffix
        self.collider.to_json(path_collider)

        if self.collider_before_bb is not None and dump_before_bb:
            path_collider_before_bb = path_collider.replace(".json", "_before_bb.json")
            self.collider_before_bb.to_json(path_collider_before_bb)
            return path_collider, path_collider_before_bb
        elif self.collider_before_bb is None and dump_before_bb:
            raise ValueError("No collider before beam-beam has been provided.")
        else:
            return path_collider


#### Twiss Check class
class TwissCheck:
    def __init__(
        self,
        path_configuration,
        path_collider=None,
        collider=None,
    ):
        """Initialize the TwissCheck class, either from a set of Twiss, or directly from a collider,
        or from a path to a collider."""

        # Store the paths and the collider (if existing)
        self.path_configuration = path_configuration
        self.path_collider = path_collider
        self.collider = collider

        # Check that either a collider or a path_to_collider has been provided
        if (self.collider is None and self.path_collider is None) or (
            self.collider is not None and self.path_collider is not None
        ):
            raise ValueError("Either a collider, or a path to a collider must be provided.")

        # If a path to collider has been provided, or a collider has been provided, load the collider
        if self.path_collider is not None:
            if path_collider is not None:
                self.collider = self.load_collider_from_path()

            # Load twiss and survey from collider
            self.tw_b1, self.df_sv_b1, self.df_tw_b1, self.tw_b2, self.df_sv_b2, self.df_tw_b2 = (
                self.load_twiss_from_collider()
            )

        # Get luminosity configuration
        self.num_particles_per_bunch, self.nemitt_x, self.nemitt_y, self.sigma_z = (
            self.load_configuration_luminosity()
        )

        # Load filling scheme
        self.array_b1, self.array_b2 = self.load_filling_scheme_arrays()

    def load_collider_from_path(self):
        """Loads the collider from a json file."""
        # Load collider
        collider = xt.Multiline.from_json(self.path_collider)

        # Build trackers
        collider.build_trackers()

        return collider

    def load_twiss_from_collider(self):
        """Returns the collider, along with the corresponding survey and twiss dataframes."""

        def return_survey_and_twiss_dataframes_from_line(collider, beam=1):
            """Returns the survey and twiss dataframes from a collider line."""

            if beam == 1:
                line = collider.lhcb1
            elif beam == 2:
                line = collider.lhcb2
            else:
                raise ValueError("Beam must be either 1 or 2.")

            # Get survey dataframes
            df_sv = line.survey().to_pandas()

            # Get Twiss dataframes
            tw = line.twiss()
            df_tw = tw.to_pandas()

            return tw, df_sv, df_tw

        if self.collider is not None:
            # Get twiss and survey dataframes for both beams
            tw_b1, df_sv_b1, df_tw_b1 = return_survey_and_twiss_dataframes_from_line(
                self.collider, beam=1
            )
            tw_b2, df_sv_b2, df_tw_b2 = return_survey_and_twiss_dataframes_from_line(
                self.collider, beam=2
            )

            return tw_b1, df_sv_b1, df_tw_b1, tw_b2, df_sv_b2, df_tw_b2
        else:
            raise ValueError("No collider has been provided.")

    def load_configuration_luminosity(self):
        """Returns the configuration file variables used to compute the luminosity."""
        with open(self.path_configuration, "r") as fid:
            configuration = yaml.safe_load(fid)["config_collider"]
            num_particles_per_bunch = float(
                configuration["config_beambeam"]["num_particles_per_bunch"]
            )
            nemitt_x = configuration["config_beambeam"]["nemitt_x"]
            nemitt_y = configuration["config_beambeam"]["nemitt_y"]
            sigma_z = configuration["config_beambeam"]["sigma_z"]
        return num_particles_per_bunch, nemitt_x, nemitt_y, sigma_z

    def load_filling_scheme_arrays(self):
        """Load the filling scheme arrays (two boolean arrays representing the buckets in the two
        beams) from a json file (whose path is in the configuration file)."""
        # First load the configuration file
        with open(self.path_configuration, "r") as fid:
            configuration = yaml.safe_load(fid)

        # Then get the filling scheme path (should already be an absolute path)
        path_filling_scheme = configuration["config_collider"]["config_beambeam"][
            "mask_with_filling_pattern"
        ]["pattern_fname"]

        # Load the arrays
        with open(path_filling_scheme) as fid:
            filling_scheme = json.load(fid)

        array_b1 = np.array(filling_scheme["beam1"])
        array_b2 = np.array(filling_scheme["beam2"])

        return array_b1, array_b2

    def return_number_of_collisions(self, IP=1):
        """Computes and returns the number of collisions at the requested IP."""
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

    def return_separation_knobs(self):
        """Returns the separation knobs at IP2 and IP8."""
        if self.collider is not None:
            return (
                self.collider.vars["on_sep8h"]._value,
                self.collider.vars["on_sep8v"]._value,
                self.collider.vars["on_sep2"]._value,
            )
        else:
            raise ValueError("No collider has been provided.")

    def return_normalized_separation(self, IP):
        """Returns the normalized separation at the requested IP."""
        if IP == 1:
            xing = float(self.tw_b1.rows[f"ip{IP}"]["px"])
            beta = float(self.tw_b1.rows[f"ip{IP}"]["bety"])
            sep = xing * np.sqrt(beta / self.nemitt_x)
        elif IP == 2:
            # ! Should I take py?
            xing = float(self.tw_b1.rows[f"ip{IP}"]["px"])
            beta = float(self.tw_b1.rows[f"ip{IP}"]["bety"])
            sep = xing * np.sqrt(beta / self.nemitt_x)
        elif IP == 5:
            xing = float(self.tw_b1.rows[f"ip{IP}"]["py"])
            beta = float(self.tw_b1.rows[f"ip{IP}"]["betx"])
            sep = xing * np.sqrt(beta / self.nemitt_y)
        elif IP == 8:
            xing = float(self.tw_b1.rows[f"ip{IP}"]["py"])
            beta = float(self.tw_b1.rows[f"ip{IP}"]["betx"])
            sep = xing * np.sqrt(beta / self.nemitt_y)
        return sep

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

        if self.collider is not None:
            # Check separation knobs
            sep8h, sep8v, sep2 = self.return_separation_knobs()
            str_file += "Separation knobs\n"
            str_file += f"sep8h = {sep8h:.4f}, sep8v = {sep8v:.4f}, sep2 = {sep2:.4f}\n"

            str_file += "\n\n"

        # Check normalized separation
        sep1 = self.return_normalized_separation(IP=1)
        sep2 = self.return_normalized_separation(IP=2)
        sep5 = self.return_normalized_separation(IP=5)
        sep8 = self.return_normalized_separation(IP=8)
        str_file += "Normalized separation\n"
        str_file += f"sep1 = {sep1:.4f}, sep2 = {sep2:.4f}, sep5 = {sep5:.4f}, sep8 = {sep8:.4f}\n"

        str_file += "\n\n"

        # Check luminosity
        lumi1 = self.return_luminosity(IP=1)
        lumi2 = self.return_luminosity(IP=2)
        lumi5 = self.return_luminosity(IP=5)
        lumi8 = self.return_luminosity(IP=8)
        str_file += "Luminosity\n"
        str_file += f"IP1 = {lumi1:.4e}, IP2 = {lumi2:.4e}, IP5 = {lumi5:.4e}, IP8 = {lumi8:.4e}\n"

        str_file += "\n\n"

        # Write to file
        with open(path_output, "w") as fid:
            fid.write(str_file)


if __name__ == "__main__":
    # Load collider
    path_config = "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/scans/opt_flathv_75_1500_withBB_chroma5_1p4_eol_tune_intensity/base_collider/xtrack_0000/config.yaml"
    build_collider = BuildCollider(path_config)

    # Dump collider
    # path_collider = build_collider.dump_collider()

    # Do Twiss check
    twiss_check = TwissCheck(
        path_config, collider=build_collider.collider
    )  # path_collider=path_collider)
    twiss_check.output_check_as_txt()
