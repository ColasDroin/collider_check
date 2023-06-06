#### Imports
import os
import numpy as np
import yaml
import json
import xtrack as xt
import xmask as xm
import xmask.lhc as xlhc


#### Build collider class
class BuildCollider:
    def __init__(self, path_configuration):
        """Initialize the BuildCollider class."""

        # Configuration path
        self.path_configuration = path_configuration

        # Load configuration
        self.configuration = self.load_configuration()

        # Correct path in configuration
        self.correct_configuration()

        # Load and tune collider
        self.collider = self.load_and_tune_collider()

    def load_configuration(self):
        """Loads the configuration from a yaml file."""
        with open(self.path_configuration, "r") as fid:
            configuration = yaml.safe_load(fid)
        return configuration

    def correct_configuration(self):
        """Corrects the paths in the configuration file."""
        self.configuration["config_simulation"]["collider_file"] = (
            self.path_configuration.split("config.yaml")[0]
            + self.configuration["config_simulation"]["collider_file"]
        )

        for lhcb in ["lhcb1", "lhcb2"]:
            self.configuration["config_collider"]["config_knobs_and_tuning"][
                "closed_orbit_correction"
            ][lhcb] = (
                self.path_configuration.split("config.yaml")[0]
                + self.configuration["config_collider"]["config_knobs_and_tuning"][
                    "closed_orbit_correction"
                ][lhcb]
            )

    def load_and_tune_collider(self):
        """Loads and tune the collider as done in the 2_tune_and_track script."""

        # Get configuration
        config_sim, config_collider = (
            self.configuration["config_simulation"],
            self.configuration["config_collider"],
        )
        # ==================================================================================================
        # --- Rebuild collider
        # ==================================================================================================
        # Load collider and build trackers
        collider = xt.Multiline.from_json(config_sim["collider_file"])

        # ==================================================================================================
        # --- Install beam-beam
        # ==================================================================================================
        config_bb = config_collider["config_beambeam"]

        # Install beam-beam lenses (inactive and not configured)
        collider.install_beambeam_interactions(
            clockwise_line="lhcb1",
            anticlockwise_line="lhcb2",
            ip_names=["ip1", "ip2", "ip5", "ip8"],
            delay_at_ips_slots=[0, 891, 0, 2670],
            num_long_range_encounters_per_side=config_bb["num_long_range_encounters_per_side"],
            num_slices_head_on=config_bb["num_slices_head_on"],
            harmonic_number=35640,
            bunch_spacing_buckets=config_bb["bunch_spacing_buckets"],
            sigmaz=config_bb["sigma_z"],
        )

        # ==================================================================================================
        # ---Knobs and tuning
        # ==================================================================================================
        # Build trackers
        collider.build_trackers()

        # Read knobs and tuning settings from config file
        conf_knobs_and_tuning = config_collider["config_knobs_and_tuning"]

        # Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
        # experimental magnets, etc.)
        for kk, vv in conf_knobs_and_tuning["knob_settings"].items():
            collider.vars[kk] = vv

        # Tunings
        for line_name in ["lhcb1", "lhcb2"]:
            knob_names = conf_knobs_and_tuning["knob_names"][line_name]

            targets = {
                "qx": conf_knobs_and_tuning["qx"][line_name],
                "qy": conf_knobs_and_tuning["qy"][line_name],
                "dqx": conf_knobs_and_tuning["dqx"][line_name],
                "dqy": conf_knobs_and_tuning["dqy"][line_name],
            }

            xm.machine_tuning(
                line=collider[line_name],
                enable_closed_orbit_correction=True,
                enable_linear_coupling_correction=True,
                enable_tune_correction=True,
                enable_chromaticity_correction=True,
                knob_names=knob_names,
                targets=targets,
                line_co_ref=collider[line_name + "_co_ref"],
                co_corr_config=conf_knobs_and_tuning["closed_orbit_correction"][line_name],
            )

        # ==================================================================================================
        # --- Compute the number of collisions in the different IPs (used for luminosity leveling)
        # ==================================================================================================

        # Get the filling scheme path (in json or csv format)
        filling_scheme_path = config_bb["mask_with_filling_pattern"]["pattern_fname"]

        # Load the filling scheme
        if filling_scheme_path.endswith(".json"):
            with open(filling_scheme_path, "r") as fid:
                filling_scheme = json.load(fid)
        else:
            raise ValueError(
                f"Unknown filling scheme file format: {filling_scheme_path}. It you provided a csv"
                " file, it should have been automatically convert when running the script"
                " 001_make_folders.py. Something went wrong."
            )

        # Extract booleans beam arrays
        array_b1 = np.array(filling_scheme["beam1"])
        array_b2 = np.array(filling_scheme["beam2"])

        # Assert that the arrays have the required length, and do the convolution
        assert len(array_b1) == len(array_b2) == 3564
        n_collisions_ip1_and_5 = array_b1 @ array_b2
        n_collisions_ip2 = np.roll(array_b1, -891) @ array_b2
        n_collisions_ip8 = np.roll(array_b1, -2670) @ array_b2

        # ==================================================================================================
        # ---Levelling
        # ==================================================================================================
        if "config_lumi_leveling" in config_collider and not config_collider["skip_leveling"]:
            # Read knobs and tuning settings from config file (already updated with the number of collisions)
            config_lumi_leveling = config_collider["config_lumi_leveling"]

            # Update the number of bunches in the configuration file
            config_lumi_leveling["ip8"]["num_colliding_bunches"] = int(n_collisions_ip8)

            # Level luminosity
            xlhc.luminosity_leveling(
                collider, config_lumi_leveling=config_lumi_leveling, config_beambeam=config_bb
            )

            # Re-match tunes, and chromaticities
            for line_name in ["lhcb1", "lhcb2"]:
                knob_names = conf_knobs_and_tuning["knob_names"][line_name]
                targets = {
                    "qx": conf_knobs_and_tuning["qx"][line_name],
                    "qy": conf_knobs_and_tuning["qy"][line_name],
                    "dqx": conf_knobs_and_tuning["dqx"][line_name],
                    "dqy": conf_knobs_and_tuning["dqy"][line_name],
                }
                xm.machine_tuning(
                    line=collider[line_name],
                    enable_tune_correction=True,
                    enable_chromaticity_correction=True,
                    knob_names=knob_names,
                    targets=targets,
                )

        else:
            print(
                "No leveling is done as no configuration has been provided, or skip_leveling"
                " is set to True."
            )

        # ==================================================================================================
        # --- Add linear coupling and rematch tune and chromaticity
        # ==================================================================================================

        # Add linear coupling as the target in the tuning of the base collider was 0
        # (not possible to set it the target to 0.001 for now)
        # ! This is commented as this affects the tune/chroma too much
        # ! We need to wait for the possibility to set the linear coupling as a target along with tune/chroma
        # collider.vars["c_minus_re_b1"] += conf_knobs_and_tuning["delta_cmr"]
        # collider.vars["c_minus_re_b2"] += conf_knobs_and_tuning["delta_cmr"]

        # Rematch tune and chromaticity
        for line_name in ["lhcb1", "lhcb2"]:
            knob_names = conf_knobs_and_tuning["knob_names"][line_name]
            targets = {
                "qx": conf_knobs_and_tuning["qx"][line_name],
                "qy": conf_knobs_and_tuning["qy"][line_name],
                "dqx": conf_knobs_and_tuning["dqx"][line_name],
                "dqy": conf_knobs_and_tuning["dqy"][line_name],
            }
            xm.machine_tuning(
                line=collider[line_name],
                enable_tune_correction=True,
                enable_chromaticity_correction=True,
                enable_linear_coupling_correction=False,
                knob_names=knob_names,
                targets=targets,
            )

        # ==================================================================================================
        # --- Assert that tune, chromaticity and linear coupling are correct before going further
        # ==================================================================================================
        for line_name in ["lhcb1", "lhcb2"]:
            tw = collider[line_name].twiss()
            assert np.isclose(tw.qx, conf_knobs_and_tuning["qx"][line_name], atol=1e-4), (
                f"tune_x is not correct for {line_name}. Expected"
                f" {conf_knobs_and_tuning['qx'][line_name]}, got {tw.qx}"
            )
            assert np.isclose(tw.qy, conf_knobs_and_tuning["qy"][line_name], atol=1e-4), (
                f"tune_y is not correct for {line_name}. Expected"
                f" {conf_knobs_and_tuning['qy'][line_name]}, got {tw.qy}"
            )
            assert np.isclose(
                tw.dqx,
                conf_knobs_and_tuning["dqx"][line_name],
                rtol=1e-2,
            ), (
                f"chromaticity_x is not correct for {line_name}. Expected"
                f" {conf_knobs_and_tuning['dqx'][line_name]}, got {tw.dqx}"
            )
            assert np.isclose(
                tw.dqy,
                conf_knobs_and_tuning["dqy"][line_name],
                rtol=1e-2,
            ), (
                f"chromaticity_y is not correct for {line_name}. Expected"
                f" {conf_knobs_and_tuning['dqy'][line_name]}, got {tw.dqy}"
            )
        # ! Commented as the linear coupling is not optimized anymore
        # ! This should be updated when possible
        # assert np.isclose(
        #     tw.c_minus,
        #     conf_knobs_and_tuning["delta_cmr"],
        #     atol=5e-3,
        # ), (
        #     f"linear coupling is not correct for {line_name}. Expected"
        #     f" {conf_knobs_and_tuning['delta_cmr']}, got {tw.c_minus}"
        # )
        return collider

    def dump_collider(self, prefix=None, suffix="collider.json"):
        """Dumps the collider to a json file."""
        path_collider = (
            self.path_configuration.split("/scans/")[1]
            .split("config.yaml")[0]
            .replace("/", "_")[:-5]
        )
        if prefix is not None:
            path_collider = prefix + path_collider + suffix
        self.collider.to_json(path_collider)
        return path_collider


#### Twiss Check class
class TwissCheck:
    def __init__(
        self,
        path_configuration,
        path_collider=None,
        collider=None,
    ):
        """Initialize the TwissCheck class."""

        # If a path has been provided, load the collider
        if path_collider is not None:
            self.load_collider_from_path(path_collider)

        elif collider is not None:
            # We assume the tracker is already built in this case
            self.collider = collider

        # Configuration path
        self.path_configuration = path_configuration

        # Load filling scheme
        self.array_b1, self.array_b2 = self.load_filling_scheme_arrays()

        # Load collider and twiss dataframes
        self.load_collider_and_twiss()

    def load_collider_from_path(self, path_collider):
        """Loads the collider from a json file."""
        # Load collider
        self.collider = xt.Multiline.from_json(path_collider)

        # Build trackers
        self.collider.build_trackers()

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

    def load_collider_and_twiss(self):
        """Returns the collider, along with the corresponding survey and twiss dataframes."""
        if self.collider is not None:
            # Get twiss and survey dataframes for both beams
            self.tw_b1, self.df_sv_b1, self.df_tw_b1 = (
                self.return_survey_and_twiss_dataframes_from_line(beam=1)
            )
            self.tw_b2, self.df_sv_b2, self.df_tw_b2 = (
                self.return_survey_and_twiss_dataframes_from_line(beam=2)
            )

            # Get luminosity configuration
            self.num_particles_per_bunch, self.nemitt_x, self.nemitt_y, self.sigma_z = (
                self.load_configuration_luminosity()
            )
        else:
            raise ValueError("No collider has been provided.")

    def return_survey_and_twiss_dataframes_from_line(self, beam=1):
        """Returns the survey and twiss dataframes from a collider line."""

        if beam == 1:
            line = self.collider.lhcb1
        elif beam == 2:
            line = self.collider.lhcb2
        else:
            raise ValueError("Beam must be either 1 or 2.")

        # Get survey dataframes
        df_sv = line.survey().to_pandas()

        # Get Twiss dataframes
        tw = line.twiss()
        df_tw = tw.to_pandas()

        return tw, df_sv, df_tw

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

    def return_luminosity(self, IP=1):
        """Computes and returns the luminosity at the requested IP."""
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
            crab=False,
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
        return (
            self.collider.vars["on_sep8h"]._value,
            self.collider.vars["on_sep8v"]._value,
            self.collider.vars["on_sep2"]._value,
        )

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
    path_config = "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/scans/opt_flathv_75_1500_withBB_chroma5_1p4_eol_bunch_scan/base_collider/xtrack_0001/config.yaml"
    build_collider = BuildCollider(path_config)

    # Dump collider
    path_collider = build_collider.dump_collider()

    # Do Twiss check
    twiss_check = TwissCheck(
        path_config, collider=build_collider.collider
    )  # path_collider=path_collider)
    twiss_check.output_check_as_txt()
