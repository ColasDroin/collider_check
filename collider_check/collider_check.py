# ==================================================================================================
# --- Imports
# ==================================================================================================
import json
import logging
import os
import pathlib
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xtrack as xt
import yaml
from scipy import constants


# ==================================================================================================
# --- Class definition
# ==================================================================================================
class ColliderCheck:
    """
    ColliderCheck class for analyzing collider configurations and computing various collider observables.

    Attributes:
        energy (float): Beam energy in GeV.
        tw_b1 (pd.DataFrame): Twiss parameters for beam 1.
        sv_b1 (pd.DataFrame): Survey data for beam 1.
        tw_b2 (pd.DataFrame): Twiss parameters for beam 2.
        sv_b2 (pd.DataFrame): Survey data for beam 2.
        df_tw_b1 (pd.DataFrame): Twiss parameters for beam 1 in pandas DataFrame format.
        df_sv_b1 (pd.DataFrame): Survey data for beam 1 in pandas DataFrame format.
        df_tw_b2 (pd.DataFrame): Twiss parameters for beam 2 in pandas DataFrame format.
        df_sv_b2 (pd.DataFrame): Survey data for beam 2 in pandas DataFrame format.
        dic_survey_per_ip (Dict[str, Dict]): Dictionary to store survey data per interaction point.

    Methods:
        configuration: Property to get and set the configuration dictionary.
        type_particles: Property to get the type of particles.
        cross_section: Property to get the cross-section based on the type of particles.
        nemitt_x: Property to get the normalized emittance in x.
        nemitt_y: Property to get the normalized emittance in y.
        n_lr_per_side: Property to get the number of long-range encounters per side.
        return_number_of_collisions(IP: int = 1) -> int: Computes and returns the number of
            collisions at the requested IP.
        return_luminosity(IP: int = 1) -> float: Computes and returns the luminosity at the
            requested IP.
        return_twiss_at_ip(beam: int = 1, ip: int = 1) -> pd.DataFrame: Returns the twiss
            parameters, position, and angle at the requested IP.
        return_tune_and_chromaticity(beam: int = 1) -> Tuple[float, float, float, float]: Returns
            the tune and chromaticity for the requested beam.
        return_linear_coupling() -> Tuple[float, float]: Returns the linear coupling for the two
            beams.
        return_momentum_compaction_factor() -> Tuple[float, float]: Returns the momentum compaction
            factor for the two beams.
        return_polarity_ip_2_8() -> Tuple[float, float]: Returns the polarity (internal angle of
            the experiments) for IP2 and IP8.
        compute_separation_variables(ip: str = "ip1", beam_weak: str = "b1") -> Dict[str, Any]:
            Computes all the variables needed to compute the separation at the requested IP, in a
                weak-strong setting.
        return_dic_position_all_ips() -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]: Computes all
            the variables needed to determine the position of the beam in all IRs.
        plot_orbits(ip: str = "ip1", beam_weak: str = "b1") -> None: Plots the beams orbits at the
            requested IP.
        plot_separation(ip: str = "ip1", beam_weak: str = "b1") -> None: Plots the normalized
            separation at the requested IP.
        output_check_as_str(path_output: Optional[str] = None) -> str: Summarizes the collider
            observables in a string, and optionally writes to a file.
    """

    def __init__(
        self,
        collider: xt.Multiline,
        path_filling_scheme: Optional[str] = None,
        type_particles: Optional[str] = None,
    ) -> None:
        """
        Initialize the ColliderCheck class directly from a collider, potentially embedding a
        configuration file.

        Args:
            collider (xt.Multiline): The collider object.
            path_filling_scheme (Optional[str]): Path to the filling scheme file.
            type_particles (Optional[str]): Type of particles ('proton' or 'lead').
        """

        # Store the collider
        self.collider = collider

        # Store the filling scheme path
        self.path_filling_scheme = path_filling_scheme

        # Define the configuration through a property since it might not be there
        self._configuration = None
        self.configuration_str = None

        # Define the type of particles
        self._type_particles = type_particles

        # Beam energy
        self.energy = self.collider.lhcb1.particle_ref._p0c[0] / 1e9

        # Get twiss and survey dataframes for both beams
        self.tw_b1, self.sv_b1 = [self.collider.lhcb1.twiss(), self.collider.lhcb1.survey()]
        self.tw_b2, self.sv_b2 = [self.collider.lhcb2.twiss(), self.collider.lhcb2.survey()]
        self.df_tw_b1, self.df_sv_b1 = [self.tw_b1.to_pandas(), self.sv_b1.to_pandas()]
        self.df_tw_b2, self.df_sv_b2 = [self.tw_b2.to_pandas(), self.sv_b2.to_pandas()]

        # Variables used to compute the separation (computed on the fly)
        self.dic_survey_per_ip = {"lhcb1": {}, "lhcb2": {}}

        # Try to load arrays for filling scheme
        if self.path_filling_scheme is None and self.configuration is not None:
            self._load_filling_scheme_arrays()

    @property
    def configuration(self) -> Optional[Dict]:
        """
        Loads the configuration, as well as the luminosity and filling scheme arrays, if they're not
        already loaded.

        Returns:
            Optional[Dict]: The configuration dictionary.
        """

        if self._configuration is not None:
            return self._configuration
        # Get the corresponding configuration if it's there
        if hasattr(self.collider, "metadata") and self.collider.metadata != {}:
            self.configuration = self.collider.metadata

        return self._configuration

    @configuration.setter
    def configuration(self, configuration_dict: Dict) -> None:
        """
        This function is used to update the configuration, and the attributes that depend on it.

        Args:
            configuration_dict (Dict): The new configuration dictionary.
        """
        self._configuration = configuration_dict
        self._update_attributes_configuration()

    def _raise_no_configuration_error(self) -> None:
        """Raises an error when no configuration is provided."""
        raise ValueError(
            "No configuration has been provided when instantiating the ColliderCheck object."
        )

    def _update_attributes_configuration(self) -> None:
        """Updates attributes based on the configuration."""

        if self.configuration is None:
            self._raise_no_configuration_error()

        # Store the configuration as a string
        self.configuration_str = yaml.dump(self.configuration)

        # Compute luminosity and filling schemes attributes
        self._load_configuration_luminosity()
        self._load_filling_scheme_arrays()

        # Clean cache for separation computation
        self.compute_separation_variables.cache_clear()

    @property
    def type_particles(self) -> str:
        """
        Returns the type of particles.

        Returns:
            str: The type of particles ('proton' or 'lead').
        """
        if self._type_particles is not None:
            if self._type_particles in ["proton", "lead"]:
                return self._type_particles
            else:
                raise ValueError("type_particles must be either 'proton' or 'lead'.")

        elif self.configuration is not None:
            if (
                "config_mad" not in self.configuration
                or "ions" not in self.configuration["config_mad"]
            ):
                raise ValueError(
                    "No type of particles provided by the user nor in the configuration. Please "
                    "provide it."
                )
            if self.configuration["config_mad"]["ions"]:
                self._type_particles = "lead"
            else:
                self._type_particles = "proton"
            return self._type_particles
        else:
            raise ValueError(
                "No type of particles provided by the user nor in the configuration. Please "
                "provide it."
            )

    @property
    def cross_section(self) -> float:
        """
        Returns the cross-section based on the type of particles.

        Returns:
            float: The cross-section value.
        """
        # Record cross-section correspondinlgy
        if self.type_particles == "proton":
            return 81e-27
        elif self.type_particles == "lead":
            return 281e-24
        else:
            raise ValueError("type_particles must be either 'proton' or 'lead'.")

    @property
    def nemitt_x(self) -> float:
        """
        Returns the normalized emittance in x.

        Returns:
            float: The normalized emittance in x.
        """
        if self.configuration is not None:
            return self.configuration["config_collider"]["config_beambeam"]["nemitt_x"]
        logging.warning("No configuration provided. Using default value of 2.2e-6 for nemitt_x.")
        return 2.2e-6

    @property
    def nemitt_y(self) -> float:
        """
        Returns the normalized emittance in y.

        Returns:
            float: The normalized emittance in y.
        """
        if self.configuration is not None:
            return self.configuration["config_collider"]["config_beambeam"]["nemitt_y"]
        logging.warning("No configuration provided. Using default value of 2.2e-6 for nemitt_y.")
        return 2.2e-6

    @property
    def n_lr_per_side(self) -> int:
        """
        Returns the number of long-range encounters per side.

        Returns:
            int: The number of long-range encounters per side.
        """

        if self.configuration is not None:
            return self.configuration["config_collider"]["config_beambeam"][
                "num_long_range_encounters_per_side"
            ]["ip1"]
        logging.warning("No configuration provided. Using default value of 16 for n_lr_per_side.")
        return 16

    def _load_configuration_luminosity(self) -> None:
        """Loads luminosity-related configuration parameters."""
        if self.configuration is None:
            self._raise_no_configuration_error()

        if (
            "final_num_particles_per_bunch"
            in self.configuration["config_collider"]["config_beambeam"]
        ):
            self.num_particles_per_bunch = float(
                self.configuration["config_collider"]["config_beambeam"][
                    "final_num_particles_per_bunch"
                ]
            )
        else:
            self.num_particles_per_bunch = float(
                self.configuration["config_collider"]["config_beambeam"]["num_particles_per_bunch"]
            )
        self.sigma_z = self.configuration["config_collider"]["config_beambeam"]["sigma_z"]

    def _load_filling_scheme_arrays(self) -> None:
        """Loads the filling scheme arrays."""
        if self.path_filling_scheme is None and self.configuration is not None:
            # Get the filling scheme path (should already be an absolute path)
            self.path_filling_scheme = self.configuration["config_collider"]["config_beambeam"][
                "mask_with_filling_pattern"
            ]["pattern_fname"]

            # Check if the file exists
            if not os.path.exists(self.path_filling_scheme):
                # Get parent of local path
                local_path = pathlib.Path(__file__).parent.parent.absolute()

                # Name folder data
                data_folder = "test_data"

                # Get the last part of the path filling scheme
                name_filling_scheme = self.path_filling_scheme.split("/")[-1]
                self.path_filling_scheme = os.path.join(
                    local_path, data_folder, name_filling_scheme
                )

                if not os.path.exists(self.path_filling_scheme):
                    raise FileNotFoundError(
                        f"File {self.path_filling_scheme} not found. Please provide a valid "
                        "path to the filling scheme."
                    )

        elif self.path_filling_scheme is None:
            raise ValueError(
                "No filling scheme path provided, and no configuration to get it from."
            )
        # Load the scheme (two boolean arrays representing the buckets in the two beams)
        with open(self.path_filling_scheme) as fid:
            filling_scheme = json.load(fid)

        self.array_b1 = np.array(filling_scheme["beam1"])
        self.array_b2 = np.array(filling_scheme["beam2"])

        # Get the bunches selected for tracking
        self.i_bunch_b1 = None
        self.i_bunch_b2 = None
        if self.configuration is not None:
            if (
                "i_bunch_b1"
                in self.configuration["config_collider"]["config_beambeam"][
                    "mask_with_filling_pattern"
                ]
            ):
                self.i_bunch_b1 = self.configuration["config_collider"]["config_beambeam"][
                    "mask_with_filling_pattern"
                ]["i_bunch_b1"]
            if (
                "i_bunch_b2"
                in self.configuration["config_collider"]["config_beambeam"][
                    "mask_with_filling_pattern"
                ]
            ):
                self.i_bunch_b2 = self.configuration["config_collider"]["config_beambeam"][
                    "mask_with_filling_pattern"
                ]["i_bunch_b2"]

        if self.i_bunch_b1 is None:
            logging.warning("No bunches selected for tracking in beam 1. Using first bunch.")
            self.i_bunch_b1 = np.where(self.array_b1)[0][0]
        if self.i_bunch_b2 is None:
            logging.warning("No bunches selected for tracking in beam 2. Using first bunch.")
            self.i_bunch_b2 = np.where(self.array_b2)[0][0]

    def return_number_of_collisions(self, IP: int = 1) -> int:
        """
        Computes and returns the number of collisions at the requested IP.

        Args:
            IP (int): The interaction point number (1, 2, 5, or 8).

        Returns:
            int: The number of collisions at the specified IP.
        """

        # Ensure configuration is defined
        if self.configuration is None:
            self._raise_no_configuration_error()

        # Assert that the arrays have the required length, and do the convolution
        assert len(self.array_b1) == len(self.array_b2) == 3564
        if IP in {1, 5}:
            return int(self.array_b1 @ self.array_b2)
        elif IP == 2:
            return int(np.roll(self.array_b1, 891) @ self.array_b2)
        elif IP == 8:
            return int(np.roll(self.array_b1, 2670) @ self.array_b2)
        else:
            raise ValueError("IP must be either 1, 2, 5 or 8.")

    def return_luminosity(self, IP: int = 1) -> float:
        """
        Computes and returns the luminosity at the requested IP.

        Args:
            IP (int): The interaction point number (1, 2, 5, or 8).

        Returns:
            float: The luminosity at the specified IP.
        """

        # Ensure configuration is defined
        if self.configuration is None:
            self._raise_no_configuration_error()

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

    def return_twiss_at_ip(self, beam: int = 1, ip: int = 1) -> pd.DataFrame:
        """
        Returns the twiss parameters, position and angle at the requested IP.

        Args:
            beam (int): The beam number (1 or 2).
            ip (int): The interaction point number.

        Returns:
            pd.DataFrame: A DataFrame containing twiss parameters at the specified IP.
        """
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

    def return_tune_and_chromaticity(self, beam: int = 1) -> Tuple[float, float, float, float]:
        """
        Returns the tune and chromaticity for the requested beam.

        Args:
            beam (int): The beam number (1 or 2).

        Returns:
            Tuple[float, float, float, float]: A tuple containing (qx, dqx, qy, dqy).
        """

        if beam == 1:
            return self.tw_b1["qx"], self.tw_b1["dqx"], self.tw_b1["qy"], self.tw_b1["dqy"]
        elif beam == 2:
            return self.tw_b2["qx"], self.tw_b2["dqx"], self.tw_b2["qy"], self.tw_b2["dqy"]
        else:
            raise ValueError("Beam must be either 1 or 2.")

    def return_linear_coupling(self) -> Tuple[float, float]:
        """
        Returns the linear coupling for the two beams.

        Returns:
            Tuple[float, float]: A tuple containing the linear coupling for beam 1 and beam 2.
        """
        return self.tw_b1["c_minus"], self.tw_b2["c_minus"]

    def return_momentum_compaction_factor(self) -> Tuple[float, float]:
        """
        Returns the momentum compaction factor for the two beams.

        Returns:
            Tuple[float, float]: A tuple containing the momentum compaction factor for beam 1 and
                beam 2.
        """
        return self.tw_b1["momentum_compaction_factor"], self.tw_b2["momentum_compaction_factor"]

    def return_polarity_ip_2_8(self) -> Tuple[float, float]:
        """
        Return the polarity (internal angle of the experiments) for IP2 and IP8.

        Returns:
            Tuple[float, float]: A tuple containing the polarity for IP2 and IP8.
        """
        # Ensure configuration is defined
        if self.configuration is None:
            self._raise_no_configuration_error()

        polarity_alice = self.configuration["config_collider"]["config_knobs_and_tuning"][
            "knob_settings"
        ]["on_alice_normalized"]
        polarity_lhcb = self.configuration["config_collider"]["config_knobs_and_tuning"][
            "knob_settings"
        ]["on_lhcb_normalized"]

        return polarity_alice, polarity_lhcb

    def _compute_ip_specific_separation(self, ip: str = "ip1", beam_weak: str = "b1") -> Tuple:
        """
        Compute IP-specific separation parameters.

        Args:
            ip (str): The interaction point name.
            beam_weak (str): The weak beam designation.

        Returns:
            Tuple: A tuple containing various separation parameters.
        """
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

    def _compute_emittances_separation(self) -> Tuple[float, float, float, float, float, float]:
        """
        Compute emittances and separation parameters.

        Returns:
            Tuple[float, float, float, float, float, float]: A tuple containing gamma_rel, beta_rel,
                and emittances.
        """
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
        twiss_filtered: Dict[str, pd.DataFrame],
        beam_weak: str,
        beam_strong: str,
        emittance_strong_x: float,
        emittance_strong_y: float,
        emittance_weak_x: float,
        emittance_weak_y: float,
        d_x_weak_strong_in_meter: np.ndarray,
        d_y_weak_strong_in_meter: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """
        Compute IP-specific normalized separation parameters.

        Args:
            twiss_filtered (Dict[str, pd.DataFrame]): Filtered twiss data for both beams.
            beam_weak (str): The weak beam designation.
            beam_strong (str): The strong beam designation.
            emittance_strong_x (float): Emittance in x for the strong beam.
            emittance_strong_y (float): Emittance in y for the strong beam.
            emittance_weak_x (float): Emittance in x for the weak beam.
            emittance_weak_y (float): Emittance in y for the weak beam.
            d_x_weak_strong_in_meter (np.ndarray): Separation in x between weak and strong beams in
                meters.
            d_y_weak_strong_in_meter (np.ndarray): Separation in y between weak and strong beams in
                meters.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
                A tuple containing the separation parameters.
        """
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
    def compute_separation_variables(
        self, ip: str = "ip1", beam_weak: str = "b1"
    ) -> Dict[str, Any]:
        """
        Computes all the variables needed to compute the separation at the requested IP, in a
        weak-strong setting. The variables are stored and returned in a dictionary.

        Args:
            ip (str): The interaction point name.
            beam_weak (str): The weak beam designation.

        Returns:
            Dict[str, any]: A dictionary containing the computed separation variables.
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

    def return_dic_position_all_ips(self) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Computes all the variables needed to determine the position of the beam in all IRs. The
        variables are stored and returned in a dictionary. The extreme positions are:

        IP1 : mqy.4l1.b1 to mqy.4r1.b1
        IP2 : mqy.b5l2.b1 to mqy.b4r2.b1
        IP5 : mqy.4l5.b1 to mqy.4r5.b1
        IP8 : mqy.b4l8.b1 to mqy.b4r8.b1

        Returns:
            Dict[str, Dict[str, Dict[str, pd.DataFrame]]]: A dictionary containing the positions
            of the beam in all IRs for both beams.
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

    def plot_orbits(self, ip: str = "ip1", beam_weak: str = "b1") -> None:
        """
        Plots the beams orbits at the requested IP.

        Args:
            ip (str): The interaction point name.
            beam_weak (str): The weak beam designation.
        """

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

    def plot_separation(self, ip: str = "ip1", beam_weak: str = "b1") -> None:
        """
        Plots the normalized separation at the requested IP.

        Args:
            ip (str): The interaction point name.
            beam_weak (str): The weak beam designation.
        """
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

    def output_check_as_str(self, path_output: Optional[str] = None) -> str:
        """
        Summarizes the collider observables in a string, and optionally writes to a file.

        Args:
            path_output (Optional[str]): The path to the output file. If None, the output is not
            written to a file.

        Returns:
            str: A string summarizing the collider observables.
        """
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
    # Run collider check with config
    path_collider = "../test_data/collider.json"
    collider = xt.Multiline.from_json(path_collider)
    collider_check = ColliderCheck(collider=collider)
    print(collider_check.output_check_as_str(path_output="../output/check.txt"))

    # Run collider check without config
    path_collider = "../test_data/collider_without_config.json"
    collider = xt.Multiline.from_json(path_collider)
    collider_check = ColliderCheck(collider=collider)
    print(collider_check.output_check_as_str(path_output="../output/check_without_config.txt"))
