#### Imports
import os
import numpy as np
import yaml
import json
import xtrack as xt

#### Twiss Check class
class TwissCheck:
    def __init__(self, path_collider, path_configuration, path_filling_scheme = None):
        """Initialize the TwissCheck class."""
        
        # Collider path
        self.path_collider = path_collider
        
        # Configuration path
        self.path_configuration = path_configuration
        
        # Load collider and twiss dataframes
        self.load_collider_and_twiss()
        
        # Load filling scheme
        self.path_filling_scheme = path_filling_scheme
        if self.path_filling_scheme is not None:
            self.array_b1, self.array_b2 = self.load_filling_scheme_arrays()
        
    def load_collider_and_twiss(self):
        """Returns the collider, along with the corresponding survey and twiss dataframes."""
        if self.path_collider is not None:
            
            # Load collider
            self.collider = xt.Multiline.from_json(self.path_collider)
            
            # Build trackers
            self.collider.build_trackers()
            
            # Get twiss and survey dataframes for both beams
            self.tw_b1, self.df_sv_b1, self.df_tw_b1 = self.return_survey_and_twiss_dataframes_from_line(beam = 1)
            self.tw_b2, self.df_sv_b2, self.df_tw_b2 = self.return_survey_and_twiss_dataframes_from_line(beam = 2)
    
            # Get luminosity configuration
            self.num_particles_per_bunch, self.nemitt_x, self.nemitt_y, self.sigma_z = self.load_configuration_luminosity()
    
    def return_survey_and_twiss_dataframes_from_line(self, beam = 1):
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
            num_particles_per_bunch = float(configuration["config_beambeam"]["num_particles_per_bunch"])
            nemitt_x = configuration["config_beambeam"]["nemitt_x"] 
            nemitt_y = configuration["config_beambeam"]["nemitt_y"] 
            sigma_z = configuration["config_beambeam"]["sigma_z"]
        return num_particles_per_bunch, nemitt_x, nemitt_y, sigma_z
    
    def load_filling_scheme_arrays(self):
        """Returns two boolean arrays representing the buckets in the two beams, from a given 
        filling scheme."""
        if self.path_filling_scheme is not None:
            with open(self.path_filling_scheme) as fid:
                filling_scheme = json.load(fid)

            array_b1 = np.array(filling_scheme["beam1"])
            array_b2 = np.array(filling_scheme["beam2"])
            
            return array_b1, array_b2
        else:
            raise ValueError("No filling scheme path provided.")
    
    def return_number_of_collisions(self, IP = 1):
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
    
    def return_luminosity(self, IP = 1):
        """Computes and returns the luminosity at the requested IP."""
        if IP not in [1,2,5,8]:
            raise ValueError("IP must be either 1, 2, 5 or 8.")
        n_col = self.return_number_of_collisions(IP = IP)
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

    def return_twiss_at_ip(self, beam = 1, ip = 1):
        """Returns the twiss parameters, position and angle at the requested IP."""
        if beam == 1:
            return self.tw_b1.rows[f"ip{ip}"].cols["s", "x", "px", "y", "py", "betx", "bety", "dx", "dy"].to_pandas()
        elif beam == 2:
            return self.tw_b2.rows[f"ip{ip}"].cols["s", "x", "px", "y", "py", "betx", "bety", "dx", "dy"].to_pandas()
        else:
            raise ValueError("Beam must be either 1 or 2.")
    
    def return_tune_and_chromaticity(self, beam = 1):
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
        return self.collider.vars['on_sep8h']._value,  self.collider.vars['on_sep8v']._value, self.collider.vars['on_sep2']._value

    def return_normalized_separation(self, IP):
        """Returns the normalized separation at the requested IP."""
        # ! Need to find a more elegant way than .to_pandas().to_numpy().squeeze()[1]
        if IP == 1:
            xing = self.tw_b1.rows[f"ip{IP}"].cols["px"].to_pandas().to_numpy().squeeze()[1]
            beta = self.tw_b1.rows[f"ip{IP}"].cols["bety"].to_pandas().to_numpy().squeeze()[1]
            sep = xing * np.sqrt(beta/self.nemitt_x)
        elif IP == 2:
            # ! Should I take py?
            xing = self.tw_b1.rows[f"ip{IP}"].cols["py"].to_pandas().to_numpy().squeeze()[1]
            beta = self.tw_b1.rows[f"ip{IP}"].cols["betx"].to_pandas().to_numpy().squeeze()[1]
            sep = xing * np.sqrt(beta/self.nemitt_y)
        elif IP == 5:
            xing = self.tw_b1.rows[f"ip{IP}"].cols["py"].to_pandas().to_numpy().squeeze()[1]
            beta = self.tw_b1.rows[f"ip{IP}"].cols["betx"].to_pandas().to_numpy().squeeze()[1]
            sep = xing * np.sqrt(beta/self.nemitt_y)
        elif IP == 8:
            xing = self.tw_b1.rows[f"ip{IP}"].cols["py"].to_pandas().to_numpy().squeeze()[1]
            beta = self.tw_b1.rows[f"ip{IP}"].cols["betx"].to_pandas().to_numpy().squeeze()[1]
            sep = xing * np.sqrt(beta/self.nemitt_y)
        return sep
    
    def output_check_as_txt(self, path_output = './check.txt'):
        str_file = ''
        
        # Check tune and chromaticity
        qx_b1, dqx_b1, qy_b1, dqy_b1 = self.return_tune_and_chromaticity(beam = 1)
        qx_b2, dqx_b2, qy_b2, dqy_b2 = self.return_tune_and_chromaticity(beam = 2)
        str_file += "Tune and chromaticity\n"
        str_file += f"Qx_b1 = {qx_b1:.4f}, Qy_b1 = {qy_b1:.4f}, dQx_b1 = {dqx_b1:.4f}, dQy_b1 = {dqy_b1:.4f}\n"
        
        # Check linear coupling
        c_minus_b1, c_minus_b2 = self.return_linear_coupling()
        str_file += "Linear coupling\n"
        str_file += f"C- b1 = {c_minus_b1:.4f}, C- b2 = {c_minus_b2:.4f}\n"
        
        # Check momentum compaction factor
        alpha_p_b1, alpha_p_b2 = self.return_momentum_compaction_factor()
        str_file += "Momentum compaction factor\n"
        str_file += f"alpha_p b1 = {alpha_p_b1:.4f}, alpha_p b2 = {alpha_p_b2:.4f}\n"
        
        # Check separation knobs
        sep8h, sep8v, sep2 = self.return_separation_knobs()
        str_file += "Separation knobs\n"
        str_file += f"sep8h = {sep8h:.4f}, sep8v = {sep8v:.4f}, sep2 = {sep2:.4f}\n"
        
        # Check normalized separation
        sep1 = self.return_normalized_separation(IP = 1)
        sep2 = self.return_normalized_separation(IP = 2)
        sep5 = self.return_normalized_separation(IP = 5)
        sep8 = self.return_normalized_separation(IP = 8)
        str_file += "Normalized separation\n"
        str_file += f"sep1 = {sep1:.4f}, sep2 = {sep2:.4f}, sep5 = {sep5:.4f}, sep8 = {sep8:.4f}\n"
        
        # Check number of collisions
        n_col1 = self.return_number_of_collisions(IP = 1)
        n_col2 = self.return_number_of_collisions(IP = 2)
        n_col5 = self.return_number_of_collisions(IP = 5)
        n_col8 = self.return_number_of_collisions(IP = 8)
        str_file += "Number of collisions\n"
        str_file += f"n_col1 = {n_col1:.4f}, n_col2 = {n_col2:.4f}, n_col5 = {n_col5:.4f}, n_col8 = {n_col8:.4f}\n"
        
        # Check luminosity
        lumi1 = self.return_luminosity(IP = 1)
        lumi2 = self.return_luminosity(IP = 2)
        lumi5 = self.return_luminosity(IP = 5)
        lumi8 = self.return_luminosity(IP = 8)
        str_file += "Luminosity\n"
        str_file += f"lumi1 = {lumi1:.4f}, lumi2 = {lumi2:.4f}, lumi5 = {lumi5:.4f}, lumi8 = {lumi8:.4f}\n"
        
        # Check twiss observables at all IPs
        str_file += "Twiss observables\n"
        for ip in [1,2,5,8]:
            tw_b1 = self.return_twiss_at_ip(beam = 1, ip = ip)
            tw_b2 = self.return_twiss_at_ip(beam = 2, ip = ip)
            str_file += f"IP{ip}\n"
            str_file += f"b1: {tw_b1}\n"
            str_file += f"b2: {tw_b2}\n"
            
        # Write to file
        with open(path_output, 'w') as fid:
            fid.write(str_file)
            
        
    
if __name__ == '__main__':
    path_collider = "/afs/cern.ch/work/c/cdroin/private/comparison_pymask_xmask/xmask/xsuite_lines/collider_03_tuned_and_leveled_bb_off.json"
    path_configuration = "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/scans/opt_flathv_75_1500_withBB_chroma5_1p4_eol_bunch_scan/base_collider/xtrack_0002/config.yaml"
    path_filling_scheme =  "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/master_jobs/filling_scheme/8b4e_1972b_1960_1178_1886_224bpi_12inj_800ns_bs200ns.json"
    check = TwissCheck(path_collider, path_configuration, path_filling_scheme)
    check.output_check_as_txt()