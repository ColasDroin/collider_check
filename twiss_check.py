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
        if self.collider_path is not None:
            
            # Load collider
            self.collider = xt.Multiline.from_json(self.collider_path)
            
            # Build trackers
            self.collider.build_trackers()
            
            # Get twiss and survey dataframes for both beams
            self.tw_b1, self.df_sv_b1, self.df_tw_b1 = self.return_survey_and_twiss_dataframes_from_line(beam = 1)
            self.tw_b2, self.df_sv_b2, self.df_tw_b2 = self.return_survey_and_twiss_dataframes_from_line(beam = 2)
    
    
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
        num_particles_per_bunch, nemitt_x, nemitt_y, sigma_z = self.load_configuration_luminosity()
        n_col = self.compute_number_of_collisions(IP = IP)
        luminosity = xt.lumi.luminosity_from_twiss(
                n_colliding_bunches=n_col,
                num_particles_per_bunch=num_particles_per_bunch,
                ip_name="ip" + str(IP),
                nemitt_x=nemitt_x,
                nemitt_y=nemitt_y,
                sigma_z=sigma_z,
                twiss_b1=self.tw_b1,
                twiss_b2=self.tw_b2,
                crab=False,
            )
        return luminosity

    def get_beta_functions(self, beam, plane):
        pass    
    
if __name__ == '__main__':
    path_collider = "/afs/cern.ch/work/c/cdroin/private/comparison_pymask_xmask/xmask/xsuite_lines/collider_03_tuned_and_leveled_bb_off.json"
    path_configuration = "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/scans/opt_flathv_75_1500_withBB_chroma5_1p4_eol_bunch_scan/base_collider/xtrack_0002/config.yaml"
    path_filling_scheme =  "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/master_jobs/filling_scheme/8b4e_1972b_1960_1178_1886_224bpi_12inj_800ns_bs200ns.json"