# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Regional Integrated model of Climate and the Economy (RICE)
"""
import logging
import os
import sys

import numpy as np
from gym.spaces import MultiDiscrete

_PUBLIC_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [_PUBLIC_REPO_DIR] + sys.path

from rice_helpers import (
    get_abatement_cost,
    get_armington_agg,
    get_aux_m,
    get_capital,
    get_capital_depreciation,
    get_carbon_intensity,
    get_consumption,
    get_damages,
    get_exogenous_emissions,
    get_global_carbon_mass,
    get_global_temperature,
    get_gross_output,
    get_investment,
    get_labor,
    get_land_emissions,
    get_max_potential_exports,
    get_mitigation_cost,
    get_production,
    get_production_factor,
    get_social_welfare,
    get_utility,
    set_rice_params,
)

# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)

_FEATURES = "features"
_ACTION_MASK = "action_mask"


class Rice:
    """
    TODO : write docstring for RICE
    Rice class. Includes all regions, interactions, etc.
    Is initialized based on yaml file.
    etc...
    """

    name = "Rice"

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=False,  # If True then negotiation is on, else off
        group_on=False  # If True then group negotiation is on, else off
    ):
        """TODO : init docstring"""
        assert (
            num_discrete_action_levels > 1
        ), "the number of action levels should be > 1."
        self.num_discrete_action_levels = num_discrete_action_levels
        self.negotiation_on = negotiation_on
        self.group_on = True
        self.float_dtype = np.float32
        self.int_dtype = np.int32
        #self.group_indicator = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8]

        # Constants
        params, num_regions = set_rice_params(
            os.path.join(_PUBLIC_REPO_DIR, "region_yamls"),
        )
        # TODO : add to yaml
        self.balance_interest_rate = 0.1

        self.num_regions = num_regions
        self.num_groups = 9

        self.rice_constant = params["_RICE_CONSTANT"]
        self.dice_constant = params["_DICE_CONSTANT"]
        self.all_constants = self.concatenate_world_and_regional_params(
            self.dice_constant, self.rice_constant
        )

        # TODO: rename constans[0] to dice_constants?
        self.start_year = self.all_constants[0]["xt_0"]
        self.end_year = (
            self.start_year
            + self.all_constants[0]["xDelta"] * self.all_constants[0]["xN"]
        )

        # These will be set in reset (see below)
        self.current_year = None  # current year in the simulation
        self.timestep = None  # episode timestep
        self.activity_timestep = None  # timestep pertaining to the activity stage

        # Parameters for Armington aggregation
        # TODO : add to yaml
        self.sub_rate = 0.5
        self.dom_pref = 0.5
        self.for_pref = [0.5 / (self.num_regions - 1)] * self.num_regions

        # Typecasting
        self.sub_rate = np.array([self.sub_rate]).astype(self.float_dtype)
        self.dom_pref = np.array([self.dom_pref]).astype(self.float_dtype)
        self.for_pref = np.array(self.for_pref, dtype=self.float_dtype)

        # Define env global state
        # These will be initialized at reset (see below)
        self.global_state = {}

        # Define the episode length
        self.episode_length = self.dice_constant["xN"]

        # Defining observation and action spaces
        self.observation_space = None  # This will be set via the env_wrapper (in utils)

        # Notation nvec: vector of counts of each categorical variable
        # Each region sets mitigation and savings rates
        self.savings_action_nvec = [self.num_discrete_action_levels]
        self.mitigation_rate_action_nvec = [self.num_discrete_action_levels]
        # Each region sets max allowed export from own region
        self.export_action_nvec = [self.num_discrete_action_levels]
        # Each region sets import bids (max desired imports from other countries)
        self.import_actions_nvec = [self.num_discrete_action_levels] * self.num_regions
        # Each region sets import tariffs imposed on other countries
        self.tariff_actions_nvec = [self.num_discrete_action_levels] 

        self.actions_nvec = (
            self.savings_action_nvec
            + self.mitigation_rate_action_nvec
            + self.export_action_nvec
            + self.import_actions_nvec
            + self.tariff_actions_nvec
        )

        # Negotiation-related initializations
        if self.negotiation_on and not self.group_on:
            self.stage = 0
            self.num_negotiation_stages = 2  # proposal and evaluation steps
            self.episode_length += (
                self.dice_constant["xN"] * self.num_negotiation_stages
            )

            # Each region proposes to each other region
            # self mitigation and their mitigation values
            self.proposal_actions_nvec = (
                [self.num_discrete_action_levels] * 2 * self.num_regions
            )

            # Each region evaluates a proposal from every other region,
            # either accept or reject.
            self.evaluation_actions_nvec = [2] * self.num_regions

            self.actions_nvec += (
                self.proposal_actions_nvec + self.evaluation_actions_nvec
            )

        # Add group proposal and evaluation actions
        if self.negotiation_on and self.group_on:
            self.stage = 0
            self.num_negotiation_stages = 3  # proposal and evaluation steps, group updating step
            self.episode_length += (
                self.dice_constant["xN"] * self.num_negotiation_stages
            )
        
            self.group_dict = {0: [26, 1, 2], 1: [3, 4, 6], 2: [5, 7, 18], 3: [19, 10, 12], 
                                    4: [20, 13, 15], 5: [14, 16, 22], 6: [8, 9, 21], 7: [11, 17, 23],
                                    8: [24, 25, 0]}
            self.group_indicator = [8, 0, 0, 1, 1, 3, 1, 2, 6, 6, 3, 7, 3, 4, 5, 4, 5, 7, 2, 3, 4, 6, 5, 7, 8, 8, 0]

            self.disagreement_indicator = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            self.updating_pool = []
            
            self.group_ratio_actions_nvec = [self.num_discrete_action_levels]

            self.group_proposal_actions_nvec = (
                [self.num_discrete_action_levels] * 2 * 9 # (9 is the group number)
            )

            self.group_evaluation_actions_nvec = [2] * 9 # (9 is the group number)
# start saving 
            self.group_savings_actions_nvec = (
                [self.num_discrete_action_levels] * 2 * 9 # (9 is the group number)
            )
# end saving
            '''
            self.group_tariff_actions_nvec = (
                [self.num_discrete_action_levels] * 2 * 9 # (9 is the group number)
            )
            '''

            self.actions_nvec += (
                self.group_ratio_actions_nvec + self.group_proposal_actions_nvec + self.group_evaluation_actions_nvec + \
                self.group_savings_actions_nvec #+ self.group_tariff_actions_nvec
            )


        # Set the env action space
        self.action_space = {
            region_id: MultiDiscrete(self.actions_nvec)
            for region_id in range(self.num_regions)
        }

        # Set the default action mask (all ones)
        self.len_actions = sum(self.actions_nvec)
        self.default_agent_action_mask = np.ones(self.len_actions, dtype=self.int_dtype)

        # Add num_agents attribute (for use with WarpDrive)
        self.num_agents = self.num_regions

    def reset(self):
        """
        Reset the environment
        """
        self.timestep = 0
        self.activity_timestep = 0
        self.current_year = self.start_year

        constants = self.all_constants

        self.set_global_state(
            key="global_temperature",
            value=np.array(
                [constants[0]["xT_AT_0"], constants[0]["xT_LO_0"]],
            ),
            timestep=self.timestep,
            norm=1e1,
        )

        self.set_global_state(
            key="global_carbon_mass",
            value=np.array(
                [
                    constants[0]["xM_AT_0"],
                    constants[0]["xM_UP_0"],
                    constants[0]["xM_LO_0"],
                ],
            ),
            timestep=self.timestep,
            norm=1e4,
        )

        self.set_global_state(
            key="capital_all_regions",
            value=np.array(
                [constants[region_id]["xK_0"] for region_id in range(self.num_regions)]
            ),
            timestep=self.timestep,
            norm=1e4,
        )

        self.set_global_state(
            key="labor_all_regions",
            value=np.array(
                [constants[region_id]["xL_0"] for region_id in range(self.num_regions)]
            ),
            timestep=self.timestep,
            norm=1e4,
        )

        self.set_global_state(
            key="production_factor_all_regions",
            value=np.array(
                [constants[region_id]["xA_0"] for region_id in range(self.num_regions)]
            ),
            timestep=self.timestep,
            norm=1e2,
        )

        self.set_global_state(
            key="intensity_all_regions",
            value=np.array(
                [
                    constants[region_id]["xsigma_0"]
                    for region_id in range(self.num_regions)
                ]
            ),
            timestep=self.timestep,
            norm=1e-1,
        )

        for key in [
            "global_exogenous_emissions",
            "global_land_emissions",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros(
                    1,
                ),
                timestep=self.timestep,
            )
        self.set_global_state(
            "timestep", self.timestep, self.timestep, dtype=self.int_dtype, norm=1e2
        )
        self.set_global_state(
            "activity_timestep",
            self.activity_timestep,
            self.timestep,
            dtype=self.int_dtype,
        )

        for key in [
            "capital_depreciation_all_regions",
            "savings_all_regions",
            "mitigation_rate_all_regions",
            "max_export_limit_all_regions",
            "mitigation_cost_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "utility_all_regions",
            "social_welfare_all_regions",
            "reward_all_regions",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros(
                    self.num_regions,
                ),
                timestep=self.timestep,
            )

        for key in [
            "consumption_all_regions",
            "current_balance_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "production_all_regions",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros(
                    self.num_regions,
                ),
                timestep=self.timestep,
                norm=1e3,
            )

        for key in [
            "tariffs",
            "future_tariffs",
            "scaled_imports",
            "desired_imports",
            "tariffed_imports",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros((self.num_regions, self.num_regions)),
                timestep=self.timestep,
                norm=1e2,
            )

        # Negotiation-related features
        self.set_global_state(
            key="stage",
            value=np.zeros(1),
            timestep=self.timestep,
            dtype=self.int_dtype,
        )
        self.set_global_state(
            key="minimum_mitigation_rate_all_regions",
            value=np.zeros(self.num_regions),
            timestep=self.timestep,
        )
 ## start saving rate       
        self.set_global_state(
            key="minimum_saving_rate_all_regions",
            value=np.zeros(self.num_regions),
            timestep=self.timestep,
        )

        self.set_global_state(
            key="group_savings_promise",
            value=np.zeros((self.num_groups, self.num_groups)),
            timestep=self.timestep,
        )
        self.set_global_state(
            key="group_savings_request",
            value=np.zeros((self.num_groups, self.num_groups)),
            timestep=self.timestep,
        )
## end saving rate
        '''
        # tariff
        self.set_global_state(
            key="minimum_tariff_all_regions",
            value=np.zeros(self.num_regions),
            timestep=self.timestep,
        )

        self.set_global_state(
            key="group_tariff_promise",
            value=np.zeros((self.num_groups, self.num_groups)),
            timestep=self.timestep,
        )
        self.set_global_state(
            key="group_tariff_request",
            value=np.zeros((self.num_groups, self.num_groups)),
            timestep=self.timestep,
        )
        # end tariff
        '''

        self.set_global_state(
            key="group_disccused_ratio", 
            value=np.zeros(self.num_regions),
            timestep=self.timestep
        )

        for k in ["group_promised_mitigation_rate", "group_requested_mitigation_rate", "group_proposal_decisions"]:
            self.set_global_state(
                key=k, 
                value=np.zeros((self.num_groups, self.num_groups)),
                timestep=self.timestep
            )

        for key in [
            "promised_mitigation_rate",
            "requested_mitigation_rate",
            "proposal_decisions",
        ]:
            self.set_global_state(
                key=key,
                value=np.zeros((self.num_regions, self.num_regions)),
                timestep=self.timestep,
            )

        return self.generate_observation()

    def step(self, actions=None):
        """
        The environment step function.
        If negotiation is enabled, it also comprises
        the proposal and evaluation steps.
        """
        # Increment timestep
        self.timestep += 1

        # Carry over the previous global states to the current timestep
        for key in self.global_state:
            if key != "reward_all_regions":
                self.global_state[key]["value"][self.timestep] = self.global_state[key][
                    "value"
                ][self.timestep - 1].copy()

        self.set_global_state(
            "timestep", self.timestep, self.timestep, dtype=self.int_dtype
        )
        if self.negotiation_on:
            # Note: The '+1` below is for the climate_and_economy_simulation_step
            self.stage = self.timestep % (self.num_negotiation_stages + 1)
            self.set_global_state(
                "stage", self.stage, self.timestep, dtype=self.int_dtype
            )
            if self.group_on:
                if self.stage == 1:
                    return self.group_proposal_step(actions)

                if self.stage == 2:
                    return self.group_evaluation_step(actions)
                if self.stage ==3:
                    return self.group_updating_step(actions)
            else:
                if self.stage == 1:
                    return self.proposal_step(actions)

                if self.stage == 2:
                    return self.evaluation_step(actions)
        

        return self.climate_and_economy_simulation_step(actions)

    def generate_observation(self):
        """
        Generate observations for each agent by concatenating global, public
        and private features.
        The observations are returned as a dictionary keyed by region index.
        Each dictionary contains the features as well as the action mask.
        """
        # Observation array features

        # Global features that are observable by all regions
        global_features = [
            "global_temperature",
            "global_carbon_mass",
            "global_exogenous_emissions",
            "global_land_emissions",
            "timestep",
        ]

        # Public features that are observable by all regions
        public_features = [
            "capital_all_regions",
            "capital_depreciation_all_regions",
            "labor_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "consumption_all_regions",
            "savings_all_regions",
            "mitigation_rate_all_regions",
            "max_export_limit_all_regions",
            "current_balance_all_regions",
            "tariffs",
        ]

        # Private features that are private to each region.
        private_features = [
            "production_factor_all_regions",
            "intensity_all_regions",
            "mitigation_cost_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "production_all_regions",
            "utility_all_regions",
            "social_welfare_all_regions",
            "reward_all_regions",
        ]

        # Features concerning two regions
        bilateral_features = []

        # Negotiation-specific features
        if self.negotiation_on:
            global_features += ["stage"]

            public_features += []

            private_features += [
                "minimum_mitigation_rate_all_regions",
            ]

            bilateral_features += [
                "promised_mitigation_rate",
                "requested_mitigation_rate",
                "proposal_decisions",
            ]
        
        # New features
        # group_disccused_ratio
        # group_promised_mitigation_rate, group_requested_mitigation_rate
        # group_proposal_decisions
        # TODO: Added
        if self.group_on:
            global_features += ["stage"]

            public_features += []
# start saving 
            private_features += [
                "minimum_mitigation_rate_all_regions",
                "group_disccused_ratio",
                "minimum_saving_rate_all_regions",
                #"minimum_tariff_all_regions"
            ]

            grouping_features = [
                "group_promised_mitigation_rate",
                "group_requested_mitigation_rate",
                "group_proposal_decisions",
                "group_savings_promise",
                "group_savings_request",
                #"group_tariff_promise",
                #"group_tariff_request"
            ]
# end saving
        shared_features = np.array([])
        for feature in global_features + public_features:
            shared_features = np.append(
                shared_features,
                self.flatten_array(
                    self.global_state[feature]["value"][self.timestep]
                    / self.global_state[feature]["norm"]
                ),
            )

        # Form the feature dictionary, keyed by region_id.
        features_dict = {}
        for region_id in range(self.num_regions):
            # Add a region indicator array to the observation
            region_indicator = np.zeros(self.num_regions, dtype=self.float_dtype)
            region_indicator[region_id] = 1

            all_features = np.append(region_indicator, shared_features)

            for feature in private_features:
                assert self.global_state[feature]["value"].shape[1] == self.num_regions
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][self.timestep, region_id]
                        / self.global_state[feature]["norm"]
                    ),
                )

            for feature in bilateral_features:
                assert self.global_state[feature]["value"].shape[1] == self.num_regions
                assert self.global_state[feature]["value"].shape[2] == self.num_regions
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][self.timestep, region_id]
                        / self.global_state[feature]["norm"]
                    ),
                )
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][self.timestep, :, region_id]
                        / self.global_state[feature]["norm"]
                    ),
                )
            #features_dict[region_id] = all_features
        
            ## Group Feature Processing
            # TODO: Add group features to all features
            if self.group_on:
                group_id = self.group_indicator[region_id]
                
                group_indicator = np.zeros(self.num_groups, dtype=self.float_dtype)
                group_indicator[group_id] = 1
                all_features = np.append(all_features, group_indicator)
                
                for feature in grouping_features:
                    assert self.global_state[feature]["value"].shape[1] == self.num_groups
                    assert self.global_state[feature]["value"].shape[2] == self.num_groups
                    all_features = np.append(
                        all_features,
                        self.flatten_array(
                            self.global_state[feature]["value"][self.timestep, group_id]
                            / self.global_state[feature]["norm"]
                        ),
                    )
                    all_features = np.append(
                        all_features,
                        self.flatten_array(
                            self.global_state[feature]["value"][self.timestep, :, group_id]
                            / self.global_state[feature]["norm"]
                        ),
                    )

            features_dict[region_id] = all_features
        
        # Fetch the action mask dictionary, keyed by region_id.
        action_mask_dict = self.generate_action_mask()

        # Form the observation dictionary keyed by region id.
        obs_dict = {}
        for region_id in range(self.num_regions):
            obs_dict[region_id] = {
                _FEATURES: features_dict[region_id],
                _ACTION_MASK: action_mask_dict[region_id],
            }

        return obs_dict

    def generate_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):
            mask = self.default_agent_action_mask.copy()
            if self.negotiation_on:
                minimum_mitigation_rate = int(round(
                    self.global_state["minimum_mitigation_rate_all_regions"]["value"][
                        self.timestep, region_id
                    ]
                    * self.num_discrete_action_levels
                ))
                ## start saving mask
                minimum_saving_rate = int(round(
                    self.global_state["minimum_saving_rate_all_regions"]["value"][
                        self.timestep, region_id
                    ]
                    * self.num_discrete_action_levels
                ))
                saving_mask = np.array(
                    [0 for _ in range(minimum_saving_rate)]
                    + [
                        1
                        for _ in range(
                            self.num_discrete_action_levels - minimum_saving_rate
                        )
                    ]
                )      
                ## end saving mask   
                '''
                minimum_tariff_rate = int(round(
                    self.global_state["minimum_tariff_all_regions"]["value"][
                        self.timestep, region_id
                    ]
                    * self.num_discrete_action_levels
                ))
                tariff_mask = np.array(
                    [0 for _ in range(minimum_tariff_rate)]
                    + [
                        1
                        for _ in range(
                            self.num_discrete_action_levels - minimum_tariff_rate
                        )
                    ]
                )     
                '''
                      
                mitigation_mask = np.array(
                    [0 for _ in range(minimum_mitigation_rate)]
                    + [
                        1
                        for _ in range(
                            self.num_discrete_action_levels - minimum_mitigation_rate
                        )
                    ]
                )
                
                mask_start = sum(self.savings_action_nvec)
                #mask_mid = mask_start + sum(self.tariff_actions_nvec)
                mask_end = mask_start + sum(self.mitigation_rate_action_nvec)

                mask[mask_start:mask_end] = mitigation_mask
                #mask[mask_start:mask_mid] = tariff_mask
                mask[0:mask_start] = saving_mask
            mask_dict[region_id] = mask

        return mask_dict

    ## TODO: Added 
    def group_proposal_step(self, actions=None):
        
        assert self.negotiation_on
        assert self.group_on
        assert self.stage == 1

        assert isinstance(actions, dict)
        assert len(actions) == self.num_regions

        action_offset_index = len(
            self.savings_action_nvec
            + self.mitigation_rate_action_nvec
            + self.export_action_nvec
            + self.import_actions_nvec
            + self.tariff_actions_nvec
            + self.group_ratio_actions_nvec
        )

        num_group_proposal_actions = len(self.group_proposal_actions_nvec)

        group_ration_all_regions = [
            actions[region_id][
                (action_offset_index - len(self.group_ratio_actions_nvec)) : action_offset_index
            ]
            for group_id in self.group_dict.keys()
            for region_id in self.group_dict[group_id]
        ]

        group_ration_all_groups = []
        for region in range(0, len(group_ration_all_regions), 3):
            for i in range(3):
                group_ration_all_groups.append(sum(group_ration_all_regions[region:region+3]))

        for region in range(len(group_ration_all_regions)):
            if group_ration_all_groups[region][0] == 0:
                group_ration_all_regions[region] = 1/3
            else:
                group_ration_all_regions[region] = group_ration_all_regions[region] / group_ration_all_groups[region]

        self.set_global_state(
            "group_disccused_ratio", np.array(group_ration_all_regions).squeeze(), self.timestep
        )

        group_m1_all_regions = [
            actions[region_id][
                action_offset_index : action_offset_index + num_group_proposal_actions : 2
            ]
            / self.num_discrete_action_levels
            for group_id in self.group_dict.keys()
            for region_id in self.group_dict[group_id]
        ]

        group_m1_all_groups = [
            sum(group_m1_all_regions[region:region+3])/3 for region in range(0, len(group_m1_all_regions), 3)
        ]

        group_m2_all_regions = [
            actions[region_id][
                action_offset_index + 1 : action_offset_index + num_group_proposal_actions + 1 : 2
            ]
            / self.num_discrete_action_levels
            for group_id in self.group_dict.keys()
            for region_id in self.group_dict[group_id]
        ]

        group_m2_all_groups = [
            sum(group_m2_all_regions[region:region+3])/3 for region in range(0, len(group_m2_all_regions), 3)
        ]
    
        self.set_global_state(
            "group_promised_mitigation_rate", np.array(group_m1_all_groups), self.timestep
        )

        self.set_global_state(
            "group_requested_mitigation_rate", np.array(group_m2_all_groups), self.timestep
        )

        # ## start of saving rates
        action_offset_index += len(self.group_savings_actions_nvec)
        num_group_savings_actions = len(self.group_savings_actions_nvec)
        group_saving_promise_all_regions = [
            actions[region_id][
                action_offset_index : action_offset_index +  num_group_savings_actions:2
            ]
            / self.num_discrete_action_levels
            for group_id in self.group_dict.keys()
            for region_id in self.group_dict[group_id]
        ]
        group_savings_promise_all_groups = [
            sum(group_saving_promise_all_regions[region:region+3])/3 for region in range(0, len(group_saving_promise_all_regions), 3)
        ]
        group_saving_request_all_regions = [
            actions[region_id][
                action_offset_index + 1 : action_offset_index + num_group_savings_actions + 1:2
            ]
            / self.num_discrete_action_levels
            for group_id in self.group_dict.keys()
            for region_id in self.group_dict[group_id]
        ]
        group_savings_request_all_groups = [
            sum(group_saving_request_all_regions[region:region+3])/3 for region in range(0, len(group_saving_request_all_regions), 3)
        ]
        self.set_global_state(
            "group_savings_promise", np.array(group_savings_promise_all_groups), self.timestep
        )
        self.set_global_state(
            "group_savings_request", np.array(group_savings_request_all_groups), self.timestep
        )
        # ## end of saving rates
        '''
        # tariff
        action_offset_index += len(self.group_tariff_actions_nvec)
        num_group_tariff_actions = len(self.group_tariff_actions_nvec)
        group_tariff_promise_all_regions = [
            actions[region_id][
                action_offset_index : action_offset_index +  num_group_tariff_actions:2
            ]
            / self.num_discrete_action_levels
            for group_id in self.group_dict.keys()
            for region_id in self.group_dict[group_id]
        ]

        group_tariff_promise_all_groups = [
            sum(group_tariff_promise_all_regions[region:region+3])/3 for region in range(0, len(group_tariff_promise_all_regions), 3)
        ]
        group_tariff_request_all_regions = [
            actions[region_id][
                action_offset_index + 1 : action_offset_index + num_group_tariff_actions + 1:2
            ]
            / self.num_discrete_action_levels
            for group_id in self.group_dict.keys()
            for region_id in self.group_dict[group_id]
        ]
        group_tariff_request_all_groups = [
            sum(group_tariff_request_all_regions[region:region+3])/3 for region in range(0, len(group_tariff_request_all_regions), 3)
        ]
        self.set_global_state(
            "group_tariff_promise", np.array(group_tariff_promise_all_groups), self.timestep
        )
        self.set_global_state(
            "group_tariff_request", np.array(group_tariff_request_all_groups), self.timestep
        )
        '''

        obs = self.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info
    
    def group_evaluation_step(self, actions=None):
        assert self.negotiation_on
        assert self.group_on
        assert self.stage == 2

        assert isinstance(actions, dict)
        assert len(actions) == self.num_regions

        action_offset_index = len(
            self.savings_action_nvec
            + self.mitigation_rate_action_nvec
            + self.export_action_nvec
            + self.import_actions_nvec
            + self.tariff_actions_nvec
            + self.group_ratio_actions_nvec
            + self.group_proposal_actions_nvec
        )

        num_group_evaluation_actions = len(self.group_evaluation_actions_nvec)

        ## TODO: Error Action bound 0-2, but currently 0-10
        group_m1_all_regions = np.array(
            [
                actions[region_id][
                    action_offset_index : action_offset_index + num_group_evaluation_actions
                ]
                for group_id in self.group_dict.keys()
                for region_id in self.group_dict[group_id]
            ]
        )

        group_proposal_decisions = []
        for region in range(0, len(group_m1_all_regions), 3):
            lst = []
            vertical_sum = sum(group_m1_all_regions[region:region+3])
            group_decisin = 0
            for value in vertical_sum:
                if value > 1:
                    lst.append(1)
                    # Disagreement record
                    for shift in range(3):
                        if group_m1_all_regions[region+shift][group_decisin] == 0:
                            self.disagreement_indicator[region+shift] += 1
                else: 
                    lst.append(0)
                    # Disagreement record
                    for shift in range(3):
                        if group_m1_all_regions[region+shift][group_decisin] == 1:
                            self.disagreement_indicator[region+shift] += 1
                group_decisin += 1

            group_proposal_decisions.append(lst)
            
        group_proposal_decisions = np.array(group_proposal_decisions)

        # group_proposal_decisions = [
        #     1 if (sum(group_m1_all_regions[region:region+3]) > 1) else 0 for region in range(0, len(group_m1_all_regions), 3)
        # ]

        for group_id in range(self.num_groups):
            group_proposal_decisions[group_id, group_id] = 0
        
        self.set_global_state("group_proposal_decisions", group_proposal_decisions, self.timestep)

        for group_id in range(self.num_groups):
        
            for region_id in self.group_dict[group_id]:
            
                outgoing_accepted_mitigation_rates = [
                    self.global_state["group_promised_mitigation_rate"]["value"][
                        self.timestep, group_id, j
                    ]
                    * self.global_state["group_proposal_decisions"]["value"][
                        self.timestep, j, group_id
                    ]
                    for j in range(self.num_groups)
                ]
                incoming_accepted_mitigation_rates = [
                    self.global_state["group_requested_mitigation_rate"]["value"][
                        self.timestep, j, group_id
                    ]
                    * self.global_state["group_proposal_decisions"]["value"][
                        self.timestep, group_id, j
                    ]
                    for j in range(self.num_groups)
                ]

                ratio = min(1, self.global_state["group_disccused_ratio"]["value"][self.timestep, region_id] * 3)
                #print(outgoing_accepted_mitigation_rates, incoming_accepted_mitigation_rates, ratio)
                
                self.global_state["minimum_mitigation_rate_all_regions"]["value"][
                    self.timestep, region_id
                ] = max(
                    outgoing_accepted_mitigation_rates + incoming_accepted_mitigation_rates
                ) * ratio 
                #print(self.global_state["minimum_mitigation_rate_all_regions"]["value"][self.timestep, region_id])
# start to modify
        for group_id in range(self.num_groups):
        
            for region_id in self.group_dict[group_id]:
            
                outgoing_accepted_saving_rates = [
                    self.global_state["group_savings_promise"]["value"][
                        self.timestep, group_id, j
                    ]
                    * self.global_state["group_proposal_decisions"]["value"][
                        self.timestep, j, group_id
                    ]
                    for j in range(self.num_groups)
                ]
                incoming_accepted_saving_rates = [
                    self.global_state["group_savings_request"]["value"][
                        self.timestep, j, group_id
                    ]
                    * self.global_state["group_proposal_decisions"]["value"][
                        self.timestep, group_id, j
                    ]
                    for j in range(self.num_groups)
                ]
                
                self.global_state["minimum_saving_rate_all_regions"]["value"][
                    self.timestep, region_id
                ] = max(
                    outgoing_accepted_saving_rates + incoming_accepted_saving_rates
                ) * ratio 

        # ## end of saving rates
        '''
        for group_id in range(self.num_groups):
            for region_id in self.group_dict[group_id]:
                outgoing_accepted_tariff = [
                    self.global_state["group_tariff_promise"]["value"][
                        self.timestep, group_id, j
                    ]
                    * self.global_state["group_proposal_decisions"]["value"][
                        self.timestep, j, group_id
                    ]
                    for j in range(self.num_groups)
                ]
                incoming_accepted_tariff = [
                    self.global_state["group_tariff_request"]["value"][
                        self.timestep, j, group_id
                    ]
                    * self.global_state["group_proposal_decisions"]["value"][
                        self.timestep, group_id, j
                    ]
                    for j in range(self.num_groups)
                ]
                
                self.global_state["minimum_saving_rate_all_regions"]["value"][
                    self.timestep, region_id
                ] = max(
                    outgoing_accepted_tariff + incoming_accepted_tariff
                ) * ratio 
        '''

        obs = self.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info
    
    def group_updating_step(self, actions=None):
        assert self.negotiation_on
        assert self.group_on
        assert self.stage == 3

        for region_id in range(self.num_regions):
            if self.disagreement_indicator[region_id] > 18:
                self.updating_pool.append(region_id)
        
        for region_i in self.updating_pool:
            capital_i = self.get_global_state(
                "capital_all_regions", timestep=self.timestep - 1, region_id=region_i
            )
            labor_i = self.get_global_state(
                "labor_all_regions", timestep=self.timestep - 1, region_id=region_i
            )
            for region_j in self.updating_pool:
                if region_i != region_j:
                    capital_j = self.get_global_state(
                        "capital_all_regions", timestep=self.timestep - 1, region_id=region_j
                    )
                    labor_j = self.get_global_state(
                        "labor_all_regions", timestep=self.timestep - 1, region_id=region_j
                    )

                    distance = (capital_i - capital_j) ** 2 + (labor_i - labor_j) ** 2
                    if distance < 50:
                        self.group_indicator[region_i], self.group_indicator[region_j] = self.group_indicator[region_j], self.group_indicator[region_i]
                        for group in self.group_dict:
                            if region_i in self.group_dict[group]:
                                self.group_dict[group].append(region_j)
                                self.group_dict[group].remove(region_i)
                            if region_j in self.group_dict[group]:
                                self.group_dict[group].append(region_i)
                                self.group_dict[group].remove(region_j)


        obs = self.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info


    def proposal_step(self, actions=None):
        """
        Update Proposal States and Observations using proposal actions
        Update Stage to 1 - Evaluation
        """
        assert self.negotiation_on
        assert self.stage == 1

        assert isinstance(actions, dict)
        assert len(actions) == self.num_regions

        action_offset_index = len(
            self.savings_action_nvec
            + self.mitigation_rate_action_nvec
            + self.export_action_nvec
            + self.import_actions_nvec
            + self.tariff_actions_nvec
        )
        num_proposal_actions = len(self.proposal_actions_nvec)

        m1_all_regions = [
            actions[region_id][
                action_offset_index : action_offset_index + num_proposal_actions : 2
            ]
            / self.num_discrete_action_levels
            for region_id in range(self.num_regions)
        ]

        m2_all_regions = [
            actions[region_id][
                action_offset_index + 1 : action_offset_index + num_proposal_actions : 2
            ]
            / self.num_discrete_action_levels
            for region_id in range(self.num_regions)
        ]

        self.set_global_state(
            "promised_mitigation_rate", np.array(m1_all_regions), self.timestep
        )
        self.set_global_state(
            "requested_mitigation_rate", np.array(m2_all_regions), self.timestep
        )

        obs = self.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info

    def evaluation_step(self, actions=None):
        """
        Update minimum mitigation rates
        """
        assert self.negotiation_on
        assert self.stage == 2

        assert isinstance(actions, dict)
        assert len(actions) == self.num_regions

        action_offset_index = len(
            self.savings_action_nvec
            + self.mitigation_rate_action_nvec
            + self.export_action_nvec
            + self.import_actions_nvec
            + self.tariff_actions_nvec
            + self.proposal_actions_nvec
        )
        num_evaluation_actions = len(self.evaluation_actions_nvec)

        proposal_decisions = np.array(
            [
                actions[region_id][
                    action_offset_index : action_offset_index + num_evaluation_actions
                ]
                for region_id in range(self.num_regions)
            ]
        )
        # Force set the evaluation for own proposal to reject
        for region_id in range(self.num_regions):
            proposal_decisions[region_id, region_id] = 0

        self.set_global_state("proposal_decisions", proposal_decisions, self.timestep)

        for region_id in range(self.num_regions):
            outgoing_accepted_mitigation_rates = [
                self.global_state["promised_mitigation_rate"]["value"][
                    self.timestep, region_id, j
                ]
                * self.global_state["proposal_decisions"]["value"][
                    self.timestep, j, region_id
                ]
                for j in range(self.num_regions)
            ]
            incoming_accepted_mitigation_rates = [
                self.global_state["requested_mitigation_rate"]["value"][
                    self.timestep, j, region_id
                ]
                * self.global_state["proposal_decisions"]["value"][
                    self.timestep, region_id, j
                ]
                for j in range(self.num_regions)
            ]

            self.global_state["minimum_mitigation_rate_all_regions"]["value"][
                self.timestep, region_id
            ] = max(
                outgoing_accepted_mitigation_rates + incoming_accepted_mitigation_rates
            )
            #print(self.global_state["minimum_mitigation_rate_all_regions"]["value"][self.timestep, region_id])

        obs = self.generate_observation()
        rew = {region_id: 0.0 for region_id in range(self.num_regions)}
        done = {"__all__": 0}
        info = {}
        return obs, rew, done, info

    def climate_and_economy_simulation_step(self, actions=None):
        """
        The step function for the climate and economy simulation.
        PLEASE DO NOT MODIFY THE CODE BELOW.
        These functions dictate the dynamics of the climate and economy simulation,
        and should not be altered hence.
        """
        self.activity_timestep += 1
        self.set_global_state(
            key="activity_timestep",
            value=self.activity_timestep,
            timestep=self.timestep,
            dtype=self.int_dtype,
        )

        if self.negotiation_on:
            assert self.stage == 0
        else:
            assert self.timestep == self.activity_timestep

        assert isinstance(actions, dict)
        assert len(actions) == self.num_regions

        # add actions to global state
        savings_action_index = 0
        mitigation_rate_action_index = savings_action_index + len(
            self.savings_action_nvec
        )
        export_action_index = mitigation_rate_action_index + len(
            self.mitigation_rate_action_nvec
        )
        tariffs_action_index = export_action_index + len(self.export_action_nvec)
        desired_imports_action_index = tariffs_action_index + len(
            self.tariff_actions_nvec
        )

        self.set_global_state(
            "savings_all_regions",
            [
                actions[region_id][savings_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ],
            self.timestep,
        )
        self.set_global_state(
            "mitigation_rate_all_regions",
            [
                actions[region_id][mitigation_rate_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ],
            self.timestep,
        )
        self.set_global_state(
            "max_export_limit_all_regions",
            [
                actions[region_id][export_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ],
            self.timestep,
        )
        self.set_global_state(
            "future_tariffs",
            [
                actions[region_id][
                    tariffs_action_index : tariffs_action_index + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ],
            self.timestep,
        )
        self.set_global_state(
            "desired_imports",
            [
                actions[region_id][
                    desired_imports_action_index : desired_imports_action_index
                    + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ],
            self.timestep,
        )

        # Constants
        constants = self.all_constants

        const = constants[0]
        aux_m_all_regions = np.zeros(self.num_regions, dtype=self.float_dtype)

        prev_global_temperature = self.get_global_state(
            "global_temperature", self.timestep - 1
        )
        t_at = prev_global_temperature[0]

        # add emissions to global state
        global_exogenous_emissions = get_exogenous_emissions(
            const["xf_0"], const["xf_1"], const["xt_f"], self.activity_timestep
        )
        global_land_emissions = get_land_emissions(
            const["xE_L0"], const["xdelta_EL"], self.activity_timestep, self.num_regions
        )

        self.set_global_state(
            "global_exogenous_emissions", global_exogenous_emissions, self.timestep
        )
        self.set_global_state(
            "global_land_emissions", global_land_emissions, self.timestep
        )
        desired_imports = self.get_global_state("desired_imports")
        scaled_imports = self.get_global_state("scaled_imports")

        for region_id in range(self.num_regions):

            # Actions
            savings = self.get_global_state("savings_all_regions", region_id=region_id)
            mitigation_rate = self.get_global_state(
                "mitigation_rate_all_regions", region_id=region_id
            )

            # feature values from previous timestep
            intensity = self.get_global_state(
                "intensity_all_regions", timestep=self.timestep - 1, region_id=region_id
            )
            production_factor = self.get_global_state(
                "production_factor_all_regions",
                timestep=self.timestep - 1,
                region_id=region_id,
            )
            capital = self.get_global_state(
                "capital_all_regions", timestep=self.timestep - 1, region_id=region_id
            )
            labor = self.get_global_state(
                "labor_all_regions", timestep=self.timestep - 1, region_id=region_id
            )
            gov_balance_prev = self.get_global_state(
                "current_balance_all_regions",
                timestep=self.timestep - 1,
                region_id=region_id,
            )

            # constants
            const = constants[region_id]

            # climate costs and damages
            mitigation_cost = get_mitigation_cost(
                const["xp_b"],
                const["xtheta_2"],
                const["xdelta_pb"],
                self.activity_timestep,
                intensity,
            )

            damages = get_damages(t_at, const["xa_1"], const["xa_2"], const["xa_3"])
            abatement_cost = get_abatement_cost(
                mitigation_rate, mitigation_cost, const["xtheta_2"]
            )
            production = get_production(
                production_factor,
                capital,
                labor,
                const["xgamma"],
            )

            gross_output = get_gross_output(damages, abatement_cost, production)
            gov_balance_prev = gov_balance_prev * (1 + self.balance_interest_rate)
            investment = get_investment(savings, gross_output)


            for j in range(self.num_regions):
                scaled_imports[region_id][j] = (
                    desired_imports[region_id][j] * gross_output
                )
            # Import bid to self is reset to zero
            scaled_imports[region_id][region_id] = 0

            total_scaled_imports = np.sum(scaled_imports[region_id])
            if total_scaled_imports > gross_output:
                for j in range(self.num_regions):
                    scaled_imports[region_id][j] = (
                        scaled_imports[region_id][j]
                        / total_scaled_imports
                        * gross_output
                    )

            # Scale imports based on gov balance
            init_capital_multiplier = 10.0
            debt_ratio = gov_balance_prev / init_capital_multiplier * const["xK_0"]
            debt_ratio = min(0.0, debt_ratio)
            debt_ratio = max(-1.0, debt_ratio)
            debt_ratio = np.array(debt_ratio).astype(self.float_dtype)
            scaled_imports[region_id] *= 1 + debt_ratio

            self.set_global_state(
                "mitigation_cost_all_regions",
                mitigation_cost,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "damages_all_regions", damages, self.timestep, region_id=region_id
            )
            self.set_global_state(
                "abatement_cost_all_regions",
                abatement_cost,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "production_all_regions", production, self.timestep, region_id=region_id
            )
            self.set_global_state(
                "gross_output_all_regions",
                gross_output,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "current_balance_all_regions",
                gov_balance_prev,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "investment_all_regions",
                investment,
                self.timestep,
                region_id=region_id,
            )

        for region_id in range(self.num_regions):
            x_max = self.get_global_state(
                "max_export_limit_all_regions", region_id=region_id
            )
            gross_output = self.get_global_state(
                "gross_output_all_regions", region_id=region_id
            )
            investment = self.get_global_state(
                "investment_all_regions", region_id=region_id
            )

            # scale desired imports according to max exports
            max_potential_exports = get_max_potential_exports(
                x_max, gross_output, investment
            )
            total_desired_exports = np.sum(scaled_imports[:, region_id])

            if total_desired_exports > max_potential_exports:
                for j in range(self.num_regions):
                    scaled_imports[j][region_id] = (
                        scaled_imports[j][region_id]
                        / total_desired_exports
                        * max_potential_exports
                    )

        self.set_global_state("scaled_imports", scaled_imports, self.timestep)

        # countries with negative gross output cannot import
        prev_tariffs = self.get_global_state(
            "future_tariffs", timestep=self.timestep - 1
        )
        tariffed_imports = self.get_global_state("tariffed_imports")
        scaled_imports = self.get_global_state("scaled_imports")

        for region_id in range(self.num_regions):
            # constants
            const = constants[region_id]

            # get variables from global state
            savings = self.get_global_state("savings_all_regions", region_id=region_id)
            gross_output = self.get_global_state(
                "gross_output_all_regions", region_id=region_id
            )
            investment = get_investment(savings, gross_output)
            labor = self.get_global_state(
                "labor_all_regions",
                timestep=self.timestep - 1,
                region_id=region_id,
            )

            # calculate tariffed imports, tariff revenue and budget balance
            for j in range(self.num_regions):
                tariffed_imports[region_id, j] = scaled_imports[region_id, j] * (
                    1 - prev_tariffs[region_id, j]
                )
            tariff_revenue = np.sum(
                scaled_imports[region_id, :] * prev_tariffs[region_id, :]
            )

            # Aggregate consumption from domestic and foreign goods
            # domestic consumption
            c_dom = get_consumption(gross_output, investment, exports=scaled_imports[:, region_id])

            consumption = get_armington_agg(
                c_dom=c_dom,
                c_for=tariffed_imports[region_id, :],  # np.array
                sub_rate=self.sub_rate,  # in (0,1)  np.array
                dom_pref=self.dom_pref,  # in [0,1]  np.array
                for_pref=self.for_pref,  # np.array, sums to (1 - dom_pref)
            )

            utility = get_utility(labor, consumption, const["xalpha"])

            social_welfare = get_social_welfare(
                utility, const["xrho"], const["xDelta"], self.activity_timestep
            )

            self.set_global_state(
                "tariff_revenue", tariff_revenue, self.timestep, region_id=region_id
            )
            self.set_global_state(
                "consumption_all_regions",
                consumption,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "utility_all_regions",
                utility,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "social_welfare_all_regions",
                social_welfare,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "reward_all_regions",
                utility,
                self.timestep,
                region_id=region_id,
            )

        # Update gov balance
        for region_id in range(self.num_regions):
            const = constants[region_id]
            gov_balance_prev = self.get_global_state(
                "current_balance_all_regions", region_id=region_id
            )
            scaled_imports = self.get_global_state("scaled_imports")

            gov_balance = gov_balance_prev + const["xDelta"] * (
                np.sum(scaled_imports[:, region_id])
                - np.sum(scaled_imports[region_id, :])
            )
            self.set_global_state(
                "current_balance_all_regions",
                gov_balance,
                self.timestep,
                region_id=region_id,
            )

        self.set_global_state(
            "tariffed_imports",
            tariffed_imports,
            self.timestep,
        )

        # Update temperature
        m_at = self.get_global_state("global_carbon_mass", timestep=self.timestep - 1)[
            0
        ]
        prev_global_temperature = self.get_global_state(
            "global_temperature", timestep=self.timestep - 1
        )

        global_exogenous_emissions = self.get_global_state(
            "global_exogenous_emissions"
        )[0]

        const = constants[0]
        global_temperature = get_global_temperature(
            np.array(const["xPhi_T"]),
            prev_global_temperature,
            const["xB_T"],
            const["xF_2x"],
            m_at,
            const["xM_AT_1750"],
            global_exogenous_emissions,
        )
        self.set_global_state(
            "global_temperature",
            global_temperature,
            self.timestep,
        )

        for region_id in range(self.num_regions):
            intensity = self.get_global_state(
                "intensity_all_regions", timestep=self.timestep - 1, region_id=region_id
            )
            mitigation_rate = self.get_global_state(
                "mitigation_rate_all_regions", region_id=region_id
            )
            production = self.get_global_state(
                "production_all_regions", region_id=region_id
            )
            land_emissions = self.get_global_state("global_land_emissions")

            aux_m = get_aux_m(
                intensity,
                mitigation_rate,
                production,
                land_emissions,
            )
            aux_m_all_regions[region_id] = aux_m

        # Update carbon mass
        const = constants[0]
        prev_global_carbon_mass = self.get_global_state(
            "global_carbon_mass", timestep=self.timestep - 1
        )
        global_carbon_mass = get_global_carbon_mass(
            const["xPhi_M"],
            prev_global_carbon_mass,
            const["xB_M"],
            np.sum(aux_m_all_regions),
        )
        self.set_global_state("global_carbon_mass", global_carbon_mass, self.timestep)

        for region_id in range(self.num_regions):
            capital = self.get_global_state(
                "capital_all_regions", timestep=self.timestep - 1, region_id=region_id
            )
            labor = self.get_global_state(
                "labor_all_regions", timestep=self.timestep - 1, region_id=region_id
            )
            production_factor = self.get_global_state(
                "production_factor_all_regions",
                timestep=self.timestep - 1,
                region_id=region_id,
            )
            intensity = self.get_global_state(
                "intensity_all_regions",
                timestep=self.timestep - 1,
                region_id=region_id,
            )
            investment = self.get_global_state(
                "investment_all_regions", timestep=self.timestep, region_id=region_id
            )

            const = constants[region_id]

            capital_depreciation = get_capital_depreciation(
                const["xdelta_K"], const["xDelta"]
            )
            updated_capital = get_capital(
                capital_depreciation, capital, const["xDelta"], investment
            )
            updated_capital = updated_capital

            updated_labor = get_labor(labor, const["xL_a"], const["xl_g"])
            updated_production_factor = get_production_factor(
                production_factor,
                const["xg_A"],
                const["xdelta_A"],
                const["xDelta"],
                self.activity_timestep,
            )
            updated_intensity = get_carbon_intensity(
                intensity,
                const["xg_sigma"],
                const["xdelta_sigma"],
                const["xDelta"],
                self.activity_timestep,
            )

            self.set_global_state(
                "capital_depreciation_all_regions",
                capital_depreciation,
                self.timestep,
            )
            self.set_global_state(
                "capital_all_regions",
                updated_capital,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "labor_all_regions",
                updated_labor,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "production_factor_all_regions",
                updated_production_factor,
                self.timestep,
                region_id=region_id,
            )
            self.set_global_state(
                "intensity_all_regions",
                updated_intensity,
                self.timestep,
                region_id=region_id,
            )

        self.set_global_state(
            "tariffs",
            self.global_state["future_tariffs"]["value"][self.timestep],
            self.timestep,
        )

        obs = self.generate_observation()
        rew = {
            region_id: self.global_state["reward_all_regions"]["value"][
                self.timestep, region_id
            ]
            for region_id in range(self.num_regions)
        }
        # Set current year
        self.current_year += self.all_constants[0]["xDelta"]
        done = {"__all__": self.current_year == self.end_year}
        info = {}
        return obs, rew, done, info

    def set_global_state(
        self, key=None, value=None, timestep=None, norm=None, region_id=None, dtype=None
    ):
        """
        Set a specific slice of the environment global state with a key and value pair.
        The value is set for a specific timestep, and optionally, a specific region_id.
        Optionally, a normalization factor (used for generating observation),
        and a datatype may also be provided.
        """
        assert key is not None
        assert value is not None
        assert timestep is not None
        if norm is None:
            norm = 1.0
        if dtype is None:
            dtype = self.float_dtype

        if isinstance(value, list):
            value = np.array(value, dtype=dtype)
        elif isinstance(value, (float, np.floating)):
            value = np.array([value], dtype=self.float_dtype)
        elif isinstance(value, (int, np.integer)):
            value = np.array([value], dtype=self.int_dtype)
        else:
            assert isinstance(value, np.ndarray)

        if key not in self.global_state:
            logging.info("Adding %s to global state.", key)
            if region_id is None:
                self.global_state[key] = {
                    "value": np.zeros(
                        (self.episode_length + 1,) + value.shape, dtype=dtype
                    ),
                    "norm": norm,
                }
            else:
                self.global_state[key] = {
                    "value": np.zeros(
                        (self.episode_length + 1,) + (self.num_regions,) + value.shape,
                        dtype=dtype,
                    ),
                    "norm": norm,
                }

        # Set the value
        if region_id is None:
            self.global_state[key]["value"][timestep] = value
        else:
            self.global_state[key]["value"][timestep, region_id] = value

    def get_global_state(self, key=None, timestep=None, region_id=None):
        assert key in self.global_state, f"Invalid key '{key}' in global state!"
        if timestep is None:
            timestep = self.timestep
        if region_id is None:
            return self.global_state[key]["value"][timestep].copy()
        return self.global_state[key]["value"][timestep, region_id].copy()

    @staticmethod
    def flatten_array(array):
        """Flatten a numpy array"""
        return np.reshape(array, -1)

    def concatenate_world_and_regional_params(self, world, regional):
        """
        This function merges the world params dict into the regional params dict.
        Inputs:
            world: global params, dict, each value is common to all regions.
            regional: region-specific params, dict,
                      length of the values should equal the num of regions.
        Outputs:
            outs: list of dicts, each dict corresponding to a region
                  and comprises the global and region-specific parameters.
        """
        vals = regional.values()
        assert all(
            len(item) == self.num_regions for item in vals
        ), "The number of regions has to be consistent!"

        outs = []
        for region_id in range(self.num_regions):
            out = world.copy()
            for key, val in regional.items():
                out[key] = val[region_id]
            outs.append(out)
        return outs
