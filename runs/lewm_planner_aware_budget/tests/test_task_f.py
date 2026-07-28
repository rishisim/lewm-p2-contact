import numpy as np
import pytest

from task_f import LinearCritic, feature_matrix, validate_feature_schema

def test_feature_allowlist_and_forbidden_inputs():
    validate_feature_schema(["native_latent_goal_cost", "candidate_actions_flat"])
    with pytest.raises(ValueError):
        validate_feature_schema(["simulator_cost", "candidate_actions_flat"])

def test_candidate_local_permutation_equivariance():
    native=np.arange(4.,dtype=float); candidates=np.arange(200.,dtype=float).reshape(4,5,10)
    features=feature_matrix(native,candidates); critic=LinearCritic(np.zeros(51),np.ones(51),np.arange(51.),0.)
    order=np.array([3,1,0,2]); assert np.allclose(critic.score(features)[order],critic.score(features[order]))

def test_shape_and_finite_rejection():
    with pytest.raises(ValueError): feature_matrix(np.zeros(2),np.zeros((2,5,9)))
    with pytest.raises(ValueError): feature_matrix(np.array([0.,np.nan]),np.zeros((2,5,10)))
