| Skill | Preconditions | Constraints | Post-Verifier | Failure Codes | Recovery |
| --- | --- | --- | --- | --- | --- |

| approach_far | TODO | TODO | verify_target_visible | infra.navigator_unavailable, perception.no_observation | L2, — |
| close_gripper | TODO | TODO | — | infra.gripper_unavailable | L3 |
| execute_grasp | TODO | TODO | verify_grasp_success | manip.ik_fail, manip.grasp_fail, contract.verification_failed | L1, L2, — |
| finalize_target_pose | TODO | TODO | verify_pose_ready | perception.depth_localization_failed, nav.nav_blocked | L1, — |
| handover_item | TODO | TODO | — | TODO | — |
| navigate_area | TODO | TODO | — | nav.nav_blocked, nav.missing_target | L1, — |
| open_gripper | TODO | TODO | — | infra.gripper_unavailable | L3 |
| pick | TODO | TODO | — | TODO | — |
| place | TODO | TODO | — | TODO | — |
| predict_grasp_point | TODO | TODO | — | manip.zerograsp_failed, manip.ik_fail | L1, — |
| recover | TODO | TODO | — | unknown | — |
| return_home | TODO | TODO | — | TODO | — |
| rotate_scan | TODO | TODO | — | nav.rotate_failed | L1 |
| search_area | TODO | TODO | verify_target_visible | perception.no_observation, nav.rotate_failed | L1, L2 |
| vla_grasp_finish | TODO | TODO | verify_grasp_success | contract.vla_no_effect, contract.vla_policy_oob, contract.verification_failed | L2, — |
