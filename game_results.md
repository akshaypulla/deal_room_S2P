# DealRoom S2P - Game Play Results

## Test Date: 2026-04-27 22:07:30

## ALIGNED Level

| Step | Action | Reward | Done | Stage | Terminal Outcome |
|------|--------|--------|------|-------|------------------|
| 1 | group_proposal | -0.0848 | False | evaluation |  |
| 2 | send_document(dpa)_to_Legal | 0.5693 | False | evaluation |  |
| 3 | send_document(security_cert)_to_TechLead | 0.7541 | False | negotiation |  |
| 4 | send_document(roi_model)_to_Finance | 0.0660 | False | negotiation |  |
| 5 | send_document(implementation_timeline)_to_Operations | -0.0427 | False | negotiation |  |
| 6 | send_document(vendor_packet)_to_Procurement | 0.4472 | False | legal_review |  |
| 7 | group_proposal_final | 0.0000 | False | ? |  |
| 8 | auto_action | 0.0000 | False | ? |  |
| 9 | auto_action | 0.0000 | False | ? |  |
| 10 | auto_action | 0.0000 | False | ? |  |
| 11 | auto_action | 0.0000 | False | ? |  |
| 12 | auto_action | 0.0000 | False | ? |  |
| 13 | auto_action | 0.0000 | False | ? |  |
| 14 | auto_action | 0.0000 | False | ? |  |
| 15 | auto_action | 0.0000 | False | ? |  |
| 16 | auto_action | 0.0000 | False | ? |  |
| 17 | auto_action | 0.0000 | False | ? |  |
| 18 | auto_action | 0.0000 | False | ? |  |
| 19 | auto_action | 0.0000 | False | ? |  |
| 20 | auto_action | 0.0000 | False | ? |  |

**Final Score**: 0.0000
**Terminal Outcome**: 

## CONFLICTED Level

| Step | Action | Reward | Done | Stage | Terminal Outcome |
|------|--------|--------|------|-------|------------------|
| 1 | group_proposal | -0.0857 | False | evaluation |  |
| 2 | send_document(dpa)_to_Legal | -0.3012 | True | evaluation | soft_veto_by_Legal |

**Final Score**: -0.3012
**Terminal Outcome**: soft_veto_by_Legal

## HOSTILE_ACQUISITION Level

| Step | Action | Reward | Done | Stage | Terminal Outcome |
|------|--------|--------|------|-------|------------------|
| 1 | group_proposal | -0.0813 | False | evaluation |  |
| 2 | send_document(dpa)_to_Legal | -0.2120 | True | evaluation | soft_veto_by_Legal |

**Final Score**: -0.2120
**Terminal Outcome**: soft_veto_by_Legal


## Summary

| Level | Final Reward | Outcome |
|-------|--------------|--------|
| aligned | 0.0000 |  |
| conflicted | -0.3012 | soft_veto_by_Legal |
| hostile_acquisition | -0.2120 | soft_veto_by_Legal |
