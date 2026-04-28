# DealRoom S2P Game Results

## Summary Table

| Level | Rounds | Final Reward | Terminal Outcome |
|-------|--------|--------------|----------------|
| Easy | 8 | 0.1612 | hard_veto::missing_final_proposal |
| Medium | 2 | -0.4297 | soft_veto_by_Legal |
| Hard | 2 | -0.3166 | soft_veto_by_Legal |

## Trajectories

### Easy (aligned) - Seed 42

| Step | Action | Target | Reward | Cumulative | Stage | Blocker(s) |
|------|--------|-------|--------|------------|-------|------------|
| 1 | send_document | Finance | 0.2711 | 0.2711 | evaluation | Legal |
| 2 | send_document | Legal | 0.1239 | 0.3950 | evaluation | - |
| 3 | send_document | TechLead | 0.6635 | 1.0586 | negotiation | - |
| 4 | send_document | Procurement | -0.0528 | 1.0057 | negotiation | Legal |
| 5 | send_document | Operations | -0.2426 | 0.7631 | negotiation | - |
| 6 | direct_message | all | 0.4432 | 1.2063 | legal_review | - |
| 7 | direct_message | Legal | -0.0533 | 1.1531 | legal_review | - |
| 8 | direct_message | Legal | -0.9919 | 0.1612 | final_approval | missing_final_proposal |

### Medium (conflicted) - Seed 64

| Step | Action | Target | Reward | Cumulative | Stage | Blocker(s) |
|------|--------|-------|--------|------------|-------|------------|
| 1 | send_document | Finance | 0.3750 | 0.3750 | evaluation | Finance, Legal |
| 2 | send_document | Legal | -0.8047 | -0.4297 | evaluation | Finance, Legal |

### Hard (hostile_acquisition) - Seed 42

| Step | Action | Target | Reward | Cumulative | Stage | Blocker(s) |
|------|--------|-------|--------|------------|-------|------------|
| 1 | send_document | Finance | 0.4903 | 0.4903 | evaluation | Finance, Legal |
| 2 | send_document | Legal | -0.8069 | -0.3166 | evaluation | Finance, Legal |


## Notes

- **Easy (aligned)**: All stakeholders start aligned. Send required documents then submit proposal.
- **Medium (conflicted)**: Finance and Legal have conflicting interests. CVaR risk preferences may trigger early veto.
- **Hard (hostile_acquisition)**: Post-acquisition scenario with high risk. CVaR vetoes are expected.
- **soft_veto_by_Legal**: Legal's CVaR risk preference triggers a veto when engagement drops - this is expected behavior for conflicted/hostile tasks.
