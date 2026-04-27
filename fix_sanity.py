import json

with open('/Users/akshaypulla/Downloads/dealroom_s2p_v3_grpo_enhanced.ipynb') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['metadata'].get('id') == 'c03':
        cell['source'] = [
            "# Cell 3: Reward function — minimal correct GRPO reward\n"
            "# Key invariants (all enforced in minimal_grpo_reward.py):\n"
            "#   1. Same base seed for ALL completions within a batch → meaningful advantage\n"
            "#   2. Slight jitter per batch (seed + 0-99) → prevents overfitting\n"
            "#   3. Hard -1.0 penalty on parse failure → parser learns correct format\n"
            "#   4. No silent fallback → no collapse loop\n"
            "#   5. 2-step rollout (0.3 discount) → credit anchored to model's action\n"
            "\n"
            "from deal_room_S2P.environment.minimal_grpo_reward import MinimalDealRoomReward, reward_fn\n"
            "print(f'Reward function: {reward_fn.__name__}')\n"
            "\n"
            "# Sanity check — validate parse penalty works\n"
            "# Valid format: should get environment reward (not -1.0)\n"
            "valid_outcome = reward_fn([\"prompt\"], [\"send_document Legal dpa | Our DPA covers GDPR obligations.\"])\n"
            "# Truly unparseable: no stakeholder name, no action keyword, no pipe — triggers PARSE_PENALTY\n"
            "invalid_outcome = reward_fn([\"prompt\"], [\"xyzabc123 invalid format no keywords here at all\"])\n"
            "print(f'Sanity check — valid output reward: {valid_outcome[0]:.4f}  (environment reward, not -1.0)')\n"
            "print(f'Sanity check — invalid output reward: {invalid_outcome[0]:.4f}  (should be -1.0 parse penalty)')\n"
            "if invalid_outcome[0] != -1.0:\n"
            "    raise RuntimeError(f'Parse penalty not working! Got {invalid_outcome[0]}, expected -1.0')\n"
            "print('Reward function ready — parse penalty verified.')\n"
        ]
        break

with open('/Users/akshaypulla/Downloads/dealroom_s2p_v3_grpo_enhanced.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Done.")