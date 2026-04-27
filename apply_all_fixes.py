import json

with open('/Users/akshaypulla/Downloads/dealroom_s2p_v3_grpo_enhanced.ipynb') as f:
    nb = json.load(f)

# --- Cell c03: reward_fn + sanity check ---
for cell in nb['cells']:
    if cell['metadata'].get('id') == 'c03':
        cell['source'] = [
            "# Cell 3: Reward function — minimal correct GRPO reward\n"
            "_reward_log = []\n"
            "\n"
            "from deal_room_S2P.environment.minimal_grpo_reward import MinimalDealRoomReward, reward_fn\n"
            "print(f'Reward function: {reward_fn.__name__}')\n"
            "\n"
            "# Sanity check\n"
            "valid_outcome = reward_fn([\"prompt\"], [\"send_document Legal dpa | Our DPA covers GDPR obligations.\"])\n"
            "invalid_outcome = reward_fn([\"prompt\"], [\"xyzabc123 invalid format no keywords here at all\"])\n"
            "print(f'Sanity — valid: {valid_outcome[0]:.4f}  invalid: {invalid_outcome[0]:.4f}  (should be -1.0)')\n"
            "if invalid_outcome[0] != -1.0:\n"
            "    raise RuntimeError(f'Parse penalty broken! Got {invalid_outcome[0]}')\n"
            "print('Reward function ready.')\n"
        ]
        break

# --- Cell c05: updated prompt with few-shot examples + PARSE_PENALTY=0.3 ---
for cell in nb['cells']:
    if cell['metadata'].get('id') == 'c05':
        cell['source'] = [
            "# Cell 5: System prompt + dataset (strict format + few-shot examples)\n"
            "\n"
            "SYSTEM_PROMPT = '''You are an enterprise B2B software sales executive negotiating a deal.\n"
            "\n"
            "Output EXACTLY ONE action in this format — no JSON, no markdown, no explanation:\n"
            "\n"
            "  send_document <target> <doc_type> | <message>\n"
            "  direct_message <target> | <message>\n"
            "  concession <target> | <term>=<value>\n"
            "  group_proposal | <message>\n"
            "  exec_escalation | <message>\n"
            "\n"
            "Examples (output exactly like these):\n"
            "send_document Legal dpa | Our DPA covers GDPR Article 28 processor obligations.\n"
            "send_document Finance roi_model | 3-year NPV shows $2.1M return, 14-month payback.\n"
            "concession Finance | price=175000\n"
            "direct_message TechLead | Our REST API has full OpenAPI documentation and sandbox access.\n"
            "group_proposal | All stakeholders are aligned on key terms.\n"
            "exec_escalation | Ready for executive sign-off.\n"
            "\n"
            "Rules:\n"
            "- Use lowercase: send_document, not Send Document\n"
            "- Use | (pipe) as delimiter, never ###\n"
            "- Do NOT output JSON, do NOT explain, output ONLY ONE action line\n"
            "\n"
            "Valid targets: Legal, Finance, TechLead, Procurement, Operations, ExecSponsor\n"
            "Valid doc types: dpa, security_cert, roi_model, implementation_timeline, compliance_report\n"
            "'''\n"
            "\n"
            "def build_prompt(obs):\n"
            "    try:\n"
            "        situation = build_situation_prompt(obs)\n"
            "    except Exception:\n"
            "        situation = f\"Round {obs.round_number}/{obs.max_rounds}, Stage: {obs.deal_stage}\\n\"\n"
            "        engagement = obs.engagement_level or {}\n"
            "        for sid, eng in engagement.items():\n"
            "            situation += f\"{sid}: engagement={int(eng*100)}%\\n\"\n"
            "    return f\"{SYSTEM_PROMPT}\\n\\n{situation}\"\n"
            "\n"
            "def build_dataset(n_samples=80, seeds=None, tasks=None):\n"
            "    if seeds is None: seeds = list(range(42, 42 + n_samples))\n"
            "    if tasks is None: tasks = ['aligned'] * 20 + ['conflicted'] * 30 + ['hostile_acquisition'] * 30\n"
            "    pairs = []\n"
            "    for i in range(n_samples):\n"
            "        env = DealRoomV3S2P()\n"
            "        task_id = tasks[i % len(tasks)]\n"
            "        obs = env.reset(seed=seeds[i], task_id=task_id)\n"
            "        prompt = build_prompt(obs)\n"
            "        if task_id == 'hostile_acquisition':\n"
            "            pos = 'send_document Legal dpa | Our DPA covers GDPR obligations and provides full liability protection.'\n"
            "        elif task_id == 'conflicted':\n"
            "            pos = 'send_document Finance roi_model | 3-year analysis shows strong ROI with protected downside.'\n"
            "        else:\n"
            "            pos = 'send_document TechLead security_cert | ISO 27001 certification covers all customer systems.'\n"
            "        baseline = 'direct_message Legal | We are committed to this deal.'\n"
            "        pairs.append({'prompt': prompt, 'completion': pos})\n"
            "        pairs.append({'prompt': prompt, 'completion': baseline})\n"
            "        env.close()\n"
            "    from datasets import Dataset\n"
            "    return Dataset.from_list(pairs)\n"
            "\n"
            "train_dataset = build_dataset(n_samples=80)\n"
            "print(f'Dataset: {len(train_dataset)} prompt-completion pairs')\n"
            "print(f'Sample lengths: {[len(tokenizer(s[\"prompt\"]+\" \"+s[\"completion\"])) for s in train_dataset.select(range(3))]}')\n"
        ]
        break

# --- Cell c06: temperature=0.9 ---
for cell in nb['cells']:
    if cell['metadata'].get('id') == 'c06':
        cell['source'] = [
            "# Cell 6: GRPOConfig + GRPOTrainer\n"
            "\n"
            "from trl import GRPOConfig, GRPOTrainer\n"
            "\n"
            "training_args = GRPOConfig(\n"
            "    output_dir='./dealroom_s2p_grpo_v3',\n"
            "    num_train_epochs=1,\n"
            "    per_device_train_batch_size=1,\n"
            "    gradient_accumulation_steps=4,\n"
            "    max_steps=60,\n"
            "    seed=42,\n"
            "    max_prompt_length=1600,\n"
            "    max_completion_length=128,\n"
            "    num_generations=16,\n"
            "    temperature=0.9,\n"
            "    epsilon=0.2,\n"
            "    beta=0.0001,\n"
            "    learning_rate=5e-6,\n"
            "    gradient_checkpointing=True,\n"
            "    fp16=not torch.cuda.is_bf16_supported(),\n"
            "    bf16=torch.cuda.is_bf16_supported(),\n"
            "    logging_steps=3,\n"
            "    save_steps=30,\n"
            "    save_total_limit=2,\n"
            "    report_to='none',\n"
            ")\n"
            "trainer = GRPOTrainer(\n"
            "    model=model,\n"
            "    reward_funcs=reward_fn,\n"
            "    args=training_args,\n"
            "    train_dataset=train_dataset,\n"
            "    processing_class=tokenizer,\n"
            ")\n"
            "print('GRPOTrainer initialized')\n"
            "print(f'  max_steps={training_args.max_steps}')\n"
            "print(f'  num_generations={training_args.num_generations}')\n"
            "print(f'  temperature={training_args.temperature}')\n"
            "print(f'  beta={training_args.beta}')\n"
        ]
        break

with open('/Users/akshaypulla/Downloads/dealroom_s2p_v3_grpo_enhanced.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("All notebook changes applied.")
print("\nChanges made:")
print("  c03: reward_fn + sanity check")
print("  c05: Added few-shot examples + 'use lowercase' rule")
print("  c06: temperature 0.8 → 0.9")
print("  PARSE_PENALTY in minimal_grpo_reward.py: -1.0 → -0.3")